#!/usr/bin/env python3
"""Ingests PDF documents from the raw data directory into a ChromaDB collection."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable

import chromadb
import pdfplumber
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
DATA_DIR = PROJECT_ROOT / "src"
DEFAULT_RAW_PDF_DIR = DATA_DIR / "pdfs"
DEFAULT_VECTOR_STORE_DIR = DATA_DIR / "vector_store"
DEFAULT_COLLECTION_NAME = "ml_image_papers"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def chunk_text(text: str) -> Iterable[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    return splitter.split_text(text)


def extract_pdf_pages(path: Path) -> Iterable[tuple[int, str]]:
    with pdfplumber.open(str(path)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                yield page_number, text


def ingest_pdf(path: Path) -> tuple[list[str], list[dict], list[str]]:
    documents: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    for page_number, text in extract_pdf_pages(path):
        page_chunks = list(chunk_text(text))
        for chunk_index, chunk in enumerate(page_chunks):
            documents.append(chunk)
            metadatas.append(
                {
                    "source": path.name,
                    "page_number": page_number,
                    "chunk_index": chunk_index,
                }
            )
            ids.append(f"{path.stem}-p{page_number}-c{chunk_index}")

    return documents, metadatas, ids


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Ingest PDF documents into ChromaDB.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_PDF_DIR,
        help="Directory containing source PDF files.",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=DEFAULT_VECTOR_STORE_DIR,
        help="Directory for the ChromaDB persistence layer.",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION_NAME,
        help="Target ChromaDB collection name.",
    )
    return parser


def main() -> None:
    parser = parse_args()
    args = parser.parse_args()

    raw_dir = args.raw_dir
    persist_dir = args.persist_dir
    collection_name = args.collection

    pdf_paths = sorted(raw_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {raw_dir}")

    client = chromadb.PersistentClient(path=str(persist_dir))
    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    collection = client.get_or_create_collection(
        name=collection_name, embedding_function=embedding_fn
    )

    if collection.count() > 0:
        collection.delete(ids=collection.get()["ids"])

    total_chunks = 0
    total_pages = 0

    for pdf_path in pdf_paths:
        documents, metadatas, ids = ingest_pdf(pdf_path)

        if not documents:
            print(f"Skipping {pdf_path.name}: no extractable text.")
            continue

        collection.add(ids=ids, documents=documents, metadatas=metadatas)
        total_chunks += len(documents)
        total_pages += len({meta["page_number"] for meta in metadatas})
        print(
            f"Ingested {len(documents)} chunks from {pdf_path.name} "
            f"({len(set(meta['page_number'] for meta in metadatas))} pages)."
        )

    print(
        f"Completed PDF ingestion: {len(pdf_paths)} files, "
        f"{total_pages} pages, {total_chunks} chunks into '{collection_name}'."
    )


if __name__ == "__main__":
    main()

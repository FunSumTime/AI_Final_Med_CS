from smolagents import Tool
from pathlib import Path
import requests

from executables.ingest_pdfs import ingest_directory, DATA_DIR, DEFAULT_VECTOR_STORE_DIR

class DownloadAndIngestPdfTool(Tool):
    name = "Download_And_Ingest_PDF_Tool"
    description = (
        "PDF-only ingest. Download a PDF for CS or MED and ingest it into the appropriate ChromaDB collection. "
        "Use ONLY when the URL clearly points to a PDF (e.g., ends with .pdf or has Content-Type application/pdf)."
    )
    inputs = {
        "topic": {"type": "string", "description": "'CS' or 'MED'"},
        "url": {"type": "string", "description": "Direct link to a PDF file"},
    }
    output_type = "string"

    def __init__(self):
        super().__init__()
        self.raw_cs_dir = DATA_DIR / "pdfs_cs"
        self.raw_med_dir = DATA_DIR / "pdfs_med"
        self.persist_dir = DEFAULT_VECTOR_STORE_DIR

        self.raw_cs_dir.mkdir(parents=True, exist_ok=True)
        self.raw_med_dir.mkdir(parents=True, exist_ok=True)

    def forward(self, topic: str, url: str) -> str:
        topic = (topic or "").upper()
        if topic not in ("CS", "MED"):
            return "ingest skipped: topic must be 'CS' or 'MED'"

        # quick check: URL shape
        if not url.lower().endswith(".pdf"):
            return "ingest skipped: URL is not a direct PDF link. Do NOT call this tool again for this URL."

        # optional: confirm content-type too
        try:
            head = requests.head(url, allow_redirects=True, timeout=10)
            ctype = (head.headers.get("Content-Type") or "").lower()
            if "pdf" not in ctype:
                return "ingest skipped: server did not report a PDF Content-Type. Use ScrapePageTool instead and do NOT retry ingestion for this URL."
        except Exception:
            # if HEAD fails, we can still try GET, but if you want to be stricter, just skip
            return "ingest skipped: could not confirm PDF. Use ScrapePageTool instead."

        # pick dir + collection
        if topic == "CS":
            raw_dir = self.raw_cs_dir
            collection_name = "CS_papers"
        else:
            raw_dir = self.raw_med_dir
            collection_name = "MED_papers"

        # 1. download
        try:
            pdf_path = self._download_pdf(url, raw_dir)
        except Exception as e:
            return f"ingest failed: download error: {e}"

        # 2. ingest
        try:
            ingest_directory(raw_dir=raw_dir, persist_dir=self.persist_dir, collection_name=collection_name)
        except Exception as e:
            # clean up the file even on ingest failure
            try:
                pdf_path.unlink(missing_ok=True)
            except Exception:
                pass
            return f"ingest failed during PDF ingestion: {e}"

        # 3. delete the downloaded PDF so we don't re-ingest it
        try:
            pdf_path.unlink(missing_ok=True)
        except Exception:
            pass

        return f"ingested_from_pdf: {pdf_path.name} into {collection_name}"

    def _download_pdf(self, url: str, raw_dir: Path) -> Path:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()

        filename = url.rstrip("/").split("/")[-1]
        if not filename.lower().endswith(".pdf"):
            filename += ".pdf"

        safe_name = filename.replace("?", "_").replace("&", "_")
        pdf_path = raw_dir / safe_name

        with open(pdf_path, "wb") as f:
            f.write(resp.content)

        return pdf_path



# tools/kb_text_ingest.py

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter

VECTOR_STORE_DIR = Path("vector_store")
CS_COLLECTION = "CS_papers"
MED_COLLECTION = "MED_papers"

class IngestHtmlTextTool(Tool):
    """
    Ingest plain text (from an HTML page) into the CS or MED Chroma collection.
    Use this when ScrapePageTool has already extracted text.
    """
    name = "Ingest_Html_Text_Tool"
    description = (
        "Ingest cleaned text into the CS or MED knowledge base. "
        "Inputs: topic('CS'|'MED'), title, text."
    )
    inputs = {
        "topic": {"type": "string", "description": "'CS' or 'MED'"},
        "title": {"type": "string", "description": "Short title for this source"},
        "text":  {"type": "string", "description": "Cleaned text content to ingest"},
    }
    output_type = "string"

    def __init__(self):
        super().__init__()
        # get the connection to chroma db
        self.client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
        self.embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.cs_col = self.client.get_or_create_collection(
            name=CS_COLLECTION,
            embedding_function=self.embedding_fn,
        )
        self.med_col = self.client.get_or_create_collection(
            name=MED_COLLECTION,
            embedding_function=self.embedding_fn,
        )

    def forward(self, topic: str, title: str, text: str) -> str:
        topic = (topic or "").upper()
        if topic not in ("CS", "MED"):
            return "ingest skipped: topic must be 'CS' or 'MED'"

        if not text or len(text.strip()) < 400:
            return "ingest skipped: text too short (<400 chars)"

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " "],
        )
        chunks = splitter.split_text(text)

        if not chunks:
            return "ingest skipped: no chunks produced"

        collection = self.cs_col if topic == "CS" else self.med_col

        # simple IDs; could add hashing later
        base_id = title.replace(" ", "_")[:40] or "html_source"
        ids = [f"{base_id}-{i}" for i in range(len(chunks))]
        metadatas = [{"source": title, "topic": topic, "kind": "html"} for _ in chunks]

        collection.add(ids=ids, documents=chunks, metadatas=metadatas)
        return f"ingested_from_html: {len(chunks)} chunks into {topic} KB"

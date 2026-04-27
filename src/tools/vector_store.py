"""ChromaDB-backed retrieval utilities."""

from pathlib import Path
from typing import List, Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


class ChromaRetriever:
    """Thin wrapper around a ChromaDB collection that supports semantic search."""

    def __init__(
        self,
        persist_directory: Path,
        collection_name: str,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_results: int = 4,
    ) -> None:
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.max_results = max_results

        self._client = chromadb.PersistentClient(path=str(self.persist_directory))
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=SentenceTransformerEmbeddingFunction(
                model_name=embedding_model
            ),
        )

    def retrieve(self, query: str, limit: Optional[int] = None) -> List[dict]:
        """Return the top-matching documents with metadata.

        Args:
            query: natural-language search string.
            limit: optional override of default max_results.
        """
        if not query:
            raise ValueError("Query must be a non-empty string.")

        n_results = limit or self.max_results
        response = self._collection.query(query_texts=[query], n_results=n_results)
        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]

        results = []
        for idx, doc in enumerate(documents):
            results.append(
                {
                    "document": doc,
                    "metadata": metadatas[idx] if idx < len(metadatas) else {},
                    "distance": distances[idx] if idx < len(distances) else None,
                }
            )
        return results

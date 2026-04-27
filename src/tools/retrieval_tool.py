"""Smolagents Tool wrapper for the Chroma retriever."""

from typing import List

from smolagents import Tool

from .vector_store import ChromaRetriever


class RetrieveCsDocumentsTool(Tool):
    name = "retrieve_cs_documents"
    description = (
        "Use semantic search over the Computer Science dataset to gather relevant evidence."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "Computer Science question or claim to search for.",
        },
        "top_k": {
            "type": "integer",
            "description": "Maximum number of documents to return.",
            "default": 4,
            "nullable": True,
        },
    }
    output_type = "array"

    def __init__(self, retriever: ChromaRetriever) -> None:
        super().__init__()
        self.retriever = retriever

    def forward(self, query: str, top_k: int = 6) -> List[dict]:
        """Return retrieved documents with metadata and distance scores."""
        return self.retriever.retrieve(query=query, limit=top_k)



class RetrieveMedDocumentsTool(Tool):
    name = "retrieve_med_documents"
    description = (
        "Use semantic search over the Medical dataset to gather relevant evidence."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "Medical question or claim to search for.",
        },
        "top_k": {
            "type": "integer",
            "description": "Maximum number of documents to return.",
            "default": 4,
            "nullable": True,
        },
    }
    output_type = "array"

    def __init__(self, retriever: ChromaRetriever) -> None:
        super().__init__()
        self.retriever = retriever

    def forward(self, query: str, top_k: int = 4) -> List[dict]:
        """Return retrieved documents with metadata and distance scores."""
        return self.retriever.retrieve(query=query, limit=top_k)

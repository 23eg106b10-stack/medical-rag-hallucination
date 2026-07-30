"""Schema for the BM25 index metadata record (M3.1.2)."""

from __future__ import annotations

from pydantic import BaseModel


class BM25Metadata(BaseModel):
    """Metadata record accompanying the BM25 index (``bm25_metadata.json``)."""

    bm25_version: str
    corpus_version: str
    document_count: int
    tokenizer: str
    algorithm: str
    library: str
    created_at: str

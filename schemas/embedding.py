"""Schema for the dense (FAISS) index metadata record (M3.1.3)."""

from __future__ import annotations

from pydantic import BaseModel


class EmbeddingMetadata(BaseModel):
    """Metadata record accompanying the FAISS index (``embedding_metadata.json``).

    Includes ``corpus_checksum``, copied verbatim from ``corpus.meta.json``
    at build time, per the M3.1.3 corpus/index integrity contract - future
    retrieval components verify this against the corpus they load before
    trusting this index.
    """

    embedding_version: str
    corpus_version: str
    corpus_checksum: str
    model_name: str
    embedding_dimension: int
    embedding_dtype: str
    faiss_index_type: str
    normalize_embeddings: bool
    document_count: int
    created_at: str

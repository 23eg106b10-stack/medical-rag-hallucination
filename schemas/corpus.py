"""Schemas for the frozen retrieval corpus (Milestone 3, M3.1.1).

Kept separate from ``schemas.common`` because these types are specific to
corpus construction and are not shared with retrieval/generation request or
response shapes (see M3.1.1 architecture decision on schema placement).
"""

from __future__ import annotations

from pydantic import BaseModel


class CorpusDocument(BaseModel):
    """A single validated document in the frozen retrieval corpus.

    Mirrors the canonical document schema fixed in the M3.1.1 architecture:
    no score, embedding, or ranking fields — those belong to later
    milestones.
    """

    pmid: str
    title: str
    abstract: str


class CorpusMetadata(BaseModel):
    """Metadata record accompanying the frozen corpus (``corpus.meta.json``).

    Exists to let downstream components (BM25/FAISS builders, evaluation)
    verify they are indexing the exact corpus produced during a given run,
    per the M3.1.1 corpus versioning requirement.
    """

    corpus_version: str
    generated_at: str
    document_count: int
    checksum_sha256: str

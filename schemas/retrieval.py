"""Schemas for hybrid retrieval (Milestone 3.2).

Kept separate from ``schemas.corpus`` and ``schemas.common`` because these
types carry retrieval-specific ranking/fusion metadata (raw sparse/dense
scores, RRF score) that has no place on the frozen ``CorpusDocument`` or a
generic single-score result shape, per the M3.2 architecture decision to
introduce dedicated retrieval schemas rather than overload existing ones.
"""

from __future__ import annotations

from pydantic import BaseModel

from schemas.corpus import CorpusDocument


class RetrievalHit(BaseModel):
    """A single scored result from one retriever (BM25 or dense), pre-fusion.

    ``rank`` is assigned by the retriever itself after deterministic
    sorting (score descending, PMID ascending) — it is never inferred
    downstream from list or dict ordering, per the M3.2 determinism
    contract.
    """

    pmid: str
    score: float
    rank: int


class FusionResult(BaseModel):
    """A single document's fused ranking after Reciprocal Rank Fusion.

    ``sparse_score`` / ``dense_score`` are ``None`` when the document did
    not appear in that retriever's Top-20 pool, per the M3.2
    no-normalization / raw-score-passthrough decision.
    """

    pmid: str
    rrf_score: float
    sparse_score: float | None
    dense_score: float | None


class HybridScoredDocument(BaseModel):
    """A fully resolved retrieval result: fused scores plus corpus text."""

    document: CorpusDocument
    sparse_score: float | None
    dense_score: float | None
    rrf_score: float

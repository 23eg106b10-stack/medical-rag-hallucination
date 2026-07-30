"""Reciprocal Rank Fusion (M3.2).

Pure functions only — no I/O, no corpus access. Operates solely on
``RetrievalHit`` metadata already produced by the sparse/dense
retrievers, per the M3.2 architecture (fusion is corpus-independent).
"""

from __future__ import annotations

from schemas.retrieval import FusionResult, RetrievalHit


def reciprocal_rank_fusion(
    sparse_hits: list[RetrievalHit],
    dense_hits: list[RetrievalHit],
    k: int = 60,
) -> list[FusionResult]:
    """Fuse sparse and dense hit lists via Reciprocal Rank Fusion.

    Args:
        sparse_hits: BM25 hits, already ranked (rank 1 = best) with
            deterministic tie-breaking applied upstream.
        dense_hits: Dense hits, already ranked (rank 1 = best) with
            deterministic tie-breaking applied upstream.
        k: RRF constant (frozen at 60 per M3.2 architecture).

    Returns:
        Fused results already sorted by ``rrf_score`` descending, with
        PMID ascending as a deterministic tie-break. Every PMID present
        in either input list appears exactly once (duplicates merged).
        Callers may safely slice this list directly (e.g. ``[:5]``)
        without re-sorting.
    """
    sparse_by_pmid = {hit.pmid: hit for hit in sparse_hits}
    dense_by_pmid = {hit.pmid: hit for hit in dense_hits}

    all_pmids = set(sparse_by_pmid) | set(dense_by_pmid)

    results: list[FusionResult] = []
    for pmid in all_pmids:
        sparse_hit = sparse_by_pmid.get(pmid)
        dense_hit = dense_by_pmid.get(pmid)

        rrf_score = 0.0
        if sparse_hit is not None:
            rrf_score += 1.0 / (k + sparse_hit.rank)
        if dense_hit is not None:
            rrf_score += 1.0 / (k + dense_hit.rank)

        results.append(
            FusionResult(
                pmid=pmid,
                rrf_score=rrf_score,
                sparse_score=sparse_hit.score if sparse_hit else None,
                dense_score=dense_hit.score if dense_hit else None,
            )
        )

    results.sort(key=lambda r: (-r.rrf_score, r.pmid))
    return results

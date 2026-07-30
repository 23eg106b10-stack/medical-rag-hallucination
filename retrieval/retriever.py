"""Hybrid retrieval orchestration (M3.2).

``HybridRetriever`` is a pure orchestrator: it holds no loading logic of
its own and depends only on already-constructed sparse/dense retrievers
and corpus reader, injected at construction time, per the M3.2
dependency injection decision.
"""

from __future__ import annotations

import logging

from retrieval.corpus import InMemoryCorpusReader
from retrieval.dense import DenseRetriever
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.sparse import BM25Retriever
from schemas.retrieval import HybridScoredDocument

logger = logging.getLogger(__name__)

_SPARSE_TOP_K = 20
_DENSE_TOP_K = 20
_RRF_K = 60
_FINAL_TOP_N = 5


class HybridRetriever:
    """Orchestrates sparse + dense retrieval, RRF fusion, and corpus lookup."""

    def __init__(
        self,
        sparse_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        corpus_reader: InMemoryCorpusReader,
    ) -> None:
        self._sparse_retriever = sparse_retriever
        self._dense_retriever = dense_retriever
        self._corpus_reader = corpus_reader

    def search(self, query: str) -> list[HybridScoredDocument]:
        """Run hybrid retrieval for a query.

        Args:
            query: Raw query text.

        Returns:
            Up to ``_FINAL_TOP_N`` fused documents, sorted by RRF score
            descending. Fewer are returned (with a logged warning) if
            fusion produces fewer than ``_FINAL_TOP_N`` unique PMIDs — no
            exception is raised in that case, per the M3.2 contract.
        """
        sparse_hits = self._sparse_retriever.search(query, top_k=_SPARSE_TOP_K)
        dense_hits = self._dense_retriever.search(query, top_k=_DENSE_TOP_K)

        fused = reciprocal_rank_fusion(sparse_hits, dense_hits, k=_RRF_K)
        top_fused = fused[:_FINAL_TOP_N]

        if len(top_fused) < _FINAL_TOP_N:
            logger.warning(
                "Only %d unique fused documents available (requested %d) for query: %r",
                len(top_fused),
                _FINAL_TOP_N,
                query,
            )

        pmids = [result.pmid for result in top_fused]
        # Safe to zip positionally: InMemoryCorpusReader.get_documents is
        # contractually guaranteed to preserve input order and length, or
        # raise — never to silently shorten its result.
        documents = self._corpus_reader.get_documents(pmids)

        return [
            HybridScoredDocument(
                document=document,
                sparse_score=result.sparse_score,
                dense_score=result.dense_score,
                rrf_score=result.rrf_score,
            )
            for document, result in zip(documents, top_fused)
        ]

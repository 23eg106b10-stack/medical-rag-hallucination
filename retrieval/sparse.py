"""BM25 sparse retrieval (M3.2).

Queries the frozen M3.1.2 BM25 index. Owns rank assignment: ranks are
derived here via deterministic sorting (score descending, PMID ascending)
and stored on each ``RetrievalHit`` — fusion never re-derives rank from
list/dict ordering, per the M3.2 determinism contract.

The tokenizer is accepted as an injected callable rather than imported
directly from ``index.bm25_tokenizer``, mirroring the exact dependency
pattern already used by ``index.bm25_pipeline.build_index``. This keeps
the retriever correct-by-construction against whatever tokenizer produced
the frozen index, without this module needing to know that module's
import path or function name.
"""

from __future__ import annotations

import pickle
from collections.abc import Callable
from pathlib import Path

from rank_bm25 import BM25Okapi

from schemas.retrieval import RetrievalHit


class BM25Retriever:
    """Wraps a loaded BM25Okapi model plus its PMID ordering."""

    def __init__(
        self,
        model: BM25Okapi,
        pmids: list[str],
        tokenizer: Callable[[str], list[str]],
    ) -> None:
        """
        Args:
            model: A fitted ``BM25Okapi`` model (M3.1.2 output).
            pmids: PMIDs in the exact same order as the model's internal
                document positions (as embedded in the M3.1.2 pickle).
            tokenizer: The same tokenizer function used to build the
                index. Must be sourced from ``index.bm25_tokenizer`` by
                the caller — this class does not import it directly.
        """
        self._model = model
        self._pmids = pmids
        self._tokenizer = tokenizer

    @classmethod
    def from_pickle(cls, index_path: Path, tokenizer: Callable[[str], list[str]]) -> BM25Retriever:
        """Load a BM25Retriever from the M3.1.2 pickle artifact.

        Args:
            index_path: Path to the BM25 index file (M3.1.2 output). The
                path's extension is not authoritative — the serialized
                payload is always a pickle dict of
                ``{"model": BM25Okapi, "pmids": [...]}``, per
                ``index.bm25_pipeline.serialize_index``, regardless of
                what ``settings.bm25_index_filename`` names it.
            tokenizer: The tokenizer function used at build time.

        Returns:
            A populated ``BM25Retriever``.

        Raises:
            FileNotFoundError: if ``index_path`` does not exist.
        """
        if not index_path.exists():
            raise FileNotFoundError(f"BM25 index not found: {index_path}")

        with index_path.open("rb") as f:
            payload = pickle.load(f)

        return cls(model=payload["model"], pmids=payload["pmids"], tokenizer=tokenizer)

    def search(self, query: str, top_k: int) -> list[RetrievalHit]:
        """Return the top-k BM25 hits for a query, deterministically ranked.

        Args:
            query: Raw query text.
            top_k: Maximum number of hits to return.

        Returns:
            Hits sorted by score descending, PMID ascending on ties, with
            ``rank`` assigned 1-indexed after that sort.
        """
        tokens = self._tokenizer(query)
        scores = self._model.get_scores(tokens)

        scored_pmids = list(zip(self._pmids, scores))
        scored_pmids.sort(key=lambda pair: (-pair[1], pair[0]))

        top = scored_pmids[:top_k]
        return [
            RetrievalHit(pmid=pmid, score=float(score), rank=i + 1)
            for i, (pmid, score) in enumerate(top)
        ]

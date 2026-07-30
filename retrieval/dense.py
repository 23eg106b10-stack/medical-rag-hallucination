"""Dense (MedCPT + FAISS) retrieval (M3.2).

Queries the frozen M3.1.3 FAISS ``IndexFlatIP`` using the MedCPT query
encoder. Owns rank assignment identically to ``BM25Retriever``: ranks are
derived here via deterministic sorting (score descending, PMID
ascending), never inferred downstream.

Resolved per ACR-001 (M3.1.3 architecture change request): the FAISS
pipeline now emits ``faiss_pmids.json`` alongside ``faiss_index.bin`` and
``embedding_metadata.json``, giving an explicit, corpus-independent
mapping from vector position to PMID. ``DenseRetriever.from_disk`` loads
all three.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from schemas.embedding import EmbeddingMetadata
from schemas.retrieval import RetrievalHit


class DenseRetriever:
    """Encodes queries and searches a FAISS IndexFlatIP for nearest documents."""

    def __init__(
        self,
        index: faiss.IndexFlatIP,
        pmids: list[str],
        tokenizer: Any,
        model: Any,
        encode_fn: Callable[[str, Any, Any], np.ndarray],
    ) -> None:
        """
        Args:
            index: A loaded, populated FAISS ``IndexFlatIP`` (M3.1.3 output).
            pmids: PMIDs in the exact same order as the index's internal
                vector positions — i.e. ``pmids[i]`` must be the PMID of
                the document embedded at position ``i`` when the index
                was built. See module-level OPEN ARCHITECTURAL ISSUE:
                correctly sourcing this list is a wiring-layer
                responsibility this class does not resolve.
            tokenizer: Loaded MedCPT query tokenizer.
            model: Loaded MedCPT query encoder model
                (``ncbi/MedCPT-Query-Encoder``, per M3.2 frozen
                architecture).
            encode_fn: Callable ``(text, tokenizer, model) -> np.ndarray``
                returning a single embedding vector, injected so this
                class does not hardcode a specific encoding routine.
        """
        self._index = index
        self._pmids = pmids
        self._tokenizer = tokenizer
        self._model = model
        self._encode_fn = encode_fn

    @classmethod
    def from_disk(
        cls,
        index_path: Path,
        metadata_path: Path,
        pmids_path: Path,
        tokenizer: Any,
        model: Any,
        encode_fn: Callable[[str, Any, Any], np.ndarray],
    ) -> DenseRetriever:
        """Load a DenseRetriever from the M3.1.3 FAISS artifacts.

        Per ACR-001, the pmids ordering is read from ``faiss_pmids.json``
        rather than derived from the BM25 artifact or from re-reading
        ``corpus.jsonl`` — both of those approaches were explicitly
        rejected. ``document_count`` from ``embedding_metadata.json`` is
        cross-checked against the loaded pmids list and the index's own
        vector count, so a corrupted or mismatched artifact set fails
        loudly at load time rather than producing silently wrong
        PMID-to-score mappings at query time.

        Args:
            index_path: Path to ``faiss_index.bin`` (M3.1.3 output).
            metadata_path: Path to ``embedding_metadata.json`` (M3.1.3
                output).
            pmids_path: Path to ``faiss_pmids.json`` (M3.1.3 output, added
                per ACR-001).
            tokenizer: Loaded MedCPT query tokenizer.
            model: Loaded MedCPT query encoder model
                (``ncbi/MedCPT-Query-Encoder``, per M3.2 frozen
                architecture). Loading this model/tokenizer is a wiring
                concern outside ACR-001's scope and is not performed here.
            encode_fn: Callable ``(text, tokenizer, model) -> np.ndarray``
                returning a single embedding vector.

        Returns:
            A populated ``DenseRetriever``.

        Raises:
            FileNotFoundError: if any of the three artifact files is
                missing.
            ValueError: if the pmids count doesn't match either the
                index's vector count or ``embedding_metadata.json``'s
                ``document_count`` — indicates a corrupted or
                out-of-sync artifact set.
        """
        for path in (index_path, metadata_path, pmids_path):
            if not path.exists():
                raise FileNotFoundError(f"FAISS artifact not found: {path}")

        index = faiss.read_index(str(index_path))
        metadata = EmbeddingMetadata.model_validate_json(metadata_path.read_text(encoding="utf-8"))
        pmids: list[str] = json.loads(pmids_path.read_text(encoding="utf-8"))

        if len(pmids) != index.ntotal:
            raise ValueError(
                f"faiss_pmids.json has {len(pmids)} entries but the FAISS "
                f"index has {index.ntotal} vectors — artifact set is out "
                "of sync."
            )
        if len(pmids) != metadata.document_count:
            raise ValueError(
                f"faiss_pmids.json has {len(pmids)} entries but "
                f"embedding_metadata.json reports document_count="
                f"{metadata.document_count} — artifact set is out of sync."
            )

        return cls(
            index=index,
            pmids=pmids,
            tokenizer=tokenizer,
            model=model,
            encode_fn=encode_fn,
        )

    def search(self, query: str, top_k: int) -> list[RetrievalHit]:
        """Return the top-k dense hits for a query, deterministically ranked.

        Args:
            query: Raw query text.
            top_k: Maximum number of hits to return.

        Returns:
            Hits sorted by score descending, PMID ascending on ties, with
            ``rank`` assigned 1-indexed after that sort.
        """
        vector = self._encode_fn(query, self._tokenizer, self._model)
        vector = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(vector)

        # IndexFlatIP is exact search, so there's no approximate-search
        # tie ambiguity from quantization — but we still sort explicitly
        # rather than trusting FAISS's returned order, per the M3.2
        # determinism contract (score desc, PMID asc).
        scores, indices = self._index.search(vector, top_k)

        scored_pmids = [
            (self._pmids[idx], float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx != -1
        ]
        scored_pmids.sort(key=lambda pair: (-pair[1], pair[0]))

        return [
            RetrievalHit(pmid=pmid, score=score, rank=i + 1)
            for i, (pmid, score) in enumerate(scored_pmids)
        ]

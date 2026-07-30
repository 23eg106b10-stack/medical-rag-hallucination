"""Tests for DenseRetriever, including the ACR-001 from_disk loader.

New coverage — no existing tests/retrieval/test_dense.py was available to
extend, so this is a standalone module.
"""

from __future__ import annotations

import json

import faiss
import numpy as np
import pytest

from retrieval.dense import DenseRetriever
from schemas.embedding import EmbeddingMetadata


def _write_toy_artifacts(tmp_path):
    vectors = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]],
        dtype=np.float32,
    )
    index = faiss.IndexFlatIP(2)
    index.add(vectors)

    index_path = tmp_path / "faiss_index.bin"
    metadata_path = tmp_path / "embedding_metadata.json"
    pmids_path = tmp_path / "faiss_pmids.json"

    faiss.write_index(index, str(index_path))
    metadata = EmbeddingMetadata(
        embedding_version="1.0",
        corpus_version="1.0",
        corpus_checksum="abc123",
        model_name="ncbi/MedCPT-Article-Encoder",
        embedding_dimension=2,
        embedding_dtype="float32",
        faiss_index_type="IndexFlatIP",
        normalize_embeddings=True,
        document_count=3,
        created_at="2026-01-01T00:00:00+00:00",
    )
    metadata_path.write_text(metadata.model_dump_json(), encoding="utf-8")
    pmids_path.write_text(json.dumps(["111", "222", "333"]), encoding="utf-8")

    return index_path, metadata_path, pmids_path


def _fake_encode_fn(query, tokenizer, model):
    # Deterministic stand-in for the real MedCPT query encoder.
    return np.array([1.0, 0.0], dtype=np.float32)


def test_from_disk_loads_matching_artifacts(tmp_path):
    index_path, metadata_path, pmids_path = _write_toy_artifacts(tmp_path)

    retriever = DenseRetriever.from_disk(
        index_path=index_path,
        metadata_path=metadata_path,
        pmids_path=pmids_path,
        tokenizer=None,
        model=None,
        encode_fn=_fake_encode_fn,
    )

    hits = retriever.search("any query", top_k=3)
    assert [hit.pmid for hit in hits] == ["111", "333", "222"]
    assert hits[0].rank == 1


def test_from_disk_rejects_pmids_index_mismatch(tmp_path):
    index_path, metadata_path, pmids_path = _write_toy_artifacts(tmp_path)
    pmids_path.write_text(json.dumps(["111", "222"]), encoding="utf-8")  # short by one

    with pytest.raises(ValueError, match="out of sync"):
        DenseRetriever.from_disk(
            index_path=index_path,
            metadata_path=metadata_path,
            pmids_path=pmids_path,
            tokenizer=None,
            model=None,
            encode_fn=_fake_encode_fn,
        )


def test_from_disk_rejects_pmids_metadata_mismatch(tmp_path):
    index_path, metadata_path, pmids_path = _write_toy_artifacts(tmp_path)
    metadata = EmbeddingMetadata.model_validate_json(metadata_path.read_text(encoding="utf-8"))
    tampered = metadata.model_copy(update={"document_count": 99})
    metadata_path.write_text(tampered.model_dump_json(), encoding="utf-8")

    with pytest.raises(ValueError, match="out of sync"):
        DenseRetriever.from_disk(
            index_path=index_path,
            metadata_path=metadata_path,
            pmids_path=pmids_path,
            tokenizer=None,
            model=None,
            encode_fn=_fake_encode_fn,
        )


def test_from_disk_raises_on_missing_artifact(tmp_path):
    index_path, metadata_path, pmids_path = _write_toy_artifacts(tmp_path)
    pmids_path.unlink()

    with pytest.raises(FileNotFoundError):
        DenseRetriever.from_disk(
            index_path=index_path,
            metadata_path=metadata_path,
            pmids_path=pmids_path,
            tokenizer=None,
            model=None,
            encode_fn=_fake_encode_fn,
        )


def test_search_is_deterministic_under_score_ties():
    # Two PMIDs with identical dot-product score against a query vector
    # of [1, 0]; tie must break PMID ascending regardless of insertion order.
    index = faiss.IndexFlatIP(2)
    vectors = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    index.add(vectors)
    retriever = DenseRetriever(
        index=index,
        pmids=["200", "100"],
        tokenizer=None,
        model=None,
        encode_fn=_fake_encode_fn,
    )

    hits = retriever.search("query", top_k=2)
    assert [hit.pmid for hit in hits] == ["100", "200"]

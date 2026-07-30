"""Tests for ACR-001: faiss_pmids.json emitted alongside the FAISS index.

New coverage — the existing M3.1.3 test suite was not available to extend
directly, so this is a standalone module targeting only the ACR-001
change (write_index_and_metadata's new pmids/pmids_path behavior).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from index.faiss_pipeline import build_index, write_index_and_metadata


def _toy_index_and_pmids() -> tuple[object, list[str]]:
    vectors = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]],
        dtype=np.float32,
    )
    index = build_index(vectors)
    pmids = ["111", "222", "333"]
    return index, pmids


def test_write_index_and_metadata_emits_faiss_pmids_json(tmp_path):
    index, pmids = _toy_index_and_pmids()
    index_path = tmp_path / "faiss_index.bin"
    metadata_path = tmp_path / "embedding_metadata.json"
    pmids_path = tmp_path / "faiss_pmids.json"

    write_index_and_metadata(
        index=index,
        pmids=pmids,
        document_count=3,
        embedding_dimension=2,
        corpus_version="1.0",
        corpus_checksum="abc123",
        model_name="ncbi/MedCPT-Article-Encoder",
        index_path=index_path,
        metadata_path=metadata_path,
        pmids_path=pmids_path,
    )

    assert pmids_path.exists()
    loaded = json.loads(pmids_path.read_text(encoding="utf-8"))
    assert loaded == pmids


def test_write_index_and_metadata_rejects_mismatched_pmids_length(tmp_path):
    index, pmids = _toy_index_and_pmids()

    with pytest.raises(ValueError, match="does not match document_count"):
        write_index_and_metadata(
            index=index,
            pmids=pmids[:-1],  # deliberately short
            document_count=3,
            embedding_dimension=2,
            corpus_version="1.0",
            corpus_checksum="abc123",
            model_name="ncbi/MedCPT-Article-Encoder",
            index_path=tmp_path / "faiss_index.bin",
            metadata_path=tmp_path / "embedding_metadata.json",
            pmids_path=tmp_path / "faiss_pmids.json",
        )


def test_write_index_and_metadata_leaves_no_partial_output_on_failure(tmp_path, monkeypatch):
    index, pmids = _toy_index_and_pmids()
    index_path = tmp_path / "faiss_index.bin"
    metadata_path = tmp_path / "embedding_metadata.json"
    pmids_path = tmp_path / "faiss_pmids.json"

    import index.faiss_pipeline as faiss_pipeline_module

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(faiss_pipeline_module.faiss, "write_index", _boom)

    with pytest.raises(RuntimeError):
        write_index_and_metadata(
            index=index,
            pmids=pmids,
            document_count=3,
            embedding_dimension=2,
            corpus_version="1.0",
            corpus_checksum="abc123",
            model_name="ncbi/MedCPT-Article-Encoder",
            index_path=index_path,
            metadata_path=metadata_path,
            pmids_path=pmids_path,
        )

    assert not index_path.exists()
    assert not metadata_path.exists()
    assert not pmids_path.exists()
    assert not any(tmp_path.glob("*.tmp"))

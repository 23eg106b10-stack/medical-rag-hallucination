"""Integration tests for index.faiss_pipeline (M3.1.3)."""

from __future__ import annotations

import json
from unittest.mock import patch

import faiss
import numpy as np
import pytest
import torch

from index.faiss_pipeline import (
    build_index,
    generate_embeddings,
    load_corpus_metadata,
    normalize_embeddings,
    write_index_and_metadata,
)
from schemas.corpus import CorpusDocument
from schemas.embedding import EmbeddingMetadata


def _fake_tokenizer(texts, truncation=True, padding=True, max_length=None, return_tensors="pt"):
    limit = max_length or 512
    all_ids = []
    for text in texts:
        tokens = text.split()[:limit]
        ids = [hash(tok) % 1000 for tok in tokens] or [0]
        all_ids.append(ids)

    pad_len = max(len(ids) for ids in all_ids)
    padded = [ids + [0] * (pad_len - len(ids)) for ids in all_ids]
    return {"input_ids": torch.tensor(padded)}


def _fake_model(dim: int = 4):
    def model(**inputs):
        ids = inputs["input_ids"].float()
        batch_size, seq_len = ids.shape
        row_values = ids[:, :1]
        hidden = row_values.unsqueeze(1).expand(batch_size, seq_len, dim).clone()
        return type("Output", (), {"last_hidden_state": hidden})()

    return model


@pytest.fixture
def sample_documents() -> list[CorpusDocument]:
    return [
        CorpusDocument(
            pmid="1",
            title="Metformin",
            abstract="Effect of metformin on type 2 diabetes.",
        ),
        CorpusDocument(
            pmid="2",
            title="Aspirin",
            abstract="Aspirin reduces inflammation in patients.",
        ),
        CorpusDocument(
            pmid="3",
            title="Statins",
            abstract="Statins lower cholesterol levels effectively.",
        ),
        CorpusDocument(
            pmid="4",
            title="Insulin",
            abstract="Insulin therapy for type 1 diabetes patients.",
        ),
        CorpusDocument(
            pmid="5",
            title="Warfarin",
            abstract="Warfarin as an anticoagulant treatment option.",
        ),
    ]


@pytest.fixture
def corpus_meta_dir(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    meta = {
        "corpus_version": "v1",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "document_count": 5,
        "checksum_sha256": "abc123checksum",
    }
    (corpus_dir / "corpus.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return corpus_dir


class TestLoadCorpusMetadata:
    def test_reads_checksum_and_version(self, corpus_meta_dir):
        metadata = load_corpus_metadata(corpus_meta_dir)

        assert metadata.corpus_version == "v1"
        assert metadata.checksum_sha256 == "abc123checksum"

    def test_raises_when_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_corpus_metadata(tmp_path / "does_not_exist")


class TestGenerateEmbeddings:
    def test_returns_one_vector_per_document_in_order(self, sample_documents):
        vectors = generate_embeddings(
            sample_documents,
            _fake_tokenizer,
            _fake_model(dim=4),
            batch_size=2,
        )

        assert vectors.shape == (5, 4)
        assert vectors.dtype == np.float32

    def test_batching_produces_same_result_as_single_batch(self, sample_documents):
        model = _fake_model(dim=4)

        batched = generate_embeddings(sample_documents, _fake_tokenizer, model, batch_size=2)
        single_batch = generate_embeddings(
            sample_documents,
            _fake_tokenizer,
            model,
            batch_size=len(sample_documents),
        )

        np.testing.assert_allclose(batched, single_batch, rtol=1e-5)

    def test_uneven_final_batch_is_handled(self, sample_documents):
        vectors = generate_embeddings(
            sample_documents,
            _fake_tokenizer,
            _fake_model(dim=4),
            batch_size=3,
        )

        assert vectors.shape == (5, 4)

    def test_aborts_on_first_batch_failure_no_partial_result(self, sample_documents):
        def broken_model(**inputs):
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            generate_embeddings(sample_documents, _fake_tokenizer, broken_model, batch_size=2)


class TestNormalizeEmbeddings:
    def test_rows_become_unit_length(self, sample_documents):
        vectors = generate_embeddings(
            sample_documents,
            _fake_tokenizer,
            _fake_model(dim=4),
            batch_size=2,
        )
        normalized = normalize_embeddings(vectors)

        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_allclose(norms, np.ones(5), rtol=1e-5)


class TestBuildIndex:
    def test_index_type_and_count(self, sample_documents):
        vectors = normalize_embeddings(
            generate_embeddings(sample_documents, _fake_tokenizer, _fake_model(dim=4), batch_size=2)
        )

        index = build_index(vectors)

        assert isinstance(index, faiss.IndexFlatIP)
        assert index.ntotal == 5


class TestWriteIndexAndMetadata:
    def _build_sample_index(self, sample_documents, dim=4):
        vectors = normalize_embeddings(
            generate_embeddings(
                sample_documents,
                _fake_tokenizer,
                _fake_model(dim=dim),
                batch_size=2,
            )
        )
        return build_index(vectors), vectors.shape[1]

    def test_writes_readable_index_and_metadata(self, tmp_path, sample_documents):
        index, dim = self._build_sample_index(sample_documents)
        index_path = tmp_path / "indexes" / "faiss_index.bin"
        metadata_path = tmp_path / "indexes" / "embedding_metadata.json"
        pmids_path = tmp_path / "indexes" / "faiss_pmids.json"

        pmids = [doc.pmid for doc in sample_documents]
        metadata = write_index_and_metadata(
            index=index,
            pmids=pmids,
            document_count=5,
            embedding_dimension=dim,
            corpus_version="v1",
            corpus_checksum="abc123checksum",
            model_name="ncbi/MedCPT-Article-Encoder",
            index_path=index_path,
            metadata_path=metadata_path,
            pmids_path=pmids_path,
        )

        assert index_path.exists()
        assert metadata_path.exists()
        assert not index_path.with_suffix(index_path.suffix + ".tmp").exists()
        assert not metadata_path.with_suffix(metadata_path.suffix + ".tmp").exists()

        reloaded_index = faiss.read_index(str(index_path))
        assert reloaded_index.ntotal == 5

        assert metadata.corpus_checksum == "abc123checksum"
        assert metadata.normalize_embeddings is True
        assert metadata.embedding_dtype == "float32"
        assert metadata.faiss_index_type == "IndexFlatIP"

        on_disk = EmbeddingMetadata.model_validate_json(metadata_path.read_text(encoding="utf-8"))
        assert on_disk.corpus_checksum == "abc123checksum"
        assert on_disk.document_count == 5

    def test_checksum_is_copied_verbatim_not_recomputed(self, tmp_path, sample_documents):
        index, dim = self._build_sample_index(sample_documents)
        metadata_path = tmp_path / "indexes" / "embedding_metadata.json"
        pmids_path = tmp_path / "indexes" / "faiss_pmids.json"

        pmids = [doc.pmid for doc in sample_documents]
        metadata = write_index_and_metadata(
            index=index,
            pmids=pmids,
            document_count=5,
            embedding_dimension=dim,
            corpus_version="v2",
            corpus_checksum="a-checksum-that-is-not-recomputed",
            model_name="ncbi/MedCPT-Article-Encoder",
            index_path=tmp_path / "indexes" / "faiss_index.bin",
            metadata_path=metadata_path,
            pmids_path=pmids_path,
        )

        assert metadata.corpus_checksum == "a-checksum-that-is-not-recomputed"

    def test_metadata_write_failure_leaves_no_orphaned_index_artifact(
        self,
        tmp_path,
        sample_documents,
    ):
        """Atomicity contract: if metadata serialization fails, neither the
        final index file nor any temp index file should remain on disk -
        an index with no matching metadata is the exact partial-output
        state the M3.1.3 error boundary forbids.
        """
        index, dim = self._build_sample_index(sample_documents)
        index_path = tmp_path / "indexes" / "faiss_index.bin"
        metadata_path = tmp_path / "indexes" / "embedding_metadata.json"
        pmids_path = tmp_path / "indexes" / "faiss_pmids.json"

        pmids = [doc.pmid for doc in sample_documents]
        original_write_text = type(metadata_path).write_text

        def failing_write_text(self, *args, **kwargs):
            if self.name.startswith("embedding_metadata"):
                raise OSError("simulated disk failure")
            return original_write_text(self, *args, **kwargs)

        with patch("pathlib.Path.write_text", new=failing_write_text):
            with pytest.raises(OSError):
                write_index_and_metadata(
                    index=index,
                    pmids=pmids,
                    document_count=5,
                    embedding_dimension=dim,
                    corpus_version="v1",
                    corpus_checksum="abc123checksum",
                    model_name="ncbi/MedCPT-Article-Encoder",
                    index_path=index_path,
                    metadata_path=metadata_path,
                    pmids_path=pmids_path,
                )

        assert not index_path.exists()
        assert not metadata_path.exists()
        assert not pmids_path.exists()
        assert not index_path.with_suffix(index_path.suffix + ".tmp").exists()
        assert not metadata_path.with_suffix(metadata_path.suffix + ".tmp").exists()

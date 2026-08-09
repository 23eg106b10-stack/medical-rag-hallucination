"""Tests for the BM25 index construction pipeline (M3.1.2).

Covers all stages: load_corpus, load_corpus_version, validate_corpus,
extract_document_text, build_index, serialize_index, and write_metadata.
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import pytest

from config.settings import Settings, get_settings
from index.bm25_pipeline import (
    build_index,
    extract_document_text,
    load_corpus,
    load_corpus_version,
    serialize_index,
    validate_corpus,
    write_metadata,
)
from index.build_bm25 import build_bm25_index
from schemas.bm25 import BM25Metadata
from schemas.corpus import CorpusDocument, CorpusMetadata

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def corpus_dir(tmp_path: Path) -> Path:
    """Create a temporary corpus directory with corpus.jsonl and metadata."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    docs = [
        CorpusDocument(pmid="1", title="First title", abstract="First abstract."),
        CorpusDocument(pmid="2", title="Second title", abstract="Second abstract."),
        CorpusDocument(pmid="3", title="Third title", abstract="Third abstract."),
    ]

    with (corpus_dir / "corpus.jsonl").open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(doc.model_dump_json() + "\n")

    meta = CorpusMetadata(
        corpus_version="test-v1",
        generated_at=datetime.now(timezone.utc).isoformat(),
        document_count=len(docs),
        checksum_sha256="abc123",
    )
    (corpus_dir / "corpus.meta.json").write_text(meta.model_dump_json(indent=2), encoding="utf-8")
    return corpus_dir


@pytest.fixture()
def corpus_documents() -> list[CorpusDocument]:
    """Return three corpus documents."""
    return [
        CorpusDocument(pmid="1", title="Title one", abstract="Abstract one."),
        CorpusDocument(pmid="2", title="Title two", abstract="Abstract two."),
        CorpusDocument(pmid="3", title="Title three", abstract="Abstract three."),
    ]


# ── load_corpus ───────────────────────────────────────────────────────────────


class TestLoadCorpus:
    """Tests for the load_corpus pipeline stage."""

    def test_loads_all_documents(self, corpus_dir: Path) -> None:
        docs = load_corpus(corpus_dir / "corpus.jsonl")
        assert len(docs) == 3

    def test_returns_list_of_corpus_documents(self, corpus_dir: Path) -> None:
        docs = load_corpus(corpus_dir / "corpus.jsonl")
        assert all(isinstance(d, CorpusDocument) for d in docs)

    def test_preserves_order(self, corpus_dir: Path) -> None:
        docs = load_corpus(corpus_dir / "corpus.jsonl")
        assert [d.pmid for d in docs] == ["1", "2", "3"]

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_corpus(tmp_path / "does_not_exist.jsonl")

    def test_empty_lines_skipped(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "corpus.jsonl"
        jsonl.write_text("\n\n", encoding="utf-8")
        docs = load_corpus(jsonl)
        assert docs == []


# ── load_corpus_version ───────────────────────────────────────────────────────


class TestLoadCorpusVersion:
    """Tests for the load_corpus_version pipeline stage."""

    def test_returns_corpus_version(self, corpus_dir: Path) -> None:
        version = load_corpus_version(corpus_dir)
        assert version == "test-v1"

    def test_missing_metadata_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_corpus_version(tmp_path)


# ── validate_corpus ───────────────────────────────────────────────────────────


class TestValidateCorpus:
    """Tests for the validate_corpus pipeline stage."""

    def test_valid_corpus_passes(self, corpus_documents: list[CorpusDocument]) -> None:
        validate_corpus(corpus_documents)  # should not raise

    def test_empty_corpus_raises(self) -> None:
        with pytest.raises(ValueError, match="Corpus is empty"):
            validate_corpus([])

    def test_missing_pmid_raises(self) -> None:
        docs = [CorpusDocument(pmid="", title="t", abstract="a")]
        with pytest.raises(ValueError, match="missing a required field"):
            validate_corpus(docs)

    def test_missing_title_raises(self) -> None:
        docs = [CorpusDocument(pmid="1", title="", abstract="a")]
        with pytest.raises(ValueError, match="missing a required field"):
            validate_corpus(docs)

    def test_missing_abstract_raises(self) -> None:
        docs = [CorpusDocument(pmid="1", title="t", abstract="")]
        with pytest.raises(ValueError, match="missing a required field"):
            validate_corpus(docs)


# ── extract_document_text ─────────────────────────────────────────────────────


class TestExtractDocumentText:
    """Tests for the extract_document_text pipeline stage."""

    def test_concatenates_title_and_abstract(self) -> None:
        doc = CorpusDocument(pmid="1", title="My Title", abstract="My abstract.")
        text = extract_document_text(doc)
        assert text == "My Title My abstract."

    def test_no_field_weighting(self) -> None:
        doc = CorpusDocument(pmid="1", title="T", abstract="A")
        text = extract_document_text(doc)
        assert text == "T A"


# ── build_index ───────────────────────────────────────────────────────────────


class TestBuildIndex:
    """Tests for the build_index pipeline stage."""

    def test_returns_model_and_pmids(self, corpus_documents: list[CorpusDocument]) -> None:
        model, pmids = build_index(corpus_documents, tokenizer=str.split)
        assert pmids == ["1", "2", "3"]

    def test_empty_token_list_raises(self, corpus_documents: list[CorpusDocument]) -> None:
        def empty_tokenizer(text: str) -> list[str]:
            return []

        with pytest.raises(ValueError, match="empty token list"):
            build_index(corpus_documents, tokenizer=empty_tokenizer)


# ── serialize_index ───────────────────────────────────────────────────────────


class TestSerializeIndex:
    """Tests for the serialize_index pipeline stage."""

    def test_creates_index_file(self, tmp_path: Path) -> None:
        from index.bm25_pipeline import build_index
        from schemas.corpus import CorpusDocument

        docs = [CorpusDocument(pmid="1", title="T", abstract="A")]
        model, pmids = build_index(docs, tokenizer=str.split)
        index_path = tmp_path / "bm25_index.pkl"
        serialize_index(model, pmids, index_path)
        assert index_path.exists()

    def test_pickled_payload_contains_model_and_pmids(self, tmp_path: Path) -> None:
        from index.bm25_pipeline import build_index
        from schemas.corpus import CorpusDocument

        docs = [CorpusDocument(pmid="1", title="T", abstract="A")]
        model, pmids = build_index(docs, tokenizer=str.split)
        index_path = tmp_path / "bm25_index.pkl"
        serialize_index(model, pmids, index_path)

        with index_path.open("rb") as f:
            payload = pickle.load(f)

        assert "model" in payload
        assert "pmids" in payload
        assert payload["pmids"] == ["1"]


# ── write_metadata ────────────────────────────────────────────────────────────


class TestWriteMetadata:
    """Tests for the write_metadata pipeline stage."""

    def test_creates_metadata_file(self, tmp_path: Path) -> None:
        metadata_path = tmp_path / "bm25_metadata.json"
        write_metadata(10, "v1", metadata_path)
        assert metadata_path.exists()

    def test_writes_valid_json(self, tmp_path: Path) -> None:
        metadata_path = tmp_path / "bm25_metadata.json"
        write_metadata(10, "v1", metadata_path)
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert raw["bm25_version"] == "1.0"
        assert raw["corpus_version"] == "v1"
        assert raw["document_count"] == 10
        assert raw["tokenizer"] == "simple"
        assert raw["algorithm"] == "BM25Okapi"
        assert raw["library"] == "rank-bm25"

    def test_returns_bm25_metadata(self, tmp_path: Path) -> None:
        metadata = write_metadata(10, "v1", tmp_path / "m.json")
        assert isinstance(metadata, BM25Metadata)

    def test_created_at_is_utc_iso(self, tmp_path: Path) -> None:
        metadata = write_metadata(10, "v1", tmp_path / "m.json")
        created = datetime.fromisoformat(metadata.created_at)
        assert created.tzinfo is not None


# ── build_bm25_index (ACR-003 regression tests) ──────────────────────────────


class TestBuildBm25Index:
    """ACR-003 regression tests for build_bm25_index configuration handling."""

    def test_default_bm25_index_filename_is_pkl(self) -> None:
        settings = get_settings()
        assert settings.bm25_index_filename == "bm25_index.pkl"

    def test_build_bm25_index_respects_custom_filename(
        self, corpus_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        custom_filename = "custom_bm25_index.pkl"
        indexes_dir = corpus_dir.parent / "indexes"
        indexes_dir.mkdir(exist_ok=True)

        test_settings = Settings(
            corpus_dir=corpus_dir,
            indexes_dir=indexes_dir,
            bm25_index_filename=custom_filename,
        )

        monkeypatch.setattr("index.build_bm25.get_settings", lambda: test_settings)

        build_bm25_index()

        expected_index_path = indexes_dir / custom_filename
        hardcoded_index_path = indexes_dir / "bm25_index.pkl"

        assert expected_index_path.exists()
        assert not hardcoded_index_path.exists()

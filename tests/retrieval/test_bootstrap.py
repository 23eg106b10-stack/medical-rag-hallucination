"""Tests for retrieval.bootstrap.build_hybrid_retriever (M3.2 composition root).

Deliberately uses mocks/fakes only — no real corpus, no real FAISS index,
no MedCPT download. Real-artifact smoke testing is an explicitly separate
validation step, not part of this suite.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

import retrieval.bootstrap as bootstrap_module
from config.settings import Settings
from retrieval.retriever import HybridRetriever


@pytest.fixture()
def custom_settings(tmp_path: Path) -> Settings:
    """A Settings instance pointing at isolated, non-existent temp paths.

    Component construction itself is mocked in every test below, so these
    paths are never actually read from disk — only used to assert that
    build_hybrid_retriever derives paths from settings correctly.
    """
    return Settings(
        indexes_dir=tmp_path / "indexes",
        corpus_dir=tmp_path / "corpus",
        bm25_index_filename="bm25_index.pkl",
        faiss_index_filename="faiss_index.bin",
        medcpt_query_encoder_model_name="ncbi/MedCPT-Query-Encoder",
    )


@pytest.fixture()
def mocked_components(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    """Patch every component constructor build_hybrid_retriever calls.

    Patched at the point of use (``retrieval.bootstrap.X``), not at
    definition, per standard mocking practice — these are the exact names
    imported into the bootstrap module's namespace.
    """
    fake_sparse = MagicMock(name="BM25Retriever_instance")
    fake_dense = MagicMock(name="DenseRetriever_instance")
    fake_corpus_reader = MagicMock(name="InMemoryCorpusReader_instance")
    fake_tokenizer = MagicMock(name="query_tokenizer")
    fake_model = MagicMock(name="query_model")

    mock_from_pickle = MagicMock(return_value=fake_sparse)
    mock_from_disk = MagicMock(return_value=fake_dense)
    mock_from_jsonl = MagicMock(return_value=fake_corpus_reader)
    mock_load_encoder = MagicMock(return_value=(fake_tokenizer, fake_model))
    mock_tokenize = MagicMock(name="tokenize")

    monkeypatch.setattr(bootstrap_module.BM25Retriever, "from_pickle", mock_from_pickle)
    monkeypatch.setattr(bootstrap_module.DenseRetriever, "from_disk", mock_from_disk)
    monkeypatch.setattr(bootstrap_module.InMemoryCorpusReader, "from_jsonl", mock_from_jsonl)
    monkeypatch.setattr(bootstrap_module, "load_encoder", mock_load_encoder)
    monkeypatch.setattr(bootstrap_module, "tokenize", mock_tokenize)

    return {
        "from_pickle": mock_from_pickle,
        "from_disk": mock_from_disk,
        "from_jsonl": mock_from_jsonl,
        "load_encoder": mock_load_encoder,
        "tokenize": mock_tokenize,
        "sparse_instance": fake_sparse,
        "dense_instance": fake_dense,
        "corpus_reader_instance": fake_corpus_reader,
        "tokenizer_instance": fake_tokenizer,
        "model_instance": fake_model,
    }


# ── 1. Returns a HybridRetriever ─────────────────────────────────────────────


class TestBuildHybridRetrieverReturnType:
    def test_returns_hybrid_retriever_instance(
        self, custom_settings: Settings, mocked_components: dict[str, MagicMock]
    ) -> None:
        result = bootstrap_module.build_hybrid_retriever(settings=custom_settings)
        assert isinstance(result, HybridRetriever)


# ── 2. Expected dependency construction occurs ───────────────────────────────


class TestDependencyConstruction:
    def test_all_four_component_loaders_are_called_exactly_once(
        self, custom_settings: Settings, mocked_components: dict[str, MagicMock]
    ) -> None:
        bootstrap_module.build_hybrid_retriever(settings=custom_settings)

        mocked_components["from_pickle"].assert_called_once()
        mocked_components["from_disk"].assert_called_once()
        mocked_components["from_jsonl"].assert_called_once()
        mocked_components["load_encoder"].assert_called_once()

    def test_hybrid_retriever_constructed_with_the_loaded_components(
        self, custom_settings: Settings, mocked_components: dict[str, MagicMock]
    ) -> None:
        result = bootstrap_module.build_hybrid_retriever(settings=custom_settings)

        assert result._sparse_retriever is mocked_components["sparse_instance"]
        assert result._dense_retriever is mocked_components["dense_instance"]
        assert result._corpus_reader is mocked_components["corpus_reader_instance"]


# ── 3. Custom/injected Settings object is respected ─────────────────────────


class TestSettingsInjection:
    def test_bm25_path_derived_from_injected_settings(
        self, custom_settings: Settings, mocked_components: dict[str, MagicMock]
    ) -> None:
        bootstrap_module.build_hybrid_retriever(settings=custom_settings)

        _, kwargs = mocked_components["from_pickle"].call_args
        called_args = mocked_components["from_pickle"].call_args.args
        index_path = kwargs.get("index_path") or (called_args[0] if called_args else None)

        expected_path = custom_settings.indexes_dir / custom_settings.bm25_index_filename
        assert index_path == expected_path

    def test_corpus_path_derived_from_injected_settings(
        self, custom_settings: Settings, mocked_components: dict[str, MagicMock]
    ) -> None:
        bootstrap_module.build_hybrid_retriever(settings=custom_settings)

        _, kwargs = mocked_components["from_jsonl"].call_args
        called_args = mocked_components["from_jsonl"].call_args.args
        corpus_path = kwargs.get("corpus_path") or (called_args[0] if called_args else None)

        expected_path = custom_settings.corpus_dir / "corpus.jsonl"
        assert corpus_path == expected_path

    def test_omitting_settings_falls_back_to_get_settings(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mocked_components: dict[str, MagicMock],
        custom_settings: Settings,
    ) -> None:
        mock_get_settings = MagicMock(return_value=custom_settings)
        monkeypatch.setattr(bootstrap_module, "get_settings", mock_get_settings)

        bootstrap_module.build_hybrid_retriever()

        mock_get_settings.assert_called_once()


# ── 4. Query encoder model name sourced from settings ───────────────────────


class TestQueryEncoderModelName:
    def test_load_encoder_called_with_query_encoder_setting(
        self, custom_settings: Settings, mocked_components: dict[str, MagicMock]
    ) -> None:
        bootstrap_module.build_hybrid_retriever(settings=custom_settings)

        mocked_components["load_encoder"].assert_called_once_with(
            custom_settings.medcpt_query_encoder_model_name
        )

    def test_a_different_query_encoder_setting_is_honored(
        self,
        tmp_path: Path,
        mocked_components: dict[str, MagicMock],
    ) -> None:
        other_settings = Settings(
            indexes_dir=tmp_path / "indexes",
            corpus_dir=tmp_path / "corpus",
            medcpt_query_encoder_model_name="some/other-query-encoder",
        )

        bootstrap_module.build_hybrid_retriever(settings=other_settings)

        mocked_components["load_encoder"].assert_called_once_with("some/other-query-encoder")


# ── 5. encode_fn adapter shape ───────────────────────────────────────────────


def _fake_tokenizer(texts, truncation=True, padding=True, max_length=None, return_tensors="pt"):
    import torch

    ids = [[1, 2, 3] for _ in texts]
    return {"input_ids": torch.tensor(ids)}


def _fake_model(dim: int = 4):
    def model(**inputs):
        ids = inputs["input_ids"].float()
        batch_size, seq_len = ids.shape
        hidden = ids[:, :1].unsqueeze(1).expand(batch_size, seq_len, dim).clone()
        return type("Output", (), {"last_hidden_state": hidden})()

    return model


class TestEncodeAdapterShape:
    def test_encode_fn_returns_single_vector_matching_embedding_dim(self) -> None:
        encode_fn = bootstrap_module._make_query_encode_fn()

        vector = encode_fn("what causes fever?", _fake_tokenizer, _fake_model(dim=4))

        assert isinstance(vector, np.ndarray)
        assert vector.shape == (4,)

    def test_encode_fn_output_is_reshape_compatible_with_dense_retriever(self) -> None:
        """DenseRetriever.search does np.asarray(vector).reshape(1, -1) —
        confirm the adapter's output survives that unchanged.
        """
        encode_fn = bootstrap_module._make_query_encode_fn()
        vector = encode_fn("a query", _fake_tokenizer, _fake_model(dim=4))

        reshaped = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        assert reshaped.shape == (1, 4)


# ── 6. Existing constructors called with expected artifact paths ────────────


class TestArtifactPaths:
    def test_faiss_paths_derived_from_settings_and_conventions(
        self, custom_settings: Settings, mocked_components: dict[str, MagicMock]
    ) -> None:
        bootstrap_module.build_hybrid_retriever(settings=custom_settings)

        _, kwargs = mocked_components["from_disk"].call_args

        assert kwargs["index_path"] == (
            custom_settings.indexes_dir / custom_settings.faiss_index_filename
        )
        assert kwargs["metadata_path"] == (custom_settings.indexes_dir / "embedding_metadata.json")
        assert kwargs["pmids_path"] == (custom_settings.indexes_dir / "faiss_pmids.json")

    def test_dense_retriever_receives_the_loaded_query_encoder(
        self, custom_settings: Settings, mocked_components: dict[str, MagicMock]
    ) -> None:
        bootstrap_module.build_hybrid_retriever(settings=custom_settings)

        _, kwargs = mocked_components["from_disk"].call_args

        assert kwargs["tokenizer"] is mocked_components["tokenizer_instance"]
        assert kwargs["model"] is mocked_components["model_instance"]
        assert callable(kwargs["encode_fn"])

    def test_bm25_retriever_receives_the_bm25_tokenizer(
        self, custom_settings: Settings, mocked_components: dict[str, MagicMock]
    ) -> None:
        bootstrap_module.build_hybrid_retriever(settings=custom_settings)

        _, kwargs = mocked_components["from_pickle"].call_args
        assert kwargs["tokenizer"] is mocked_components["tokenize"]

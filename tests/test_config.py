"""Sanity test: settings load correctly from environment variables."""

from __future__ import annotations

from pathlib import Path

from config.settings import Settings, get_settings


def test_settings_defaults(settings_env: None) -> None:
    """Verify Settings picks up the monkeypatched env vars."""
    settings = get_settings()
    assert settings.environment == "test"
    assert settings.log_level == "DEBUG"
    assert settings.hf_token == "hf_test_fake_token"
    assert settings.medcpt_query_encoder_name == "ncbi/MedCPT-Query-Encoder"
    assert settings.medcpt_article_encoder_name == "ncbi/MedCPT-Article-Encoder"
    assert settings.medcpt_embedding_dimension == 768
    assert settings.generation_model_name == "meta-llama/Llama-3.1-8B-Instruct"
    assert settings.generation_quantization_bits == 4
    assert settings.verifier_model_name == "pritamdeka/PubMedBERT-MNLI-MedNLI"
    assert settings.retrieval_bm25_top_k == 20
    assert settings.retrieval_dense_top_k == 20
    assert settings.retrieval_fused_top_k == 5
    assert settings.calibration_split_size == 20
    assert settings.holdout_split_size == 80


def test_settings_singleton(settings_env: None) -> None:
    """get_settings() should return the same instance each time."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_settings_paths(settings_env: None) -> None:
    """Verify path-type fields are Path objects."""
    settings = Settings()  # type: ignore[call-arg]
    assert isinstance(settings.faiss_index_path, Path)
    assert str(settings.faiss_index_path) == "index/artifacts/faiss_flat.index"
    assert isinstance(settings.log_dir, Path)
    assert isinstance(settings.outputs_dir, Path)


def test_settings_verifier_id2label(settings_env: None) -> None:
    """Verify the verifier id2label mapping is set explicitly per §7.2."""
    settings = get_settings()
    assert settings.verifier_id2label == {
        0: "contradiction",
        1: "entailment",
        2: "neutral",
    }


def test_settings_log_level_int(settings_env: None) -> None:
    """Verify log_level_int() returns a valid logging level."""
    import logging

    settings = get_settings()
    assert settings.log_level_int() == logging.DEBUG
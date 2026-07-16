"""Shared pytest fixtures for the medical-rag-hallucination test suite."""

from __future__ import annotations

from pathlib import Path

from _pytest.monkeypatch import MonkeyPatch

import pytest


@pytest.fixture
def settings_env(monkeypatch: MonkeyPatch) -> None:
    """Fixture that sets minimal env vars so Settings loads without a real .env."""
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("HF_TOKEN", "hf_test_fake_token")
    monkeypatch.setenv("MEDCPT_QUERY_ENCODER_NAME", "ncbi/MedCPT-Query-Encoder")
    monkeypatch.setenv("MEDCPT_ARTICLE_ENCODER_NAME", "ncbi/MedCPT-Article-Encoder")
    monkeypatch.setenv("MEDCPT_EMBEDDING_DIMENSION", "768")
    monkeypatch.setenv("GENERATION_MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
    monkeypatch.setenv("GENERATION_QUANTIZATION_BITS", "4")
    monkeypatch.setenv(
        "VERIFIER_MODEL_NAME", "pritamdeka/PubMedBERT-MNLI-MedNLI"
    )
    monkeypatch.setenv("RETRIEVAL_BM25_TOP_K", "20")
    monkeypatch.setenv("RETRIEVAL_DENSE_TOP_K", "20")
    monkeypatch.setenv("RETRIEVAL_FUSED_TOP_K", "5")
    monkeypatch.setenv("CALIBRATION_SPLIT_SIZE", "20")
    monkeypatch.setenv("HOLDOUT_SPLIT_SIZE", "80")


@pytest.fixture
def test_data_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for test data."""
    return tmp_path
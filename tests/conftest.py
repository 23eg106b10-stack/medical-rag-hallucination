"""Shared pytest fixtures for the medical-rag-hallucination test suite."""

from __future__ import annotations

from pathlib import Path

from _pytest.monkeypatch import MonkeyPatch

import pytest


@pytest.fixture
def settings_env(monkeypatch: MonkeyPatch) -> None:
    """Fixture that sets minimal env vars so Settings loads without a real .env."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key")
    monkeypatch.setenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
    monkeypatch.setenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "384")


@pytest.fixture
def test_data_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for test data."""
    return tmp_path
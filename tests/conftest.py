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


@pytest.fixture
def test_data_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for test data."""
    return tmp_path
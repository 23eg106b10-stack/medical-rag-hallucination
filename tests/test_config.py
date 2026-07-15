"""Sanity test: settings load correctly from environment variables."""

from __future__ import annotations

from pathlib import Path

from config.settings import LLMProvider, Settings, get_settings


def test_settings_defaults(settings_env: None) -> None:
    """Verify Settings picks up the monkeypatched env vars."""
    settings = get_settings()
    assert settings.llm_provider == LLMProvider.OPENAI
    assert settings.openai_api_key == "sk-test-fake-key"
    assert settings.openai_model_name == "gpt-4o-mini"
    assert settings.embedding_model_name == "all-MiniLM-L6-v2"
    assert settings.embedding_dimension == 384


def test_settings_singleton(settings_env: None) -> None:
    """get_settings() should return the same instance each time."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_settings_paths(settings_env: None) -> None:
    """Verify path-type fields are Path objects."""
    settings = Settings()  # type: ignore[call-arg]
    assert isinstance(settings.faiss_index_path, Path)
    assert str(settings.faiss_index_path) == "data/faiss_index.bin"

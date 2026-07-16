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
    assert settings.corpus_dir == Path("data/corpus")
    assert settings.outputs_dir == Path("outputs")


def test_settings_singleton(settings_env: None) -> None:
    """get_settings() should return the same instance each time."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_settings_paths(settings_env: None) -> None:
    """Verify path-type fields are Path objects."""
    settings = Settings()  # type: ignore[call-arg]
    assert isinstance(settings.log_dir, Path)
    assert isinstance(settings.corpus_dir, Path)
    assert isinstance(settings.outputs_dir, Path)


def test_settings_log_level_int(settings_env: None) -> None:
    """Verify log_level_int() returns a valid logging level."""
    import logging

    settings = get_settings()
    assert settings.log_level_int() == logging.DEBUG


def test_settings_invalid_log_level_raises() -> None:
    """Verify an invalid log level raises ValueError."""
    settings = Settings(log_level="NONSENSE")  # type: ignore[call-arg]
    try:
        settings.log_level_int()
        assert False, "Expected ValueError"
    except ValueError:
        pass

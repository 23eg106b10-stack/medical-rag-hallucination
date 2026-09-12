"""Tests for config.settings."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

import config.settings as settings_module
from config.settings import Settings, get_settings


@pytest.fixture(autouse=True)
def _reset_settings_singleton() -> None:
    """Force get_settings() to rebuild from env on every test."""
    settings_module._settings = None
    yield
    settings_module._settings = None


def test_get_settings_returns_same_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_settings() caches a singleton across repeated calls."""
    first = get_settings()
    second = get_settings()
    assert first is second


def test_default_environment_and_log_level(monkeypatch: pytest.MonkeyPatch) -> None:
    """Defaults apply when no relevant env vars are set."""
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    settings = get_settings()
    assert settings.environment == "development"
    assert settings.log_level == "INFO"


def test_log_dir_defaults_to_relative_logs_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """log_dir defaults to Path('logs') when LOG_DIR is unset."""
    monkeypatch.delenv("LOG_DIR", raising=False)
    settings = get_settings()
    assert settings.log_dir == Path("logs")


def test_env_var_overrides_log_level(monkeypatch: pytest.MonkeyPatch) -> None:
    """LOG_LEVEL env var overrides the default."""
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    settings = get_settings()
    assert settings.log_level == "DEBUG"


def test_env_var_overrides_log_dir_as_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """LOG_DIR env var overrides the default and is coerced to a Path."""
    custom_dir = tmp_path / "custom_logs"
    monkeypatch.setenv("LOG_DIR", str(custom_dir))
    settings = get_settings()
    assert settings.log_dir == custom_dir
    assert isinstance(settings.log_dir, Path)


def test_log_level_int_resolves_known_level(monkeypatch: pytest.MonkeyPatch) -> None:
    """log_level_int() maps a valid level name to the logging module constant."""
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    settings = get_settings()
    assert settings.log_level_int() == logging.WARNING


def test_log_level_int_is_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    """log_level_int() uppercases the configured level before resolving it."""
    monkeypatch.setenv("LOG_LEVEL", "debug")
    settings = get_settings()
    assert settings.log_level_int() == logging.DEBUG


def test_log_level_int_invalid_level_raises_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unrecognized log_level raises ValueError, per the frozen contract."""
    monkeypatch.setenv("LOG_LEVEL", "NOT_A_REAL_LEVEL")
    settings = get_settings()
    with pytest.raises(ValueError):
        settings.log_level_int()


def test_unset_data_path_fields_have_expected_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Data directory fields default to their documented relative paths."""
    for var in (
        "CORPUS_DIR",
        "PUBMEDQA_DIR",
        "MEDMCQA_DIR",
        "ANNOTATIONS_DIR",
        "OUTPUTS_DIR",
    ):
        monkeypatch.delenv(var, raising=False)
    settings = get_settings()
    assert settings.corpus_dir == Path("data/corpus")
    assert settings.pubmedqa_dir == Path("data/pubmedqa")
    assert settings.medmcqa_dir == Path("data/medmcqa")
    assert settings.annotations_dir == Path("data/annotations")
    assert settings.outputs_dir == Path("outputs")


def test_unknown_env_vars_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings construction does not fail on extra, unrecognized env vars."""
    monkeypatch.setenv("SOME_UNRELATED_VARIABLE", "whatever")
    settings = get_settings()
    assert isinstance(settings, Settings)


def test_llm_quantized_layers_gpu_resident_defaults_to_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """llm_quantized_layers_gpu_resident defaults to False (ACR-004) —
    preserves existing device_map passthrough behavior unless explicitly
    enabled.
    """
    monkeypatch.delenv("LLM_QUANTIZED_LAYERS_GPU_RESIDENT", raising=False)
    settings = get_settings()
    assert settings.llm_quantized_layers_gpu_resident is False


def test_confidence_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Confidence scoring settings default to provisional uncalibrated constants (ADR-M6-005)."""
    monkeypatch.delenv("CONFIDENCE_CONTRADICTION_CEILING", raising=False)
    monkeypatch.delenv("CONFIDENCE_LEVEL_HIGH_THRESHOLD", raising=False)
    monkeypatch.delenv("CONFIDENCE_LEVEL_MEDIUM_THRESHOLD", raising=False)
    settings = get_settings()
    assert settings.confidence_contradiction_ceiling == 0.2
    assert settings.confidence_level_high_threshold == 0.8
    assert settings.confidence_level_medium_threshold == 0.5


def test_confidence_settings_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Confidence scoring settings can be overridden via environment variables."""
    monkeypatch.setenv("CONFIDENCE_CONTRADICTION_CEILING", "0.35")
    monkeypatch.setenv("CONFIDENCE_LEVEL_HIGH_THRESHOLD", "0.85")
    monkeypatch.setenv("CONFIDENCE_LEVEL_MEDIUM_THRESHOLD", "0.60")
    settings = get_settings()
    assert settings.confidence_contradiction_ceiling == 0.35
    assert settings.confidence_level_high_threshold == 0.85
    assert settings.confidence_level_medium_threshold == 0.60

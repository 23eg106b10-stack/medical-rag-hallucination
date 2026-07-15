"""Central Pydantic Settings object that reads from .env."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM Provider ──────────────────────────────────────────────────────
    llm_provider: LLMProvider = LLMProvider.OPENAI
    openai_api_key: str = ""
    openai_model_name: str = "gpt-4o-mini"
    anthropic_api_key: str = ""
    anthropic_model_name: str = "claude-sonnet-4-20250514"

    # ── Embedding Model ───────────────────────────────────────────────────
    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # ── Index Paths ───────────────────────────────────────────────────────
    faiss_index_path: Path = Path("data/faiss_index.bin")
    bm25_index_path: Path = Path("data/bm25_index.pkl")

    # ── Data Paths ────────────────────────────────────────────────────────
    corpus_dir: Path = Path("data/corpus")
    pubmedqa_dir: Path = Path("data/pubmedqa")
    medmcqa_dir: Path = Path("data/medmcqa")
    annotations_dir: Path = Path("data/annotations")
    log_dir: Path = Path("logs")


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()  # type: ignore[call-arg]
    return _settings
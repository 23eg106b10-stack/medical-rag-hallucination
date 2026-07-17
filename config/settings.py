"""Central Pydantic Settings object that reads from .env.

This is the ONLY module in the project permitted to read environment
variables directly. Every other module receives configuration via the
``settings`` singleton (see ``get_settings()``) — never via ``os.getenv``.

Fields are added here only in the milestone that first needs them. Do not
front-load retrieval/generation/verification config before those milestones
begin — see engineering standard #7 (dependency management).
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Environment / Logging ────────────────────────────────────────────
    environment: str = "development"
    log_level: str = "INFO"
    log_dir: Path = Path("logs")

    # ── Hugging Face Hub ──────────────────────────────────────────────────
    # Needed as soon as any milestone downloads a checkpoint (MedCPT,
    # verifier, or Llama). Present now because it's provider-agnostic
    # infrastructure, not tied to one specific model choice.
    hf_token: str = ""

    # ── Data Paths ────────────────────────────────────────────────────────
    corpus_dir: Path = Path("data/corpus")
    pubmedqa_dir: Path = Path("data/pubmedqa")
    medmcqa_dir: Path = Path("data/medmcqa")
    annotations_dir: Path = Path("data/annotations")
    indexes_dir: Path = Path("data/indexes")

    # ── Outputs ──────────────────────────────────────────────────────────
    outputs_dir: Path = Path("outputs")

    # ── Indexes (M3.1.2+) ────────────────────────────────────────────────
    bm25_index_filename: str = "bm25_index.json"

    # ── Corpus Construction (M3.1.1) ────────────────────────────────────
    # Canonical location of the frozen, version-controlled MeSH query
    # specification consumed (never written) by corpus construction. See
    # M3.1.1 "Query Specification Contract".
    queries_path: Path = Path("config/queries.yaml")

    # Filename of the PubMedQA release file (under pubmedqa_dir) that
    # provides the PMID list for Source A coverage. Configurable rather
    # than hardcoded since the exact release layout is a dataset detail,
    # not an architectural one.
    pubmedqa_filename: str = "ori_pqal.json"

    # NCBI E-utilities requires an identifying contact per their usage
    # policy; api_key is optional but raises the allowed request rate.
    ncbi_email: str = ""
    ncbi_api_key: str = ""

    def log_level_int(self) -> int:
        """Resolve the configured level name to a ``logging`` module constant.

        Raises:
            ValueError: if ``log_level`` is not a recognized level name.
        """
        level = logging.getLevelName(self.log_level.upper())
        if not isinstance(level, int):
            raise ValueError(f"Invalid log_level: {self.log_level!r}")
        return level


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return a cached singleton ``Settings`` instance.

    Returns:
        The process-wide ``Settings`` instance, constructed once from
        ``.env`` on first call and reused thereafter.
    """
    global _settings
    if _settings is None:
        _settings = Settings()  # type: ignore[call-arg]
    return _settings
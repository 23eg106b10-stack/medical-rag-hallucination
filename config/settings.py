"""Central Pydantic Settings object that reads from .env.

This is the ONLY module in the project permitted to read environment
variables directly. Every other module receives configuration via the
``settings`` singleton (see ``get_settings()``) — never via ``os.getenv``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file.

    Field values mirror the fixed choices in Architecture v1.0 of the
    Implementation Blueprint (§4 Final Technology Stack). Values here are
    defaults matching the frozen architecture, not tuning knobs — changing
    the model names/revisions requires updating the blueprint first.
    """

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
    # Required to download gated/rate-limited checkpoints (Llama-3.1-8B-
    # Instruct is a gated model on HF Hub). Verifier and MedCPT checkpoints
    # are public but a token avoids anonymous rate limits.
    hf_token: str = ""

    # ── Dense Retrieval: MedCPT (§4, §5.2) ──────────────────────────────
    # Two distinct encoders — do NOT reuse one for both sides (blueprint's
    # own documented accidental-substitution risk, §5.2).
    medcpt_query_encoder_name: str = "ncbi/MedCPT-Query-Encoder"
    medcpt_article_encoder_name: str = "ncbi/MedCPT-Article-Encoder"
    medcpt_embedding_dimension: int = 768

    # ── Generation LLM (§4, §6) ──────────────────────────────────────────
    # Pinned by revision hash once selected — recorded here AND in README
    # per blueprint §6/§11 reproducibility requirement. Left blank until
    # Milestone 8 pins an exact commit hash; must not silently default to
    # "latest" in code that resolves this field.
    generation_model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    generation_model_revision: str = ""
    generation_quantization_bits: int = 4

    # ── Verifier: fixed checkpoint (§4, §7.2) ────────────────────────────
    verifier_model_name: str = "pritamdeka/PubMedBERT-MNLI-MedNLI"
    verifier_model_revision: str = ""
    # Non-sequential id2label per blueprint §7.2 — set explicitly, never
    # inferred from the checkpoint's own config, to avoid silent mislabeling.
    verifier_id2label: dict[int, str] = {
        0: "contradiction",
        1: "entailment",
        2: "neutral",
    }

    # ── Retrieval tuning (§5.3) ───────────────────────────────────────────
    retrieval_bm25_top_k: int = 20
    retrieval_dense_top_k: int = 20
    retrieval_fused_top_k: int = 5
    retrieval_min_similarity_threshold: float = 0.0  # calibrated later

    # ── Confidence calibration (§8) ───────────────────────────────────────
    calibration_split_size: int = 20
    holdout_split_size: int = 80

    # ── Index Paths ───────────────────────────────────────────────────────
    faiss_index_path: Path = Path("index/artifacts/faiss_flat.index")
    bm25_index_path: Path = Path("index/artifacts/bm25.pkl")

    # ── Data Paths ────────────────────────────────────────────────────────
    corpus_dir: Path = Path("data/corpus")
    pubmedqa_dir: Path = Path("data/pubmedqa")
    medmcqa_dir: Path = Path("data/medmcqa")
    annotations_dir: Path = Path("data/annotations")

    # ── Outputs ──────────────────────────────────────────────────────────
    outputs_dir: Path = Path("outputs")

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
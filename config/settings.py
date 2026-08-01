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

    # ── Dense Index (M3.1.3) ─────────────────────────────────────────────
    faiss_index_filename: str = "faiss_index.bin"
    medcpt_model_name: str = "ncbi/MedCPT-Article-Encoder"
    embedding_batch_size: int = 16

    # ── Hybrid Retrieval (M3.2) ──────────────────────────────────────────
    # Query-side MedCPT encoder, distinct from medcpt_model_name above
    # (the article/document-side encoder used to build the FAISS index).
    # Frozen per the M3.2 architecture decision.
    medcpt_query_encoder_model_name: str = "ncbi/MedCPT-Query-Encoder"

    # ── LLM Generation (M3.3) ────────────────────────────────────────────
    # Per ADR-M3.3-001: Transformers + bitsandbytes is the mandatory
    # inference backend. Only fields actually consumed by
    # generation.llm_loader are added here — deliberately not adding
    # decoding-parameter fields (max_new_tokens, temperature, etc.) yet,
    # since no module in the frozen M3.3 scope owns the concrete
    # generate_fn that would consume them. See implementation report.
    llm_model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    llm_load_in_4bit: bool = True
    llm_bnb_4bit_quant_type: str = "nf4"
    llm_bnb_4bit_compute_dtype: str = "float16"
    llm_device_map: str = "auto"

    # Decoding parameters, consumed by generation.generator.make_transformers_generate_fn.
    # do_sample=False (greedy) is a provisional default pending a formal
    # decoding-determinism ADR — see the M3.3 implementation report.
    llm_max_new_tokens: int = 512
    llm_do_sample: bool = False

    # Prompt token budget, consumed by generation.context_builder.PromptBuilder.
    # Budgets the prompt only; does not reserve headroom for
    # llm_max_new_tokens — see PromptBuilder's constructor docstring.
    llm_context_window: int = 3072

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

"""Model/tokenizer loading for LLM generation (M3.3).

Loading only — per the frozen M3.3 architecture, this module owns no
generation or prompt-assembly logic. Mirrors the loading-only pattern
already established by ``index.embedding_generator``.

Per ADR-M3.3-001: the mandatory inference backend is Hugging Face
Transformers + bitsandbytes (4-bit quantization). No other engine is
permitted.

Takes explicit parameters rather than a ``Settings`` object, matching
the convention already established by ``index.bm25_pipeline`` and
``index.faiss_pipeline`` — neither imports ``config.settings`` directly;
only their ``build_*.py`` entry points read settings and pass concrete
values in. This keeps the module testable without constructing a full
``Settings`` instance.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)


def load_generation_model(
    model_name: str,
    load_in_4bit: bool,
    bnb_4bit_quant_type: str,
    bnb_4bit_compute_dtype: str,
    device_map: str,
    hf_token: str = "",
) -> tuple[Any, Any]:
    """Load the quantized generation tokenizer and model per ADR-M3.3-001.

    Args:
        model_name: HuggingFace Hub identifier for the causal LM (e.g.
            ``settings.llm_model_name``).
        load_in_4bit: Whether to load in 4-bit quantized mode.
        bnb_4bit_quant_type: bitsandbytes quantization type (e.g.
            ``"nf4"``).
        bnb_4bit_compute_dtype: Name of a ``torch`` dtype attribute (e.g.
            ``"float16"``) used for compute during 4-bit inference.
        device_map: Passed through to ``from_pretrained`` (e.g.
            ``"auto"``).
        hf_token: Hugging Face Hub token, required for gated model
            repositories. Empty string is treated as "no token".

    Returns:
        A ``(tokenizer, model)`` tuple.

    Raises:
        RuntimeError: if loading the tokenizer or model fails for any
            reason (missing weights, OOM, auth failure for a gated
            repo, invalid dtype name, etc.) — per the project's
            fail-loud convention, no partial or fallback model is ever
            returned.
    """
    try:
        compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    except AttributeError as exc:
        raise RuntimeError(
            f"Invalid bnb_4bit_compute_dtype: {bnb_4bit_compute_dtype!r} is "
            "not a valid torch dtype name."
        ) from exc

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    token = hf_token or None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            token=token,
        )
    except Exception as exc:
        logger.error("Failed to load generation model %s", model_name)
        raise RuntimeError(f"Failed to load generation model {model_name!r}") from exc

    logger.info("Loaded generation model: %s", model_name)
    return tokenizer, model

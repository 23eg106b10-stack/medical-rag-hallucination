"""NLI model and tokenizer loader for hallucination verification (Milestone 5).

Loads the fixed NLI classification checkpoint (pritamdeka/PubMedBERT-MNLI-MedNLI)
using standard Transformers AutoModelForSequenceClassification and AutoTokenizer.
Validates model.config.id2label at load time to dynamically resolve semantic
label mappings without hardcoding numeric label indices.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config.settings import get_settings

logger = logging.getLogger(__name__)

CANONICAL_LABELS = frozenset({"entailment", "neutral", "contradiction"})


class VerifierLabelMappingError(RuntimeError):
    """Raised when the NLI model's id2label configuration does not match the canonical set."""


def resolve_label_mapping(id2label: dict[int | str, str]) -> dict[str, int]:
    """Inspect and resolve model.config.id2label to canonical semantic labels.

    Args:
        id2label: Mapping from class index (int or str) to label string.

    Returns:
        Dictionary mapping canonical label names ("entailment", "neutral",
        "contradiction") to their corresponding integer class indices.

    Raises:
        VerifierLabelMappingError: If the labels cannot be resolved to exactly
            the canonical set {"entailment", "neutral", "contradiction"}.
    """
    if not isinstance(id2label, dict):
        raise VerifierLabelMappingError(
            f"Expected id2label to be a dict, got {type(id2label).__name__}."
        )

    resolved: dict[str, int] = {}
    for idx_raw, label_raw in id2label.items():
        try:
            idx = int(idx_raw)
        except (ValueError, TypeError) as exc:
            raise VerifierLabelMappingError(
                f"Invalid class index {idx_raw!r} in id2label."
            ) from exc

        label_norm = str(label_raw).strip().lower()
        if label_norm in CANONICAL_LABELS:
            if label_norm in resolved:
                raise VerifierLabelMappingError(
                    f"Duplicate mapping for canonical label {label_norm!r} in id2label: {id2label}."
                )
            resolved[label_norm] = idx

    if set(resolved.keys()) != CANONICAL_LABELS:
        raise VerifierLabelMappingError(
            f"NLI model id2label {id2label} does not resolve to the canonical "
            f"label set {sorted(CANONICAL_LABELS)}. Resolved: {resolved}."
        )

    return resolved


def load_nli_model(
    model_name: str | None = None,
    device: str | None = None,
    hf_token: str | None = None,
) -> tuple[Any, Any, dict[str, int]]:
    """Load the tokenizer and sequence classification model for NLI verification.

    Args:
        model_name: Hugging Face model identifier. If None, loaded from
            settings.verifier_model_name.
        device: Target device ("cpu" or "cuda"). If None, loaded from
            settings.verifier_device.
        hf_token: Hugging Face authentication token. If None, loaded from
            settings.hf_token.

    Returns:
        Tuple of (tokenizer, model, label_mapping) where model is set to eval
        mode and moved to the specified device.

    Raises:
        RuntimeError: If model or tokenizer fails to load.
        VerifierLabelMappingError: If model id2label mapping is invalid.
    """
    settings = get_settings()
    resolved_model_name = model_name if model_name is not None else settings.verifier_model_name
    resolved_device = device if device is not None else settings.verifier_device
    resolved_hf_token = hf_token if hf_token is not None else settings.hf_token

    token = resolved_hf_token or None

    try:
        tokenizer = AutoTokenizer.from_pretrained(resolved_model_name, token=token)
        model = AutoModelForSequenceClassification.from_pretrained(
            resolved_model_name,
            token=token,
        )
        target_device = torch.device(resolved_device)
        model.to(target_device)
        model.eval()
    except Exception as exc:
        logger.error("Failed to load NLI model %s on %s", resolved_model_name, resolved_device)
        raise RuntimeError(
            f"Failed to load NLI verifier model {resolved_model_name!r}: {exc}"
        ) from exc

    id2label = getattr(model.config, "id2label", None)
    if not id2label:
        raise VerifierLabelMappingError(
            f"Model {resolved_model_name} config does not contain an id2label mapping."
        )

    label_mapping = resolve_label_mapping(id2label)
    logger.info(
        "Loaded NLI model %s on %s with label mapping %s",
        resolved_model_name,
        resolved_device,
        label_mapping,
    )
    return tokenizer, model, label_mapping

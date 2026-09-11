"""Batched NLI inference for hallucination verification (Milestone 5).

Runs deterministic batched sequence classification scoring on (premise, hypothesis)
pairs using a loaded cross-encoder NLI model.

Enforces ADR-M5-001 & ADR-M5-003:
- PREMISE = evidence passage (HybridScoredDocument.document.abstract)
- HYPOTHESIS = extracted atomic medical claim (ExtractedClaim.claim_text)
- Internal batching and chunking: caller passes complete pair list without chunking.
- Raw probabilities only: returns NLIScore objects without thresholding or verdicts.
- Deterministic inference under torch.inference_mode().
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from schemas.verification_result import NLIScore
from verification.nli_loader import resolve_label_mapping

logger = logging.getLogger(__name__)


class NLIInferenceError(RuntimeError):
    """Raised when NLI tokenization, forward pass, or output validation fails."""


def run_nli_batch(
    pairs: list[tuple[str, str]],
    tokenizer: Any,
    model: Any,
    batch_size: int,
) -> list[NLIScore]:
    """Execute batched NLI scoring on (passage_text, claim_text) pairs.

    Args:
        pairs: List of (premise, hypothesis) tuples where premise is the
            evidence passage text and hypothesis is the extracted claim text.
        tokenizer: Hugging Face tokenizer instance.
        model: Hugging Face sequence classification model in eval mode.
        batch_size: Maximum number of pairs to evaluate per forward pass.
    Returns:
        List of NLIScore objects in the exact order of the input pairs.

    Raises:
        NLIInferenceError: If inputs are invalid, model execution crashes, or
            output contains NaN/Inf values.
    """
    if not isinstance(pairs, list):
        raise NLIInferenceError(f"Expected pairs to be a list, got {type(pairs).__name__}.")

    if not pairs:
        return []

    if batch_size <= 0:
        raise NLIInferenceError(f"batch_size must be positive, got {batch_size}.")

    # Validate pairs format
    for idx, item in enumerate(pairs):
        if not (isinstance(item, (tuple, list)) and len(item) == 2):
            raise NLIInferenceError(
                f"Pair at index {idx} must be a 2-tuple of (passage, claim), got {item!r}."
            )
        if not isinstance(item[0], str) or not isinstance(item[1], str):
            raise NLIInferenceError(
                f"Pair elements at index {idx} must be strings, got "
                f"({type(item[0])}, {type(item[1])})."
            )

    # Resolve label mapping
    id2label = getattr(getattr(model, "config", None), "id2label", None)
    if not id2label:
        raise NLIInferenceError("NLI model has no id2label configuration.")
    try:
        label_mapping = resolve_label_mapping(id2label)
    except Exception as exc:
        raise NLIInferenceError(f"Failed to resolve label mapping: {exc}") from exc

    try:
        ent_idx = label_mapping["entailment"]
        neu_idx = label_mapping["neutral"]
        con_idx = label_mapping["contradiction"]
    except KeyError as exc:
        raise NLIInferenceError(f"Missing required canonical label in mapping: {exc}") from exc

    # Identify device
    try:
        device = next(model.parameters()).device
    except (StopIteration, AttributeError):
        device = torch.device("cpu")

    results: list[NLIScore] = []
    total_pairs = len(pairs)

    for start_idx in range(0, total_pairs, batch_size):
        end_idx = min(start_idx + batch_size, total_pairs)
        batch_pairs = pairs[start_idx:end_idx]

        premises = [p[0] for p in batch_pairs]
        hypotheses = [p[1] for p in batch_pairs]

        try:
            encoded = tokenizer(
                premises,
                hypotheses,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
        except Exception as exc:
            raise NLIInferenceError(
                f"Tokenization failed for batch {start_idx}:{end_idx}: {exc}"
            ) from exc

        try:
            with torch.inference_mode():
                outputs = model(**encoded)
        except Exception as exc:
            raise NLIInferenceError(
                f"Model forward pass failed for batch {start_idx}:{end_idx}: {exc}"
            ) from exc

        logits = getattr(outputs, "logits", None)
        if logits is None:
            raise NLIInferenceError(f"Model output has no 'logits' attribute: {type(outputs)}.")

        if torch.isnan(logits).any() or torch.isinf(logits).any():
            raise NLIInferenceError(
                f"Model output contains NaN or Inf logits in batch {start_idx}:{end_idx}."
            )

        probs = torch.softmax(logits, dim=-1)

        for i in range(len(batch_pairs)):
            ent_prob = float(probs[i, ent_idx].item())
            neu_prob = float(probs[i, neu_idx].item())
            con_prob = float(probs[i, con_idx].item())

            results.append(
                NLIScore(
                    passage_pmid="",
                    entailment_prob=ent_prob,
                    neutral_prob=neu_prob,
                    contradiction_prob=con_prob,
                )
            )

    return results

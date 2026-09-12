"""Confidence scoring for generated answers based on verification results (Milestone 6).

Implements deterministic, pure answer-level confidence scoring over frozen M5
`list[ClaimVerification]` outputs.

Enforces ADR-M6-001 through ADR-M6-005:
- ADR-M6-001: Contradiction hard ceiling. If contradicted_count > 0, score is capped
  at contradiction_ceiling. contradiction_ceiling_applied is True only when the ceiling
  actually constrained the score (base_score > contradiction_ceiling).
- ADR-M6-002: UNVERIFIABLE claims contribute to total_claims (denominator dilution)
  without explicit additional penalty and do not trigger the contradiction ceiling.
- ADR-M6-003: Verdict-count scoring. NLI softmax probabilities are excluded from the
  headline score formula and remain in the M5 audit trail.
- ADR-M6-004: Counts-only ConfidenceResult schema; zero claims produces score=None,
  level="NOT_APPLICABLE", contradiction_ceiling_applied=False.
- ADR-M6-005: Threshold defaults (0.2, 0.8, 0.5) are provisional and uncalibrated.
"""

from __future__ import annotations

from typing import Literal

from schemas.confidence_result import ConfidenceResult
from schemas.verification_result import ClaimVerification


class ConfidenceInputError(RuntimeError):
    """Raised when input to confidence scoring violates type or invariant contracts."""


def _validate_thresholds(
    contradiction_ceiling: float,
    level_high_threshold: float,
    level_medium_threshold: float,
) -> None:
    """Validate threshold ranges and ordering invariants.

    Raises:
        ConfidenceInputError: if any threshold is not a valid float/int,
            falls outside [0.0, 1.0], or if level_high_threshold < level_medium_threshold.
    """
    for name, val in [
        ("contradiction_ceiling", contradiction_ceiling),
        ("level_high_threshold", level_high_threshold),
        ("level_medium_threshold", level_medium_threshold),
    ]:
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise ConfidenceInputError(
                f"Threshold {name!r} must be a real number, got {type(val).__name__} ({val!r})."
            )
        if not (0.0 <= float(val) <= 1.0):
            raise ConfidenceInputError(f"Threshold {name!r} must be in [0.0, 1.0], got {val}.")

    if level_high_threshold < level_medium_threshold:
        raise ConfidenceInputError(
            f"level_high_threshold ({level_high_threshold}) must be greater than or "
            f"equal to level_medium_threshold ({level_medium_threshold})."
        )


def _categorize(
    score: float | None,
    level_high_threshold: float,
    level_medium_threshold: float,
) -> Literal["HIGH", "MEDIUM", "LOW", "NOT_APPLICABLE"]:
    """Categorize a confidence score into discrete confidence levels.

    Boundaries are inclusive on the higher category (>=).
    """
    if score is None:
        return "NOT_APPLICABLE"
    if score >= level_high_threshold:
        return "HIGH"
    if score >= level_medium_threshold:
        return "MEDIUM"
    return "LOW"


def score_confidence(
    verifications: list[ClaimVerification],
    contradiction_ceiling: float,
    level_high_threshold: float,
    level_medium_threshold: float,
) -> ConfidenceResult:
    """Score answer-level confidence deterministically from claim verifications.

    Args:
        verifications: List of `ClaimVerification` results from M5.
        contradiction_ceiling: Maximum allowed score when at least one claim is contradicted.
        level_high_threshold: Minimum score required for HIGH confidence level.
        level_medium_threshold: Minimum score required for MEDIUM confidence level.

    Returns:
        A `ConfidenceResult` containing score, categorical level, diagnostic counts,
        and ceiling application flag.

    Raises:
        ConfidenceInputError: if `verifications` is not a list of `ClaimVerification`
            instances, or if thresholds violate required invariants.
    """
    # 1. Validate inputs
    if not isinstance(verifications, list):
        raise ConfidenceInputError(
            f"verifications must be a list, got {type(verifications).__name__}."
        )

    for i, v in enumerate(verifications):
        if not isinstance(v, ClaimVerification):
            raise ConfidenceInputError(
                f"Item at index {i} in verifications is not a ClaimVerification "
                f"instance: {type(v).__name__}."
            )

    _validate_thresholds(
        contradiction_ceiling=contradiction_ceiling,
        level_high_threshold=level_high_threshold,
        level_medium_threshold=level_medium_threshold,
    )

    # 2. Handle zero claims (ADR-M6-004)
    n = len(verifications)
    if n == 0:
        return ConfidenceResult(
            score=None,
            level="NOT_APPLICABLE",
            total_claims=0,
            supported_count=0,
            contradicted_count=0,
            unverifiable_count=0,
            contradiction_ceiling_applied=False,
        )

    # 3. Verdict counting
    n_supported = sum(1 for v in verifications if v.verdict == "SUPPORTED")
    n_contradicted = sum(1 for v in verifications if v.verdict == "CONTRADICTED")
    n_unverifiable = sum(1 for v in verifications if v.verdict == "UNVERIFIABLE")

    # 4. Base score and contradiction hard ceiling (ADR-M6-001, ADR-M6-002)
    base_score = n_supported / n
    ceiling_applied = n_contradicted > 0 and base_score > contradiction_ceiling
    score = min(base_score, contradiction_ceiling) if n_contradicted > 0 else base_score

    # 5. Categorization
    level = _categorize(
        score=score,
        level_high_threshold=level_high_threshold,
        level_medium_threshold=level_medium_threshold,
    )

    return ConfidenceResult(
        score=score,
        level=level,
        total_claims=n,
        supported_count=n_supported,
        contradicted_count=n_contradicted,
        unverifiable_count=n_unverifiable,
        contradiction_ceiling_applied=ceiling_applied,
    )

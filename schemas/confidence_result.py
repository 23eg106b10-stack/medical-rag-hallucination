"""Schemas for answer-level confidence scoring results (Milestone 6)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, model_validator


class ConfidenceResult(BaseModel):
    """Answer-level confidence derived from M5 claim verifications.

    ``score`` is an uncalibrated, verification-derived heuristic
    representing the proportion of extracted claims that received a
    SUPPORTED verdict, subject to the contradiction hard-ceiling
    policy. It is NOT a statistically calibrated probability of
    factual correctness, NOT a clinical risk probability.
    """

    score: float | None
    level: Literal["HIGH", "MEDIUM", "LOW", "NOT_APPLICABLE"]
    total_claims: int
    supported_count: int
    contradicted_count: int
    unverifiable_count: int
    contradiction_ceiling_applied: bool

    @model_validator(mode="after")
    def _validate_invariants(self) -> ConfidenceResult:
        """Validate frozen schema invariants across counts, score, and level."""
        # Non-negative counts
        if (
            self.total_claims < 0
            or self.supported_count < 0
            or self.contradicted_count < 0
            or self.unverifiable_count < 0
        ):
            raise ValueError("All claim counts must be non-negative (>= 0).")

        # Count additivity invariant
        if self.total_claims != (
            self.supported_count + self.contradicted_count + self.unverifiable_count
        ):
            raise ValueError(
                f"total_claims ({self.total_claims}) must equal sum of "
                f"supported ({self.supported_count}) + contradicted ({self.contradicted_count}) "
                f"+ unverifiable ({self.unverifiable_count})."
            )

        # Zero-claim invariants
        if self.total_claims == 0:
            if self.score is not None:
                raise ValueError("score must be None when total_claims == 0.")
            if self.level != "NOT_APPLICABLE":
                raise ValueError(
                    f"level must be 'NOT_APPLICABLE' when total_claims == 0, got {self.level!r}."
                )
            if self.contradiction_ceiling_applied:
                raise ValueError(
                    "contradiction_ceiling_applied must be False when total_claims == 0."
                )
        else:
            # Non-zero claim invariants
            if self.score is None:
                raise ValueError("score must not be None when total_claims > 0.")
            if not (0.0 <= self.score <= 1.0):
                raise ValueError(f"score must be in [0.0, 1.0], got {self.score}.")
            if self.level not in {"HIGH", "MEDIUM", "LOW"}:
                raise ValueError(
                    "level must be one of HIGH, MEDIUM, LOW when total_claims > 0, "
                    f"got {self.level!r}."
                )

        return self

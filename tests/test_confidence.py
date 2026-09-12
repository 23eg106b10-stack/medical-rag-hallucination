"""Unit tests for M6 answer-level confidence scoring (verification/confidence.py).

Enforces ADR-M6-001 through ADR-M6-005:
- ADR-M6-001: Contradiction hard-ceiling policy and precise flag semantics
- ADR-M6-002: UNVERIFIABLE denominator dilution without explicit penalty
- ADR-M6-003: Verdict-count scoring; NLI probabilities excluded from formula
- ADR-M6-004: Counts-only ConfidenceResult schema and zero-claim sentinel semantics
- ADR-M6-005: Provisional uncalibrated thresholds (0.2, 0.8, 0.5)
"""

from __future__ import annotations

import pytest

from schemas.verification_result import (
    ClaimVerification,
    EvidenceAttribution,
    NLIScore,
)
from verification.confidence import ConfidenceInputError, score_confidence


def _make_claim(
    text: str,
    verdict: str,
    entailment_prob: float = 0.8,
    contradiction_prob: float = 0.1,
    neutral_prob: float = 0.1,
) -> ClaimVerification:
    """Helper to construct a ClaimVerification with dummy NLI scores."""
    nli_score = NLIScore(
        passage_pmid="12345678",
        entailment_prob=entailment_prob,
        contradiction_prob=contradiction_prob,
        neutral_prob=neutral_prob,
    )
    attribution = EvidenceAttribution(
        attributed_pmid="12345678" if verdict != "UNVERIFIABLE" else None,
        all_scores=[nli_score],
    )
    return ClaimVerification(
        claim_text=text,
        verdict=verdict,  # type: ignore[arg-type]
        evidence=attribution,
    )


# ── Basic Scoring & Verdict Coverage ─────────────────────────────────────────


def test_score_confidence_all_supported() -> None:
    """All SUPPORTED claims produce score=1.0 and level=HIGH."""
    claims = [
        _make_claim("Claim 1", "SUPPORTED"),
        _make_claim("Claim 2", "SUPPORTED"),
        _make_claim("Claim 3", "SUPPORTED"),
    ]
    result = score_confidence(
        verifications=claims,
        contradiction_ceiling=0.2,
        level_high_threshold=0.8,
        level_medium_threshold=0.5,
    )
    assert result.score == 1.0
    assert result.level == "HIGH"
    assert result.total_claims == 3
    assert result.supported_count == 3
    assert result.contradicted_count == 0
    assert result.unverifiable_count == 0
    assert result.contradiction_ceiling_applied is False


def test_score_confidence_all_unverifiable() -> None:
    """All UNVERIFIABLE claims produce score=0.0 and level=LOW with no ceiling."""
    claims = [
        _make_claim("Claim 1", "UNVERIFIABLE"),
        _make_claim("Claim 2", "UNVERIFIABLE"),
    ]
    result = score_confidence(
        verifications=claims,
        contradiction_ceiling=0.2,
        level_high_threshold=0.8,
        level_medium_threshold=0.5,
    )
    assert result.score == 0.0
    assert result.level == "LOW"
    assert result.total_claims == 2
    assert result.supported_count == 0
    assert result.contradicted_count == 0
    assert result.unverifiable_count == 2
    assert result.contradiction_ceiling_applied is False


def test_score_confidence_all_contradicted() -> None:
    """All CONTRADICTED claims produce score=0.0; ceiling does not actively constrain."""
    claims = [
        _make_claim("Claim 1", "CONTRADICTED"),
        _make_claim("Claim 2", "CONTRADICTED"),
    ]
    result = score_confidence(
        verifications=claims,
        contradiction_ceiling=0.2,
        level_high_threshold=0.8,
        level_medium_threshold=0.5,
    )
    # base_score = 0/2 = 0.0 <= 0.2, so ceiling does not actively constrain
    assert result.score == 0.0
    assert result.level == "LOW"
    assert result.total_claims == 2
    assert result.supported_count == 0
    assert result.contradicted_count == 2
    assert result.unverifiable_count == 0
    assert result.contradiction_ceiling_applied is False


def test_score_confidence_mixed_supported_unverifiable_dilution() -> None:
    """UNVERIFIABLE claims dilute denominator without triggering ceiling (ADR-M6-002)."""
    # 3 supported out of 5 total = 3/5 = 0.60
    claims = [
        _make_claim("Claim 1", "SUPPORTED"),
        _make_claim("Claim 2", "SUPPORTED"),
        _make_claim("Claim 3", "SUPPORTED"),
        _make_claim("Claim 4", "UNVERIFIABLE"),
        _make_claim("Claim 5", "UNVERIFIABLE"),
    ]
    result = score_confidence(
        verifications=claims,
        contradiction_ceiling=0.2,
        level_high_threshold=0.8,
        level_medium_threshold=0.5,
    )
    assert result.score == pytest.approx(0.60)
    assert result.level == "MEDIUM"
    assert result.total_claims == 5
    assert result.supported_count == 3
    assert result.contradicted_count == 0
    assert result.unverifiable_count == 2
    assert result.contradiction_ceiling_applied is False


def test_score_confidence_mixed_all_three_verdicts() -> None:
    """Mixture of all three verdicts with ceiling actively constraining."""
    # 8 supported, 1 contradicted, 1 unverifiable = base_score 8/10 = 0.80
    # Ceiling = 0.20 caps score at 0.20
    claims = [_make_claim(f"S{i}", "SUPPORTED") for i in range(8)]
    claims.append(_make_claim("C1", "CONTRADICTED"))
    claims.append(_make_claim("U1", "UNVERIFIABLE"))

    result = score_confidence(
        verifications=claims,
        contradiction_ceiling=0.2,
        level_high_threshold=0.8,
        level_medium_threshold=0.5,
    )
    assert result.score == 0.2
    assert result.level == "LOW"
    assert result.total_claims == 10
    assert result.supported_count == 8
    assert result.contradicted_count == 1
    assert result.unverifiable_count == 1
    assert result.contradiction_ceiling_applied is True


# ── Contradiction Ceiling Semantics (ADR-M6-001) ─────────────────────────────


def test_score_confidence_ceiling_actively_constrains() -> None:
    """Case A: base_score > ceiling, contradiction present -> ceiling_applied is True."""
    # 2 supported, 1 contradicted -> base_score = 2/3 ~ 0.667 > 0.20
    claims = [
        _make_claim("Claim 1", "SUPPORTED"),
        _make_claim("Claim 2", "SUPPORTED"),
        _make_claim("Claim 3", "CONTRADICTED"),
    ]
    result = score_confidence(
        verifications=claims,
        contradiction_ceiling=0.2,
        level_high_threshold=0.8,
        level_medium_threshold=0.5,
    )
    assert result.score == 0.2
    assert result.contradiction_ceiling_applied is True
    assert result.level == "LOW"


def test_score_confidence_ceiling_does_not_constrain_when_base_score_below_ceiling() -> None:
    """Case B: base_score <= ceiling, contradiction present -> ceiling_applied is False."""
    # 1 supported, 9 contradicted -> base_score = 1/10 = 0.10 <= 0.20
    claims = [_make_claim("Claim 1", "SUPPORTED")] + [
        _make_claim(f"C{i}", "CONTRADICTED") for i in range(9)
    ]
    result = score_confidence(
        verifications=claims,
        contradiction_ceiling=0.2,
        level_high_threshold=0.8,
        level_medium_threshold=0.5,
    )
    assert result.score == pytest.approx(0.10)
    assert result.contradiction_ceiling_applied is False
    assert result.level == "LOW"


def test_score_confidence_ceiling_exact_equality() -> None:
    """When base_score == contradiction_ceiling, ceiling did not constrain (flag False)."""
    # 1 supported, 4 contradicted -> base_score = 1/5 = 0.20 == ceiling 0.20
    claims = [_make_claim("Claim 1", "SUPPORTED")] + [
        _make_claim(f"C{i}", "CONTRADICTED") for i in range(4)
    ]
    result = score_confidence(
        verifications=claims,
        contradiction_ceiling=0.2,
        level_high_threshold=0.8,
        level_medium_threshold=0.5,
    )
    assert result.score == pytest.approx(0.20)
    assert result.contradiction_ceiling_applied is False


# ── Zero-Claim Semantics (ADR-M6-004) ─────────────────────────────────────────


def test_score_confidence_zero_claims() -> None:
    """Empty verifications produces score=None, level=NOT_APPLICABLE, counts=0."""
    result = score_confidence(
        verifications=[],
        contradiction_ceiling=0.2,
        level_high_threshold=0.8,
        level_medium_threshold=0.5,
    )
    assert result.score is None
    assert result.level == "NOT_APPLICABLE"
    assert result.total_claims == 0
    assert result.supported_count == 0
    assert result.contradicted_count == 0
    assert result.unverifiable_count == 0
    assert result.contradiction_ceiling_applied is False


# ── Level Categorization Boundaries ──────────────────────────────────────────


def test_score_confidence_exact_high_threshold() -> None:
    """Exact high threshold score produces HIGH (inclusive >=)."""
    # 8 supported, 2 unverifiable = 8/10 = 0.80
    claims = [_make_claim(f"S{i}", "SUPPORTED") for i in range(8)] + [
        _make_claim(f"U{i}", "UNVERIFIABLE") for i in range(2)
    ]
    result = score_confidence(
        verifications=claims,
        contradiction_ceiling=0.2,
        level_high_threshold=0.8,
        level_medium_threshold=0.5,
    )
    assert result.score == pytest.approx(0.80)
    assert result.level == "HIGH"


def test_score_confidence_exact_medium_threshold() -> None:
    """Exact medium threshold score produces MEDIUM (inclusive >=)."""
    # 1 supported, 1 unverifiable = 1/2 = 0.50
    claims = [
        _make_claim("Claim 1", "SUPPORTED"),
        _make_claim("Claim 2", "UNVERIFIABLE"),
    ]
    result = score_confidence(
        verifications=claims,
        contradiction_ceiling=0.2,
        level_high_threshold=0.8,
        level_medium_threshold=0.5,
    )
    assert result.score == pytest.approx(0.50)
    assert result.level == "MEDIUM"


def test_score_confidence_just_below_medium_threshold() -> None:
    """Score strictly below medium threshold produces LOW."""
    # 4 supported, 6 unverifiable = 4/10 = 0.40 < 0.50
    claims = [_make_claim(f"S{i}", "SUPPORTED") for i in range(4)] + [
        _make_claim(f"U{i}", "UNVERIFIABLE") for i in range(6)
    ]
    result = score_confidence(
        verifications=claims,
        contradiction_ceiling=0.2,
        level_high_threshold=0.8,
        level_medium_threshold=0.5,
    )
    assert result.score == pytest.approx(0.40)
    assert result.level == "LOW"


# ── Threshold Validation Invariants ──────────────────────────────────────────


@pytest.mark.parametrize(
    ("ceiling", "high", "medium"),
    [
        (-0.1, 0.8, 0.5),  # ceiling < 0
        (1.1, 0.8, 0.5),  # ceiling > 1
        (0.2, -0.1, 0.5),  # high < 0
        (0.2, 1.1, 0.5),  # high > 1
        (0.2, 0.8, -0.1),  # medium < 0
        (0.2, 0.8, 1.1),  # medium > 1
        (0.2, 0.4, 0.6),  # high < medium (inversion)
    ],
)
def test_score_confidence_invalid_thresholds_raises(
    ceiling: float, high: float, medium: float
) -> None:
    """Invalid or out-of-range thresholds raise ConfidenceInputError."""
    with pytest.raises(ConfidenceInputError):
        score_confidence(
            verifications=[],
            contradiction_ceiling=ceiling,
            level_high_threshold=high,
            level_medium_threshold=medium,
        )


@pytest.mark.parametrize(
    ("ceiling", "high", "medium"),
    [
        ("0.2", 0.8, 0.5),  # type: ignore[arg-type]
        (0.2, True, 0.5),  # type: ignore[arg-type]
        (0.2, 0.8, None),  # type: ignore[arg-type]
    ],
)
def test_score_confidence_non_numeric_thresholds_raises(
    ceiling: float, high: float, medium: float
) -> None:
    """Non-numeric or boolean thresholds raise ConfidenceInputError."""
    with pytest.raises(ConfidenceInputError):
        score_confidence(
            verifications=[],
            contradiction_ceiling=ceiling,
            level_high_threshold=high,
            level_medium_threshold=medium,
        )


# ── Input Validation ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("invalid_input", [None, "claims", {"a": 1}, (1, 2, 3), 42])
def test_score_confidence_non_list_input_raises(invalid_input: object) -> None:
    """Non-list verifications input raises ConfidenceInputError."""
    with pytest.raises(ConfidenceInputError):
        score_confidence(
            verifications=invalid_input,  # type: ignore[arg-type]
            contradiction_ceiling=0.2,
            level_high_threshold=0.8,
            level_medium_threshold=0.5,
        )


def test_score_confidence_list_with_invalid_element_raises() -> None:
    """List containing items other than ClaimVerification raises ConfidenceInputError."""
    claims = [_make_claim("Valid", "SUPPORTED"), "not a claim"]
    with pytest.raises(ConfidenceInputError):
        score_confidence(
            verifications=claims,  # type: ignore[arg-type]
            contradiction_ceiling=0.2,
            level_high_threshold=0.8,
            level_medium_threshold=0.5,
        )


# ── NLI Probability Exclusion (ADR-M6-003) ───────────────────────────────────


def test_score_confidence_nli_probabilities_do_not_affect_score() -> None:
    """Different NLI probability distributions produce identical ConfidenceResult."""
    # List A: High entailment probabilities
    claims_a = [
        _make_claim("Claim 1", "SUPPORTED", entailment_prob=0.99, contradiction_prob=0.01),
        _make_claim("Claim 2", "SUPPORTED", entailment_prob=0.95, contradiction_prob=0.05),
    ]
    # List B: Marginal entailment probabilities (just above threshold)
    claims_b = [
        _make_claim("Claim 1", "SUPPORTED", entailment_prob=0.51, contradiction_prob=0.49),
        _make_claim("Claim 2", "SUPPORTED", entailment_prob=0.52, contradiction_prob=0.48),
    ]

    result_a = score_confidence(claims_a, 0.2, 0.8, 0.5)
    result_b = score_confidence(claims_b, 0.2, 0.8, 0.5)

    assert result_a.score == result_b.score == 1.0
    assert result_a.level == result_b.level == "HIGH"
    assert result_a.supported_count == result_b.supported_count == 2
    assert result_a == result_b


# ── Determinism & Immutability ────────────────────────────────────────────────


def test_score_confidence_determinism() -> None:
    """Identical input and configuration deterministically produces identical output."""
    claims = [
        _make_claim("Claim 1", "SUPPORTED"),
        _make_claim("Claim 2", "CONTRADICTED"),
        _make_claim("Claim 3", "UNVERIFIABLE"),
    ]
    res1 = score_confidence(claims, 0.2, 0.8, 0.5)
    res2 = score_confidence(claims, 0.2, 0.8, 0.5)
    assert res1 == res2
    assert res1.model_dump() == res2.model_dump()


def test_score_confidence_does_not_mutate_input() -> None:
    """score_confidence does not modify the input list or its ClaimVerification elements."""
    claims = [
        _make_claim("Claim 1", "SUPPORTED"),
        _make_claim("Claim 2", "CONTRADICTED"),
    ]
    orig_len = len(claims)
    orig_c0_verdict = claims[0].verdict
    orig_c1_verdict = claims[1].verdict

    _ = score_confidence(claims, 0.2, 0.8, 0.5)

    assert len(claims) == orig_len
    assert claims[0].verdict == orig_c0_verdict
    assert claims[1].verdict == orig_c1_verdict

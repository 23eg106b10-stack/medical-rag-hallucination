"""Tests for the shared schemas package."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from schemas.common import Document, RetrievedDocument
from schemas.confidence_result import ConfidenceResult
from schemas.requests import QuestionRequest
from schemas.responses import AnswerResponse
from schemas.verification_result import (
    ClaimVerification,
    EvidenceAttribution,
    NLIScore,
)


def test_document_valid_construction() -> None:
    """Document accepts its three required string fields."""
    doc = Document(id="1", title="Title", text="Body text")
    assert doc.id == "1"
    assert doc.title == "Title"
    assert doc.text == "Body text"


@pytest.mark.parametrize("missing_field", ["id", "title", "text"])
def test_document_missing_required_field_raises(missing_field: str) -> None:
    """Document construction fails if any required field is absent."""
    fields = {"id": "1", "title": "Title", "text": "Body text"}
    del fields[missing_field]
    with pytest.raises(ValidationError):
        Document(**fields)


def test_retrieved_document_valid_construction_with_nested_document() -> None:
    """RetrievedDocument accepts a nested Document plus a score."""
    doc = Document(id="1", title="Title", text="Body text")
    retrieved = RetrievedDocument(document=doc, score=0.87)
    assert retrieved.document == doc
    assert retrieved.score == 0.87


def test_retrieved_document_accepts_nested_dict_and_validates_it() -> None:
    """RetrievedDocument builds its nested Document from a dict."""
    retrieved = RetrievedDocument(
        document={"id": "1", "title": "Title", "text": "Body text"},
        score=0.5,
    )
    assert isinstance(retrieved.document, Document)


def test_retrieved_document_invalid_nested_document_raises() -> None:
    """An invalid nested Document (missing field) fails validation."""
    with pytest.raises(ValidationError):
        RetrievedDocument(document={"id": "1", "title": "Title"}, score=0.5)


def test_retrieved_document_missing_score_raises() -> None:
    """RetrievedDocument requires a score."""
    doc = Document(id="1", title="Title", text="Body text")
    with pytest.raises(ValidationError):
        RetrievedDocument(document=doc)


def test_question_request_valid_construction() -> None:
    """QuestionRequest accepts a question string."""
    req = QuestionRequest(question="What is the treatment?")
    assert req.question == "What is the treatment?"


def test_question_request_missing_question_raises() -> None:
    """QuestionRequest requires the question field."""
    with pytest.raises(ValidationError):
        QuestionRequest()


def test_answer_response_valid_construction() -> None:
    """AnswerResponse accepts an answer string."""
    resp = AnswerResponse(answer="The treatment is...")
    assert resp.answer == "The treatment is..."


def test_answer_response_missing_answer_raises() -> None:
    """AnswerResponse requires the answer field."""
    with pytest.raises(ValidationError):
        AnswerResponse()


# ── Milestone 5 Verification Schemas ─────────────────────────────────────────


def test_nli_score_valid_construction() -> None:
    """NLIScore accepts passage_pmid and float probabilities."""
    score = NLIScore(
        passage_pmid="12345",
        entailment_prob=0.85,
        neutral_prob=0.10,
        contradiction_prob=0.05,
    )
    assert score.passage_pmid == "12345"
    assert score.entailment_prob == 0.85
    assert score.neutral_prob == 0.10
    assert score.contradiction_prob == 0.05


@pytest.mark.parametrize(
    "missing_field",
    ["passage_pmid", "entailment_prob", "neutral_prob", "contradiction_prob"],
)
def test_nli_score_missing_field_raises(missing_field: str) -> None:
    """NLIScore requires all probability and pmid fields."""
    fields = {
        "passage_pmid": "12345",
        "entailment_prob": 0.8,
        "neutral_prob": 0.1,
        "contradiction_prob": 0.1,
    }
    del fields[missing_field]
    with pytest.raises(ValidationError):
        NLIScore(**fields)


def test_evidence_attribution_valid_with_pmid_and_scores() -> None:
    """EvidenceAttribution accepts an attributed_pmid and list of NLIScore."""
    score = NLIScore(
        passage_pmid="12345",
        entailment_prob=0.9,
        neutral_prob=0.05,
        contradiction_prob=0.05,
    )
    attribution = EvidenceAttribution(
        attributed_pmid="12345",
        all_scores=[score],
    )
    assert attribution.attributed_pmid == "12345"
    assert len(attribution.all_scores) == 1
    assert attribution.all_scores[0].passage_pmid == "12345"


def test_evidence_attribution_optional_pmid_none() -> None:
    """EvidenceAttribution accepts None for attributed_pmid (e.g. UNVERIFIABLE)."""
    attribution = EvidenceAttribution(
        attributed_pmid=None,
        all_scores=[],
    )
    assert attribution.attributed_pmid is None
    assert attribution.all_scores == []


def test_evidence_attribution_missing_all_scores_raises() -> None:
    """EvidenceAttribution requires the all_scores list."""
    with pytest.raises(ValidationError):
        EvidenceAttribution(attributed_pmid=None)


@pytest.mark.parametrize("verdict", ["SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"])
def test_claim_verification_valid_verdicts(verdict: str) -> None:
    """ClaimVerification accepts the three frozen verdicts."""
    attribution = EvidenceAttribution(attributed_pmid=None, all_scores=[])
    cv = ClaimVerification(
        claim_text="Aspirin reduces fever.",
        verdict=verdict,  # type: ignore[arg-type]
        evidence=attribution,
    )
    assert cv.claim_text == "Aspirin reduces fever."
    assert cv.verdict == verdict
    assert cv.evidence == attribution


@pytest.mark.parametrize("invalid_verdict", ["CONFLICTING", "SUPPORT", "REFUTED", "UNKNOWN", ""])
def test_claim_verification_invalid_verdict_raises(invalid_verdict: str) -> None:
    """ClaimVerification rejects unapproved verdicts (e.g. CONFLICTING or REFUTED)."""
    attribution = EvidenceAttribution(attributed_pmid=None, all_scores=[])
    with pytest.raises(ValidationError):
        ClaimVerification(
            claim_text="Aspirin reduces fever.",
            verdict=invalid_verdict,  # type: ignore[arg-type]
            evidence=attribution,
        )


def test_claim_verification_missing_fields_raises() -> None:
    """ClaimVerification requires claim_text, verdict, and evidence."""
    with pytest.raises(ValidationError):
        ClaimVerification(claim_text="Test claim", verdict="SUPPORTED")  # type: ignore[call-arg]


# ── Milestone 6 Confidence Schemas ─────────────────────────────────────────


def test_confidence_result_valid_construction() -> None:
    """ConfidenceResult accepts valid score, level, and consistent counts."""
    result = ConfidenceResult(
        score=0.75,
        level="MEDIUM",
        total_claims=4,
        supported_count=3,
        contradicted_count=1,
        unverifiable_count=0,
        contradiction_ceiling_applied=True,
    )
    assert result.score == 0.75
    assert result.level == "MEDIUM"
    assert result.total_claims == 4
    assert result.supported_count == 3
    assert result.contradicted_count == 1
    assert result.unverifiable_count == 0
    assert result.contradiction_ceiling_applied is True


def test_confidence_result_zero_claims_valid_construction() -> None:
    """ConfidenceResult accepts valid zero-claim sentinel values."""
    result = ConfidenceResult(
        score=None,
        level="NOT_APPLICABLE",
        total_claims=0,
        supported_count=0,
        contradicted_count=0,
        unverifiable_count=0,
        contradiction_ceiling_applied=False,
    )
    assert result.score is None
    assert result.level == "NOT_APPLICABLE"
    assert result.total_claims == 0
    assert result.contradiction_ceiling_applied is False


@pytest.mark.parametrize("invalid_level", ["MODERATE", "VERY_HIGH", "UNKNOWN", ""])
def test_confidence_result_invalid_level_raises(invalid_level: str) -> None:
    """ConfidenceResult rejects invalid level strings."""
    with pytest.raises(ValidationError):
        ConfidenceResult(
            score=0.8,
            level=invalid_level,  # type: ignore[arg-type]
            total_claims=1,
            supported_count=1,
            contradicted_count=0,
            unverifiable_count=0,
            contradiction_ceiling_applied=False,
        )


def test_confidence_result_count_mismatch_raises() -> None:
    """ConfidenceResult raises ValidationError if total_claims != s + c + u."""
    with pytest.raises(ValidationError):
        ConfidenceResult(
            score=0.5,
            level="MEDIUM",
            total_claims=5,  # Mismatch: 2 + 1 + 0 = 3 != 5
            supported_count=2,
            contradicted_count=1,
            unverifiable_count=0,
            contradiction_ceiling_applied=False,
        )


@pytest.mark.parametrize(
    ("supported", "contradicted", "unverifiable"),
    [(-1, 0, 0), (0, -1, 0), (0, 0, -1)],
)
def test_confidence_result_negative_counts_raises(
    supported: int, contradicted: int, unverifiable: int
) -> None:
    """ConfidenceResult raises ValidationError if any claim count is negative."""
    total = supported + contradicted + unverifiable
    with pytest.raises(ValidationError):
        ConfidenceResult(
            score=0.5,
            level="MEDIUM",
            total_claims=total,
            supported_count=supported,
            contradicted_count=contradicted,
            unverifiable_count=unverifiable,
            contradiction_ceiling_applied=False,
        )


@pytest.mark.parametrize("invalid_score", [-0.01, 1.01, 2.0])
def test_confidence_result_score_range_raises(invalid_score: float) -> None:
    """ConfidenceResult raises ValidationError if non-zero score is outside [0.0, 1.0]."""
    with pytest.raises(ValidationError):
        ConfidenceResult(
            score=invalid_score,
            level="HIGH",
            total_claims=1,
            supported_count=1,
            contradicted_count=0,
            unverifiable_count=0,
            contradiction_ceiling_applied=False,
        )


def test_confidence_result_zero_claims_invariants_raises() -> None:
    """ConfidenceResult raises ValidationError if zero-claim sentinel rules are violated."""
    # score cannot be non-None when total_claims == 0
    with pytest.raises(ValidationError):
        ConfidenceResult(
            score=0.0,
            level="NOT_APPLICABLE",
            total_claims=0,
            supported_count=0,
            contradicted_count=0,
            unverifiable_count=0,
            contradiction_ceiling_applied=False,
        )

    # level cannot be anything other than NOT_APPLICABLE when total_claims == 0
    with pytest.raises(ValidationError):
        ConfidenceResult(
            score=None,
            level="LOW",
            total_claims=0,
            supported_count=0,
            contradicted_count=0,
            unverifiable_count=0,
            contradiction_ceiling_applied=False,
        )

    # ceiling_applied cannot be True when total_claims == 0
    with pytest.raises(ValidationError):
        ConfidenceResult(
            score=None,
            level="NOT_APPLICABLE",
            total_claims=0,
            supported_count=0,
            contradicted_count=0,
            unverifiable_count=0,
            contradiction_ceiling_applied=True,
        )


def test_confidence_result_non_zero_claims_invariants_raises() -> None:
    """ConfidenceResult raises ValidationError if non-zero claim invariants are violated."""
    # score cannot be None when total_claims > 0
    with pytest.raises(ValidationError):
        ConfidenceResult(
            score=None,
            level="LOW",
            total_claims=1,
            supported_count=1,
            contradicted_count=0,
            unverifiable_count=0,
            contradiction_ceiling_applied=False,
        )

    # level cannot be NOT_APPLICABLE when total_claims > 0
    with pytest.raises(ValidationError):
        ConfidenceResult(
            score=0.5,
            level="NOT_APPLICABLE",
            total_claims=1,
            supported_count=1,
            contradicted_count=0,
            unverifiable_count=0,
            contradiction_ceiling_applied=False,
        )

"""Unit tests for M5 ClaimVerifier orchestration layer."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from schemas.corpus import CorpusDocument
from schemas.retrieval import HybridScoredDocument
from schemas.verification import ExtractedClaim
from schemas.verification_result import NLIScore
from verification.nli_inference import NLIInferenceError
from verification.verifier import ClaimVerifier, VerificationInputError


def _make_doc(pmid: str, abstract: str, title: str = "Title") -> HybridScoredDocument:
    return HybridScoredDocument(
        document=CorpusDocument(pmid=pmid, title=title, abstract=abstract),
        sparse_score=1.0,
        dense_score=1.0,
        rrf_score=0.03,
    )


def test_verify_claims_zero_claims_returns_empty_list() -> None:
    """Zero claims returns [] without running any inference."""
    mock_inference = MagicMock()
    verifier = ClaimVerifier(nli_inference_fn=mock_inference)
    context = [_make_doc("100", "Some abstract.")]

    results = verifier.verify_claims([], context)
    assert results == []
    mock_inference.assert_not_called()


def test_verify_claims_empty_context_marks_all_unverifiable() -> None:
    """Empty context returns UNVERIFIABLE for all claims with zero NLI calls."""
    mock_inference = MagicMock()
    verifier = ClaimVerifier(nli_inference_fn=mock_inference)
    claims = [
        ExtractedClaim(claim_text="Claim A."),
        ExtractedClaim(claim_text="Claim B."),
    ]

    results = verifier.verify_claims(claims, [])
    assert len(results) == 2
    for res in results:
        assert res.verdict == "UNVERIFIABLE"
        assert res.evidence.attributed_pmid is None
        assert res.evidence.all_scores == []
    mock_inference.assert_not_called()


def test_verify_claims_single_claim_evaluated_against_all_context_passages() -> None:
    """Single claim evaluates against all context passages with correct premise/hypothesis."""
    recorded_pairs: list[tuple[str, str]] = []

    def mock_nli(pairs, tok, model, batch_size):
        recorded_pairs.extend(pairs)
        # NLI layer returns empty passage_pmid per frozen architecture contract
        return [
            NLIScore(
                passage_pmid="",
                entailment_prob=0.9,
                neutral_prob=0.05,
                contradiction_prob=0.05,
            ),
            NLIScore(
                passage_pmid="",
                entailment_prob=0.2,
                neutral_prob=0.7,
                contradiction_prob=0.1,
            ),
        ]

    verifier = ClaimVerifier(nli_inference_fn=mock_nli)
    claims = [ExtractedClaim(claim_text="Aspirin treats fever.")]
    context = [
        _make_doc("101", "Aspirin is an effective antipyretic."),
        _make_doc("102", "Penicillin is an antibiotic."),
    ]

    results = verifier.verify_claims(claims, context)
    assert len(results) == 1
    assert results[0].claim_text == "Aspirin treats fever."
    assert results[0].verdict == "SUPPORTED"
    # Demonstrates that ClaimVerifier attached context PMIDs to the empty NLI scores
    assert results[0].evidence.attributed_pmid == "101"
    assert len(results[0].evidence.all_scores) == 2
    assert [s.passage_pmid for s in results[0].evidence.all_scores] == ["101", "102"]

    # PREMISE must be passage abstract, HYPOTHESIS must be claim text
    assert recorded_pairs == [
        ("Aspirin is an effective antipyretic.", "Aspirin treats fever."),
        ("Penicillin is an antibiotic.", "Aspirin treats fever."),
    ]


def test_verify_claims_multiple_claims_and_multiple_passages() -> None:
    """Every claim is evaluated against every context passage."""
    claims = [
        ExtractedClaim(claim_text="Claim 1."),
        ExtractedClaim(claim_text="Claim 2."),
    ]
    context = [
        _make_doc("201", "Abstract 201"),
        _make_doc("202", "Abstract 202"),
        _make_doc("203", "Abstract 203"),
    ]

    def mock_nli(pairs, tok, model, batch_size):
        # 2 claims * 3 docs = 6 pairs
        assert len(pairs) == 6
        # NLI layer returns empty passage_pmid per frozen architecture contract
        # Claim 1 scores: doc 201=neutral, doc 202=contra (0.85), doc 203=entail (0.6)
        # -> Claim 1 should be CONTRADICTED (priority) with PMID 202
        # Claim 2 scores: doc 201=entail (0.95), doc 202=neutral, doc 203=neutral
        # -> Claim 2 should be SUPPORTED with PMID 201
        return [
            # Claim 1
            NLIScore(
                passage_pmid="",
                entailment_prob=0.1,
                neutral_prob=0.8,
                contradiction_prob=0.1,
            ),
            NLIScore(
                passage_pmid="",
                entailment_prob=0.05,
                neutral_prob=0.1,
                contradiction_prob=0.85,
            ),
            NLIScore(
                passage_pmid="",
                entailment_prob=0.6,
                neutral_prob=0.3,
                contradiction_prob=0.1,
            ),
            # Claim 2
            NLIScore(
                passage_pmid="",
                entailment_prob=0.95,
                neutral_prob=0.03,
                contradiction_prob=0.02,
            ),
            NLIScore(
                passage_pmid="",
                entailment_prob=0.1,
                neutral_prob=0.8,
                contradiction_prob=0.1,
            ),
            NLIScore(
                passage_pmid="",
                entailment_prob=0.1,
                neutral_prob=0.8,
                contradiction_prob=0.1,
            ),
        ]

    verifier = ClaimVerifier(nli_inference_fn=mock_nli)
    results = verifier.verify_claims(claims, context)

    assert len(results) == 2
    assert results[0].claim_text == "Claim 1."
    assert results[0].verdict == "CONTRADICTED"
    assert results[0].evidence.attributed_pmid == "202"
    assert len(results[0].evidence.all_scores) == 3
    # Demonstrates ClaimVerifier attached context PMIDs to the empty NLI scores
    assert [s.passage_pmid for s in results[0].evidence.all_scores] == ["201", "202", "203"]

    assert results[1].claim_text == "Claim 2."
    assert results[1].verdict == "SUPPORTED"
    assert results[1].evidence.attributed_pmid == "201"
    assert len(results[1].evidence.all_scores) == 3
    assert [s.passage_pmid for s in results[1].evidence.all_scores] == ["201", "202", "203"]


def test_verify_claims_invalid_input_raises_verification_input_error() -> None:
    """Non-list or incorrectly typed items raise VerificationInputError."""
    verifier = ClaimVerifier()
    doc = _make_doc("1", "Text")

    with pytest.raises(VerificationInputError, match="Expected claims to be a list"):
        verifier.verify_claims("not a list", [doc])  # type: ignore[arg-type]

    with pytest.raises(VerificationInputError, match="Claim at index 0 is not an ExtractedClaim"):
        verifier.verify_claims(["not a claim object"], [doc])  # type: ignore[list-item]

    with pytest.raises(VerificationInputError, match="Expected context to be a list"):
        verifier.verify_claims([ExtractedClaim(claim_text="Claim")], "not a list")  # type: ignore[arg-type]

    with pytest.raises(
        VerificationInputError,
        match="Context item at index 0 is not a HybridScoredDocument",
    ):
        verifier.verify_claims(
            [ExtractedClaim(claim_text="Claim")],
            ["not a doc"],  # type: ignore[list-item]
        )


def test_verify_claims_nli_inference_error_propagates() -> None:
    """NLI failure propagates as NLIInferenceError and is not converted to UNVERIFIABLE."""

    def crashing_nli(*args, **kwargs):
        raise NLIInferenceError("CUDA failure during forward pass.")

    verifier = ClaimVerifier(nli_inference_fn=crashing_nli)
    claims = [ExtractedClaim(claim_text="Aspirin works.")]
    context = [_make_doc("1", "Abstract.")]

    with pytest.raises(NLIInferenceError, match="CUDA failure"):
        verifier.verify_claims(claims, context)

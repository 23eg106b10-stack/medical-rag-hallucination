"""Unit tests for M5 evidence aggregation logic."""

from __future__ import annotations

from schemas.verification_result import NLIScore
from verification.evidence_aggregation import aggregate, aggregate_evidence


def test_aggregate_supported_single_passage() -> None:
    """High entailment exceeding threshold produces SUPPORTED with winning PMID."""
    scores = [
        NLIScore(
            passage_pmid="1001",
            entailment_prob=0.88,
            neutral_prob=0.08,
            contradiction_prob=0.04,
        )
    ]
    verdict, attr = aggregate_evidence("Claim A", scores, 0.5, 0.5)
    assert verdict == "SUPPORTED"
    assert attr.attributed_pmid == "1001"
    assert len(attr.all_scores) == 1


def test_aggregate_supported_multiple_passages_winning_pmid() -> None:
    """SUPPORTED attributes the passage with the highest entailment probability."""
    scores = [
        NLIScore(
            passage_pmid="1001",
            entailment_prob=0.60,
            neutral_prob=0.30,
            contradiction_prob=0.10,
        ),
        NLIScore(
            passage_pmid="1002",
            entailment_prob=0.92,
            neutral_prob=0.05,
            contradiction_prob=0.03,
        ),
        NLIScore(
            passage_pmid="1003",
            entailment_prob=0.40,
            neutral_prob=0.50,
            contradiction_prob=0.10,
        ),
    ]
    verdict, attr = aggregate("Claim A", scores, 0.5, 0.5)
    assert verdict == "SUPPORTED"
    assert attr.attributed_pmid == "1002"
    assert len(attr.all_scores) == 3


def test_aggregate_contradicted_single_passage() -> None:
    """High contradiction exceeding threshold produces CONTRADICTED with winning PMID."""
    scores = [
        NLIScore(
            passage_pmid="2001",
            entailment_prob=0.05,
            neutral_prob=0.15,
            contradiction_prob=0.80,
        )
    ]
    verdict, attr = aggregate_evidence("Claim B", scores, 0.5, 0.5)
    assert verdict == "CONTRADICTED"
    assert attr.attributed_pmid == "2001"
    assert len(attr.all_scores) == 1


def test_aggregate_contradicted_multiple_passages_winning_pmid() -> None:
    """CONTRADICTED attributes the passage with the highest contradiction probability."""
    scores = [
        NLIScore(
            passage_pmid="2001",
            entailment_prob=0.10,
            neutral_prob=0.30,
            contradiction_prob=0.60,
        ),
        NLIScore(
            passage_pmid="2002",
            entailment_prob=0.05,
            neutral_prob=0.10,
            contradiction_prob=0.85,
        ),
    ]
    verdict, attr = aggregate("Claim B", scores, 0.5, 0.5)
    assert verdict == "CONTRADICTED"
    assert attr.attributed_pmid == "2002"
    assert len(attr.all_scores) == 2


def test_aggregate_unverifiable_neither_threshold_crossed() -> None:
    """Neither threshold crossed produces UNVERIFIABLE with attributed_pmid=None."""
    scores = [
        NLIScore(
            passage_pmid="3001",
            entailment_prob=0.35,
            neutral_prob=0.45,
            contradiction_prob=0.20,
        ),
        NLIScore(
            passage_pmid="3002",
            entailment_prob=0.20,
            neutral_prob=0.60,
            contradiction_prob=0.20,
        ),
    ]
    verdict, attr = aggregate_evidence("Claim C", scores, 0.5, 0.5)
    assert verdict == "UNVERIFIABLE"
    assert attr.attributed_pmid is None
    assert len(attr.all_scores) == 2
    assert [s.passage_pmid for s in attr.all_scores] == ["3001", "3002"]


def test_aggregate_contradiction_priority_both_thresholds_crossed() -> None:
    """When both entailment and contradiction thresholds are crossed, CONTRADICTED wins."""
    scores = [
        NLIScore(
            passage_pmid="4001",
            entailment_prob=0.90,
            neutral_prob=0.05,
            contradiction_prob=0.05,
        ),
        NLIScore(
            passage_pmid="4002",
            entailment_prob=0.10,
            neutral_prob=0.10,
            contradiction_prob=0.80,
        ),
    ]
    verdict, attr = aggregate_evidence("Claim D", scores, 0.5, 0.5)
    assert verdict == "CONTRADICTED"
    assert attr.attributed_pmid == "4002"
    assert len(attr.all_scores) == 2


def test_aggregate_contradiction_priority_same_passage_crossing_both() -> None:
    """With lowered thresholds, if a single passage crosses both, contradiction wins."""
    scores = [
        NLIScore(
            passage_pmid="4003",
            entailment_prob=0.45,
            neutral_prob=0.10,
            contradiction_prob=0.45,
        )
    ]
    verdict, attr = aggregate_evidence("Claim E", scores, 0.4, 0.4)
    assert verdict == "CONTRADICTED"
    assert attr.attributed_pmid == "4003"


def test_aggregate_empty_scores() -> None:
    """Empty scores list returns UNVERIFIABLE with None attributed_pmid and empty all_scores."""
    verdict, attr = aggregate_evidence("Claim F", [], 0.5, 0.5)
    assert verdict == "UNVERIFIABLE"
    assert attr.attributed_pmid is None
    assert attr.all_scores == []


def test_aggregate_all_scores_audit_trail_preserved() -> None:
    """All evaluated passage scores are preserved regardless of verdict or threshold."""
    scores = [
        NLIScore(
            passage_pmid="5001",
            entailment_prob=0.1,
            neutral_prob=0.8,
            contradiction_prob=0.1,
        ),
        NLIScore(
            passage_pmid="5002",
            entailment_prob=0.9,
            neutral_prob=0.05,
            contradiction_prob=0.05,
        ),
        NLIScore(
            passage_pmid="5003",
            entailment_prob=0.2,
            neutral_prob=0.7,
            contradiction_prob=0.1,
        ),
    ]
    verdict, attr = aggregate_evidence("Claim G", scores, 0.5, 0.5)
    assert verdict == "SUPPORTED"
    assert len(attr.all_scores) == 3
    assert [s.passage_pmid for s in attr.all_scores] == ["5001", "5002", "5003"]

"""Pure evidence aggregation for NLI verification (Milestone 5).

Evaluates candidate passage NLI scores against entailment and contradiction
thresholds to produce a categorical verdict and evidence attribution.

Enforces ADR-M5-002:
- Max-pooling across candidate passages.
- Contradiction priority (contradiction checked first; simultaneous crossing
  resolves to CONTRADICTED).
- Winning evidence attribution (highest contradiction PMID for CONTRADICTED,
  highest entailment PMID for SUPPORTED, None for UNVERIFIABLE).
- Full candidate audit trail preserved in `EvidenceAttribution.all_scores`.
"""

from __future__ import annotations

from typing import Literal

from schemas.verification_result import EvidenceAttribution, NLIScore


def aggregate_evidence(
    claim_text: str,
    scores: list[NLIScore],
    entailment_threshold: float = 0.5,
    contradiction_threshold: float = 0.5,
) -> tuple[Literal["SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"], EvidenceAttribution]:
    """Aggregate per-passage NLI scores into a verdict and evidence attribution.

    Args:
        claim_text: The extracted atomic claim text.
        scores: NLI scores for all candidate evidence passages evaluated against
            this claim.
        entailment_threshold: Minimum entailment probability for SUPPORTED.
        contradiction_threshold: Minimum contradiction probability for CONTRADICTED.

    Returns:
        A tuple of (verdict, EvidenceAttribution) where verdict is one of
        "SUPPORTED", "CONTRADICTED", or "UNVERIFIABLE".
    """
    if not scores:
        return "UNVERIFIABLE", EvidenceAttribution(attributed_pmid=None, all_scores=[])

    # Find highest contradiction and entailment scores
    # max() with key preserves the first encounter on ties
    best_contra = max(scores, key=lambda s: s.contradiction_prob)
    best_entail = max(scores, key=lambda s: s.entailment_prob)

    # 1. Contradiction checked first per ADR-M5-002 (contradiction priority)
    if best_contra.contradiction_prob >= contradiction_threshold:
        verdict: Literal["SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"] = "CONTRADICTED"
        attributed_pmid = best_contra.passage_pmid
    elif best_entail.entailment_prob >= entailment_threshold:
        verdict = "SUPPORTED"
        attributed_pmid = best_entail.passage_pmid
    else:
        verdict = "UNVERIFIABLE"
        attributed_pmid = None

    attribution = EvidenceAttribution(
        attributed_pmid=attributed_pmid,
        all_scores=list(scores),
    )
    return verdict, attribution


# Alias for concise import per architecture specification
aggregate = aggregate_evidence

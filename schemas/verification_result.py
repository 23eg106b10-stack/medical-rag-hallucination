"""Schemas for hallucination verification results (Milestone 5)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class NLIScore(BaseModel):
    passage_pmid: str
    entailment_prob: float
    neutral_prob: float
    contradiction_prob: float


class EvidenceAttribution(BaseModel):
    attributed_pmid: str | None
    all_scores: list[NLIScore]


class ClaimVerification(BaseModel):
    claim_text: str
    verdict: Literal["SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"]
    evidence: EvidenceAttribution

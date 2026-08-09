"""Schemas for hallucination verification (Milestone 4)."""

from pydantic import BaseModel


class ExtractedClaim(BaseModel):
    claim_text: str

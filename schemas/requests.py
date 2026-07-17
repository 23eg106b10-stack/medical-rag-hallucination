"""Inbound request schemas."""

from __future__ import annotations

from pydantic import BaseModel


class QuestionRequest(BaseModel):
    """A user-submitted question."""

    question: str

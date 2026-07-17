"""Outbound response schemas."""

from __future__ import annotations

from pydantic import BaseModel


class AnswerResponse(BaseModel):
    """An answer returned to the user."""

    answer: str

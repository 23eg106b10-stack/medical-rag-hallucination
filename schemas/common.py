"""Common domain schemas shared across requests and responses."""

from __future__ import annotations

from pydantic import BaseModel


class Document(BaseModel):
    """A single source document."""

    id: str
    title: str
    text: str


class RetrievedDocument(BaseModel):
    """A document paired with its retrieval score."""

    document: Document
    score: float

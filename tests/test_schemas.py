"""Tests for the shared schemas package."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from schemas.common import Document, RetrievedDocument
from schemas.requests import QuestionRequest
from schemas.responses import AnswerResponse


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

"""Schemas for LLM answer generation (Milestone 3.3).

Kept separate from ``schemas.responses`` because ``GeneratedAnswer`` is
the internal generation-layer output carrying the retrieval context it
was conditioned on — that context has no place on the outbound
``AnswerResponse`` shape returned to the user.
"""

from __future__ import annotations

from pydantic import BaseModel

from schemas.retrieval import HybridScoredDocument


class GeneratedAnswer(BaseModel):
    """A generated answer plus the context it was grounded in.

    ``context`` is preserved verbatim (the exact ``HybridScoredDocument``
    list handed to the generator) so downstream verification can audit
    which retrieved evidence the answer was conditioned on, per the M3.3
    grounding/traceability requirement.
    """

    question: str
    answer_text: str
    context: list[HybridScoredDocument]
    model_name: str

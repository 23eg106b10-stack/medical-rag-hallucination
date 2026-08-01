"""Prompt template constants for LLM generation (M3.3).

Per ADR-M3.3-002 (Prompt Template Ownership): this module contains ONLY
declarative template constants. No string concatenation, formatting
logic, context assembly, or truncation belongs here — that logic lives
exclusively in ``generation.context_builder``, per the frozen M3.3
architecture.

Field order (System, then Evidence, then Question, then Answer) per
review feedback citing "Architecture v1.0" — not independently verified
against a document shown to this implementation; see the M3.3
implementation report for that caveat. The wording itself remains a
functional placeholder, not tuned content.
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a medical question-answering assistant. Answer the question "
    "using only the evidence provided below. If the evidence does not "
    "support an answer, say so explicitly rather than guessing."
)

DOCUMENT_TEMPLATE = "[PMID: {pmid}] {title}\n{abstract}"

USER_TURN_TEMPLATE = "Evidence:\n{documents}\n\nQuestion: {question}\n\nAnswer:"

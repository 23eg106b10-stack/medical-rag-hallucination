"""Prompt template constants for claim extraction (M4).

Per ADR-M3.3-002 and ADR-M4-003 §3 (Prompt Template Ownership): this module
contains ONLY declarative template constants. No string concatenation,
formatting logic, context assembly, or parsing logic belongs here — that
orchestration lives exclusively in ``verification.claim_extraction``.
"""

from __future__ import annotations

CLAIM_EXTRACTION_SYSTEM_PROMPT = (
    "You are an expert medical text analyzer. Your task is to decompose the provided "
    "medical text into individual atomic factual claims.\n\n"
    "Rules:\n"
    "1. Output exactly one claim per line.\n"
    "2. Each claim must be an atomic, independently verifiable medical proposition.\n"
    "3. Decompose compound sentences into multiple separate single-claim lines.\n"
    "4. Resolve all anaphoric references, pronouns, and referring expressions (such as "
    "'these risk factors', 'this condition', 'they', or 'these treatments') to their "
    "explicit named medical entities so each claim is completely standalone.\n"
    "5. Output plain text only. Do NOT use JSON, markdown headers, or other formatting."
)

CLAIM_EXTRACTION_USER_TEMPLATE = (
    "Extract all atomic medical claims from the following text, one claim per line:\n\n"
    "{answer_text}\n\n"
    "Claims:"
)

# Aliases for brevity / consistency with generation/prompts.py conventions
SYSTEM_PROMPT = CLAIM_EXTRACTION_SYSTEM_PROMPT
USER_TURN_TEMPLATE = CLAIM_EXTRACTION_USER_TEMPLATE

"""Deterministic tokenizer for BM25 index construction (M3.1.2).

Intentionally simple, per the M3.1.2 architecture: lowercase, strip
punctuation, whitespace split, discard empty tokens. No stemming,
lemmatization, stopword removal, or third-party NLP libraries — those
change retrieval behavior and complicate reproducibility.
"""

from __future__ import annotations

import re

_PUNCTUATION_RE = re.compile(r"[^\w\s]", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase, punctuation-free whitespace tokens.

    Args:
        text: Raw text to tokenize.

    Returns:
        List of non-empty lowercase tokens, in original order.
    """
    lowered = text.lower()
    stripped = _PUNCTUATION_RE.sub(" ", lowered)
    return [token for token in stripped.split() if token]

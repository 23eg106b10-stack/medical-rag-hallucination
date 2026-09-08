"""Atomic claim extraction from generated answers (Milestone 4).

Implements the M4 claim extraction pipeline per ADR-M4-001, ADR-M4-002,
and ADR-M4-003:
- Verification unit = atomic medical claim (ADR-M4-001).
- Evidence attribution is verification-side (M5); ExtractedClaim contains
  only claim_text: str (ADR-M4-002).
- LLM-based decomposition using the already-loaded Llama-3.1-8B-Instruct
  generation infrastructure with do_sample=False (ADR-M4-003 §1, §2).
- Plain-text, one-claim-per-line contract — explicitly NOT JSON (ADR-M4-003 §4).
- Deterministic, string-level parsing, citation stripping, fragment filtering,
  process-commentary filtering, and exact-match deduplication (ADR-M4-003 §5).
- Atomicity and anaphora resolution are a generation-side contract, not a
  parser responsibility (ADR-M4-003 §6).
- Claim-count safety bound: MAX_CLAIMS = 35 (ADR-M4-003 §7, §8).
- Fail-loud error boundary on malformed output or claim count > 35 via
  ClaimExtractionError (ADR-M4-003 §7, §9).
"""

from __future__ import annotations

import logging
import re
import string
from collections.abc import Callable
from typing import Any

from generation.generator import make_transformers_generate_fn
from schemas.generation import GeneratedAnswer
from schemas.verification import ExtractedClaim
from verification.claim_extraction_prompts import (
    CLAIM_EXTRACTION_SYSTEM_PROMPT,
    CLAIM_EXTRACTION_USER_TEMPLATE,
)

logger = logging.getLogger(__name__)

# Named constant per ADR-M4-003 §8. Not in Settings.
MAX_CLAIMS: int = 35


class ClaimExtractionError(RuntimeError):
    """Raised on malformed extraction output or when claim count exceeds MAX_CLAIMS.

    Per ADR-M4-003 §9, this single exception type covers both:
    1. Malformed / empty model output where zero candidate lines exist after
       splitting and blank-line discard.
    2. Post-filtering and post-deduplication valid claim count exceeding
       MAX_CLAIMS (35), where silent truncation is explicitly forbidden.
    """


def _has_terminal_punctuation(text: str) -> bool:
    """Check rule 5(a): ends in '.', '!', '?', or closing quote/paren immediately after.

    Observable, string-level check. Discards truncated fragments lacking
    terminal punctuation (e.g. 'Therefore, it', 'pharmacogenomics biom').
    """
    stripped = text.strip()
    return bool(re.search(r"[.!?]['\"\)]*$", stripped))


def _is_balanced(text: str) -> bool:
    """Check rule 5(b): all '(' and '[' brackets are properly balanced and closed.

    Observable, string-level check. Discards incomplete fragments with unclosed
    parentheses or brackets (most commonly truncated citation markers like '(PMID:').
    """
    stack: list[str] = []
    matching = {")": "(", "]": "["}
    for ch in text:
        if ch in "([":
            stack.append(ch)
        elif ch in ")]":
            if not stack or stack[-1] != matching[ch]:
                return False
            stack.pop()
    return len(stack) == 0


def _strip_leading_markers(text: str) -> str:
    """Strip leading numbering (1., 1), (1), [1]) or bullet markers (-, *, •, +).

    Repeats until all leading structural prefixes are removed, preserving
    actual numbers embedded in prose (e.g. '1.5 mg').
    """
    line = text.strip()
    while True:
        # Match leading numbers with punctuation or bullet markers followed by whitespace or EOL
        new_line = re.sub(
            r"^(?:\d+[\.\)]|\(\d+\)|\[\d+\]|[-*•+>])(?:\s+|$)",
            "",
            line,
        ).strip()
        if new_line == line:
            break
        line = new_line
    return line


def _strip_citations(text: str) -> str:
    """Strip inline PMID and reference markers per ADR-M4-003 §5.

    Handles forms including:
    - [PMID: ...]
    - (PMID: ...)
    - (References: [PMID: ...])
    - (as mentioned in [PMID: ...])
    """
    # 1. References blocks
    line = re.sub(
        r"\((?:References|Reference):[^)]*\)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    line = re.sub(
        r"\[(?:References|Reference):[^\]]*\]",
        "",
        line,
        flags=re.IGNORECASE,
    )

    # 2. General parenthetical or bracketed blocks containing PMID references
    line = re.sub(
        r"\((?:as\s+(?:mentioned|noted|stated|described|reported)\s+in\s+)?[^)]*PMID:[^)]*\)",
        "",
        line,
        flags=re.IGNORECASE,
    )
    line = re.sub(
        r"\[(?:as\s+(?:mentioned|noted|stated|described|reported)\s+in\s+)?[^\]]*PMID:[^\]]*\]",
        "",
        line,
        flags=re.IGNORECASE,
    )

    # 3. Clean up empty parens/brackets if left behind
    line = re.sub(r"\(\s*\)", "", line)
    line = re.sub(r"\[\s*\]", "", line)

    # 4. Clean up whitespace before trailing punctuation (e.g. 'Hypertension .' -> 'Hypertension.')
    line = re.sub(r"\s+([.,!?;:])", r"\1", line)

    # 5. Collapse internal whitespace
    return " ".join(line.split()).strip()


_PROCESS_COMMENTARY_PATTERNS: list[re.Pattern[str]] = [
    # Offers to rephrase or conversational management
    re.compile(r"\b(?:please\s+)?let\s+me\s+know\s+if\b", re.IGNORECASE),
    re.compile(r"\b(?:please\s+)?feel\s+free\s+to\s+(?:ask|reach\s+out)\b", re.IGNORECASE),
    re.compile(r"\bhow\s+I\s+can\s+(?:assist|help)\s+you\b", re.IGNORECASE),
    re.compile(
        r"\bif\s+you\s+(?:have\s+any\s+further\s+questions|need\s+(?:more|any)\s+clarification)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bif\s+you\s+(?:would\s+like|want)\s+me\s+to\s+rephrase\b", re.IGNORECASE),
    re.compile(r"\bif\s+you\s+need\s+information\s+on\b.*\bi\s+recommend\b", re.IGNORECASE),
    # First-person conversational meta-statements
    re.compile(r"^i\s+(?:cannot|can't)\s+provide\s+a\s+definitive\s+answer\b", re.IGNORECASE),
    re.compile(r"^i\s+do\s+not\s+have\s+enough\s+information\b", re.IGNORECASE),
    re.compile(r"^i\s+apologize\b", re.IGNORECASE),
    re.compile(r"^as\s+an\s+ai\b", re.IGNORECASE),
    re.compile(r"^as\s+a\s+language\s+model\b", re.IGNORECASE),
    re.compile(
        r"^here\s+(?:is|are)\s+the\s+(?:extracted\s+claims|atomic\s+claims|claims|answer)\b",
        re.IGNORECASE,
    ),
    # Conversational disclaimers to user
    re.compile(
        r"^note:\s*(?:the\s+answer\s+is\s+based|these\s+causes\s+are\s+not\s+exhaustive|please\s+note)\b",
        re.IGNORECASE,
    ),
]


def _is_process_commentary(text: str) -> bool:
    """Check rule 5: discard only pure conversational/generation-process commentary.

    Narrowly matches statements about the model's own response, behavior, or
    the exchange itself. Does NOT match substantive medical uncertainty (such as
    'The evidence does not support...', 'Statin therapy may reduce...', 'X might be
    uncertain...', 'The exact mechanism is unclear...').
    """
    stripped = text.strip()
    return any(pattern.search(stripped) for pattern in _PROCESS_COMMENTARY_PATTERNS)


def _strip_rephrased_framing(text: str) -> str:
    """Strip self-referential framing prefixes, e.g. 'Revised rephrased answer: '.

    Per ADR-M4-003 §5 (referencing Calibration Sample 6), the framing prefix is
    process commentary, but any substantive evidentiary statement following it
    is retained.
    """
    return re.sub(
        r"^(?:(?:revised\s+)?rephrased\s+answer|revised\s+answer):\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()


def _normalize_for_dedup(text: str) -> str:
    """Exact string normalization for deduplication per ADR-M4-003 §5.

    Performs only:
    1. Lowercase
    2. Strip punctuation
    3. Collapse whitespace

    No stemming, lemmatization, semantic similarity, or fuzzy matching.
    """
    lowered = text.lower()
    no_punct = "".join(ch for ch in lowered if ch not in string.punctuation)
    return " ".join(no_punct.split())


def parse_claims(raw_output: str, max_claims: int = MAX_CLAIMS) -> list[ExtractedClaim]:
    """Parse and filter raw model extraction output into ExtractedClaim objects.

    Implements the complete deterministic parser contract (ADR-M4-003 §5, §7, §9):
    1. Split on newlines and discard blank lines.
    2. If zero candidate lines exist, raise ClaimExtractionError (malformed boundary).
    3. Strip leading numbering and bullet markers.
    4. Strip inline PMID/reference markers.
    5. Discard incomplete fragments (missing terminal punctuation or unbalanced brackets).
    6. Discard pure conversational/process commentary.
    7. Deduplicate via exact string normalization.
    8. Enforce claim-count gate (0 allowed, 1..max_claims allowed, >max_claims raises).

    Args:
        raw_output: Raw text output from extraction model generation.
        max_claims: Upper bound on valid unique claims (default: MAX_CLAIMS = 35).

    Returns:
        List of unique, valid ExtractedClaim instances.

    Raises:
        ClaimExtractionError: if raw output is empty/whitespace-only (malformed)
            or if the unique valid claim count exceeds max_claims.
    """
    if not raw_output or not raw_output.strip():
        raise ClaimExtractionError("Claim extraction output is empty or whitespace-only.")

    raw_lines = raw_output.splitlines()
    candidate_lines = [line.strip() for line in raw_lines if line.strip()]

    # ADR-M4-003 §9: Malformed-output failure boundary
    if not candidate_lines:
        raise ClaimExtractionError(
            "Claim extraction produced zero candidate lines after blank-line discard."
        )

    processed_lines: list[str] = []
    for line in candidate_lines:
        # Step 3: Strip leading bullet / numbering markers
        cleaned = _strip_leading_markers(line)
        if not cleaned:
            continue

        # Strip self-referential framing prefixes (e.g. 'Revised rephrased answer:')
        cleaned = _strip_rephrased_framing(cleaned)
        if not cleaned:
            continue

        # Step 4: Strip inline PMID / reference markers
        cleaned = _strip_citations(cleaned)
        if not cleaned:
            continue

        # Step 5: Filter incomplete fragments via observable string-level rules
        if not _has_terminal_punctuation(cleaned):
            logger.debug("Discarding line lacking terminal punctuation: %r", cleaned)
            continue

        if not _is_balanced(cleaned):
            logger.debug("Discarding line with unbalanced brackets/parentheses: %r", cleaned)
            continue

        # Step 6: Discard pure conversational / generation-process commentary
        if _is_process_commentary(cleaned):
            logger.debug("Discarding process commentary line: %r", cleaned)
            continue

        processed_lines.append(cleaned)

    # Step 7: Deduplicate using exact normalized string equality only
    seen_normalized: set[str] = set()
    unique_claims: list[str] = []
    for line in processed_lines:
        norm_key = _normalize_for_dedup(line)
        if norm_key not in seen_normalized:
            seen_normalized.add(norm_key)
            unique_claims.append(line)

    # Step 8: Claim-count gate (ADR-M4-003 §7, §8)
    if len(unique_claims) > max_claims:
        raise ClaimExtractionError(
            f"Extracted {len(unique_claims)} unique claims, which exceeds the maximum "
            f"permitted limit of {max_claims}. Refusing to silently truncate."
        )

    return [ExtractedClaim(claim_text=c) for c in unique_claims]


def build_claim_extraction_prompt(answer_text: str) -> str:
    """Assemble the claim extraction prompt from declarative template constants."""
    user_turn = CLAIM_EXTRACTION_USER_TEMPLATE.format(answer_text=answer_text.strip())
    return f"{CLAIM_EXTRACTION_SYSTEM_PROMPT}\n\n{user_turn}"


def make_extraction_generate_fn(
    max_new_tokens: int = 512,
    do_sample: bool = False,
) -> Callable[[str, Any, Any], str]:
    """Build a deterministic generation function for claim extraction.

    Reuses the existing M3.3 generation infrastructure (`make_transformers_generate_fn`),
    mandating `do_sample=False` (greedy decoding) per ADR-M4-003 §2.
    """
    if do_sample:
        raise ValueError(
            "ADR-M4-003 strictly requires deterministic greedy decoding (do_sample=False)."
        )
    return make_transformers_generate_fn(max_new_tokens=max_new_tokens, do_sample=False)


class ClaimExtractor:
    """Orchestrates LLM-based claim extraction and deterministic parsing (M4)."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        generate_fn: Callable[[str, Any, Any], str],
        prompt_builder: Callable[[str], str] | None = None,
        max_claims: int = MAX_CLAIMS,
    ) -> None:
        """
        Args:
            model: Loaded causal LM (reused from M3.3).
            tokenizer: Matching tokenizer.
            generate_fn: Callable `(prompt, tokenizer, model) -> str`
                performing deterministic extraction generation (do_sample=False).
            prompt_builder: Optional prompt assembly callable.
                Defaults to `build_claim_extraction_prompt`.
            max_claims: Upper bound on unique valid claims (default: MAX_CLAIMS = 35).
        """
        self._model = model
        self._tokenizer = tokenizer
        self._generate_fn = generate_fn
        self._prompt_builder = prompt_builder or build_claim_extraction_prompt
        self._max_claims = max_claims

    def extract_claims(self, answer: GeneratedAnswer | str) -> list[ExtractedClaim]:
        """Extract atomic medical claims from a GeneratedAnswer or raw answer string.

        Args:
            answer: A `GeneratedAnswer` instance or raw answer text string.

        Returns:
            List of unique, valid `ExtractedClaim` instances.

        Raises:
            ClaimExtractionError: if generation produces malformed/empty output
                or if the unique valid claim count exceeds `max_claims`.
        """
        answer_text = answer.answer_text if isinstance(answer, GeneratedAnswer) else answer
        prompt = self._prompt_builder(answer_text)
        raw_output = self._generate_fn(prompt, self._tokenizer, self._model)
        return parse_claims(raw_output, max_claims=self._max_claims)

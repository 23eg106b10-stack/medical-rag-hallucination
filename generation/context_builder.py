"""Prompt assembly for LLM generation (M3.3, revised per ACR-002).

Per ADR-M3.3-002 (Prompt Template Ownership): all prompt assembly logic
lives here; ``generation.prompts`` holds only declarative template
constants.

Per ACR-002 (Prompt Builder Contract Revision): the module-level
``build_prompt(question, context) -> str`` function has been replaced by
``PromptBuilder``, an injectable class that receives a tokenizer and a
context window budget at construction time. This resolves a genuine
contract contradiction found during M3.3 implementation review: the
frozen "raise on context overflow" failure boundary requires counting
tokens against a real tokenizer, but the original free-function
signature had no way to receive one. ``Generator``'s DI contract is
unaffected — a bound ``PromptBuilder(...).build_prompt`` method still
satisfies ``Callable[[str, list[HybridScoredDocument]], str]``.
"""

from __future__ import annotations

from typing import Any

from generation.prompts import DOCUMENT_TEMPLATE, SYSTEM_PROMPT, USER_TURN_TEMPLATE
from schemas.retrieval import HybridScoredDocument


class PromptOverflowError(RuntimeError):
    """Raised when the assembled prompt exceeds the configured context window.

    No silent truncation is performed — dropping retrieved evidence
    without visibly failing risks silently removing exactly the context
    the answer was supposed to be grounded in, per the M3.3 failure
    boundary.
    """


class PromptBuilder:
    """Assembles prompts and enforces a hard context-window token budget."""

    def __init__(self, tokenizer: Any, context_window: int) -> None:
        """
        Args:
            tokenizer: A loaded tokenizer exposing an ``encode`` method
                that returns a token-id sequence (the same tokenizer
                used by ``llm_loader.load_generation_model``).
            context_window: Maximum number of prompt tokens permitted.
                NOTE — known simplification: this budgets the prompt
                only; it does not reserve headroom for the model's own
                response tokens (``max_new_tokens``). If that reservation
                is needed, the caller must subtract it when constructing
                this value. Flagging this explicitly rather than
                assuming it silently.
        """
        self._tokenizer = tokenizer
        self._context_window = context_window

    def build_prompt(self, question: str, context: list[HybridScoredDocument]) -> str:
        """Assemble the full prompt from a question and retrieved context.

        Args:
            question: The user's question.
            context: Fused retrieval results (M3.2 output), typically the
                Top-5 ``HybridScoredDocument`` list from
                ``HybridRetriever``. Order is preserved exactly as given
                — this method never re-sorts by PMID, title, or any
                other key; it trusts the RRF ordering it was handed.

        Returns:
            The fully assembled prompt string.

        Raises:
            PromptOverflowError: if the assembled prompt exceeds
                ``context_window`` tokens. No truncation is performed.
        """
        documents_text = "\n\n".join(
            DOCUMENT_TEMPLATE.format(
                pmid=item.document.pmid,
                title=item.document.title,
                abstract=item.document.abstract,
            )
            for item in context
        )
        user_turn = USER_TURN_TEMPLATE.format(question=question, documents=documents_text)
        prompt = f"{SYSTEM_PROMPT}\n\n{user_turn}"

        token_count = len(self._tokenizer.encode(prompt))
        if token_count > self._context_window:
            raise PromptOverflowError(
                f"Prompt requires {token_count} tokens, exceeding the "
                f"configured context_window of {self._context_window}. "
                "Refusing to silently truncate retrieved evidence."
            )

        return prompt

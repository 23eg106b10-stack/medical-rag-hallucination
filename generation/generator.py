"""Generation orchestration (M3.3).

``Generator`` is the orchestration layer only, per the frozen M3.3
architecture: it holds no prompt-assembly logic (that belongs to
``generation.context_builder``) and no model-loading logic (that
belongs to ``generation.llm_loader``). All heavy dependencies are
injected, per the project's mandatory Dependency Injection pattern —
the same pattern used throughout ``retrieval/``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from schemas.generation import GeneratedAnswer
from schemas.retrieval import HybridScoredDocument

logger = logging.getLogger(__name__)


class EmptyGenerationError(RuntimeError):
    """Raised when the model produces empty or whitespace-only output.

    No partial or empty answer is ever returned as a valid
    ``GeneratedAnswer`` — silently propagating an empty answer risks it
    being mistaken downstream for "nothing to verify" rather than "the
    model failed," per the M3.3 fail-loud failure boundary.
    """


class Generator:
    """Orchestrates prompt assembly and LLM generation."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        context_builder: Callable[[str, list[HybridScoredDocument]], str],
        generate_fn: Callable[[str, Any, Any], str],
        model_name: str,
    ) -> None:
        """
        Args:
            model: A loaded causal LM (e.g. from
                ``llm_loader.load_generation_model``).
            tokenizer: The matching tokenizer.
            context_builder: Callable assembling a prompt string from
                ``(question, context)`` — e.g.
                ``generation.context_builder.build_prompt``.
            generate_fn: Callable ``(prompt, tokenizer, model) -> str``
                performing the actual decoding call. Injected so unit
                tests never load a real model.
            model_name: Recorded on the returned ``GeneratedAnswer`` for
                traceability.
        """
        self._model = model
        self._tokenizer = tokenizer
        self._context_builder = context_builder
        self._generate_fn = generate_fn
        self._model_name = model_name

    def generate(self, question: str, context: list[HybridScoredDocument]) -> GeneratedAnswer:
        """Generate an answer grounded in the given retrieval context.

        Args:
            question: The user's question.
            context: Fused retrieval results from M3.2 (typically Top-5).

        Returns:
            A ``GeneratedAnswer`` carrying the raw generated text and the
            context it was conditioned on.

        Raises:
            EmptyGenerationError: if the model produces empty or
                whitespace-only output.
        """
        prompt = self._context_builder(question, context)
        raw_text = self._generate_fn(prompt, self._tokenizer, self._model)

        if not raw_text or not raw_text.strip():
            raise EmptyGenerationError(
                f"Generation produced empty output for question: {question!r}"
            )

        return GeneratedAnswer(
            question=question,
            answer_text=raw_text.strip(),
            context=context,
            model_name=self._model_name,
        )


def make_transformers_generate_fn(
    max_new_tokens: int, do_sample: bool
) -> Callable[[str, Any, Any], str]:
    """Build a concrete, production ``generate_fn`` backed by Transformers.

    Per ADR-M3.3-001 (Transformers + bitsandbytes is the mandatory
    inference backend). Kept in ``generator.py`` rather than
    ``llm_loader.py`` — loading a model and generating from it are
    distinct responsibilities, per M3.3 implementation review.
    Decoding parameters are bound via closure rather than threaded
    through the 3-arg ``generate_fn`` contract itself, so
    ``Generator``'s constructor signature and test doubles are
    unaffected.

    NOTE: ``do_sample=False`` (greedy decoding) is a provisional default
    consistent with the project's stated reproducibility principle, not
    yet the subject of a frozen ADR on decoding determinism. See the
    M3.3 implementation report.

    Args:
        max_new_tokens: Maximum number of new tokens to generate.
        do_sample: Whether to sample (``True``) or decode greedily
            (``False``).

    Returns:
        A callable matching ``Generator``'s ``generate_fn`` contract:
        ``(prompt, tokenizer, model) -> str``.
    """

    def _generate_fn(prompt: str, tokenizer: Any, model: Any) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
        generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        return tokenizer.decode(generated_ids, skip_special_tokens=True)

    return _generate_fn

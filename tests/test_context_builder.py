"""Tests for generation.context_builder.PromptBuilder (ACR-002)."""

from __future__ import annotations

import pytest

from generation.context_builder import PromptBuilder, PromptOverflowError
from schemas.corpus import CorpusDocument
from schemas.retrieval import HybridScoredDocument


class _FakeTokenizer:
    """Minimal stand-in: one 'token' per whitespace-separated word."""

    def encode(self, text: str) -> list[str]:
        return text.split()


def _make_hit(
    pmid: str, title: str, abstract: str, rrf_score: float = 0.03
) -> HybridScoredDocument:
    return HybridScoredDocument(
        document=CorpusDocument(pmid=pmid, title=title, abstract=abstract),
        sparse_score=1.0,
        dense_score=1.0,
        rrf_score=rrf_score,
    )


def test_build_prompt_includes_question():
    builder = PromptBuilder(tokenizer=_FakeTokenizer(), context_window=10_000)
    prompt = builder.build_prompt("What causes fever?", [])
    assert "What causes fever?" in prompt


def test_build_prompt_includes_every_document():
    builder = PromptBuilder(tokenizer=_FakeTokenizer(), context_window=10_000)
    context = [
        _make_hit("111", "Fever mechanisms", "Fever is caused by pyrogens."),
        _make_hit("222", "Inflammation basics", "Inflammation involves cytokines."),
    ]
    prompt = builder.build_prompt("What causes fever?", context)

    assert "111" in prompt
    assert "Fever mechanisms" in prompt
    assert "Fever is caused by pyrogens." in prompt
    assert "222" in prompt
    assert "Inflammation basics" in prompt
    assert "Inflammation involves cytokines." in prompt


def test_build_prompt_preserves_context_order():
    builder = PromptBuilder(tokenizer=_FakeTokenizer(), context_window=10_000)
    context = [
        _make_hit("111", "First doc", "First abstract."),
        _make_hit("222", "Second doc", "Second abstract."),
    ]
    prompt = builder.build_prompt("q", context)

    assert prompt.index("First doc") < prompt.index("Second doc")


def test_context_builder_preserves_rrf_order():
    """PromptBuilder must never re-sort by PMID, title, or any other key —
    it trusts the RRF ordering HybridRetriever already produced. This
    guards against someone later 'helpfully' sorting alphabetically or
    by PMID.
    """
    builder = PromptBuilder(tokenizer=_FakeTokenizer(), context_window=10_000)
    # Deliberately RRF-descending but PMID- and title-ascending, so any
    # accidental re-sort by PMID or title would silently pass unless we
    # assert against RRF order specifically.
    context = [
        _make_hit("999", "Zebra doc", "Highest-ranked evidence.", rrf_score=0.05),
        _make_hit("500", "Mango doc", "Second-ranked evidence.", rrf_score=0.03),
        _make_hit("100", "Apple doc", "Lowest-ranked evidence.", rrf_score=0.01),
    ]
    prompt = builder.build_prompt("q", context)

    assert prompt.index("Zebra doc") < prompt.index("Mango doc") < prompt.index("Apple doc")


def test_build_prompt_handles_empty_context():
    builder = PromptBuilder(tokenizer=_FakeTokenizer(), context_window=10_000)
    prompt = builder.build_prompt("q", [])
    assert "q" in prompt


def test_build_prompt_raises_on_overflow():
    builder = PromptBuilder(tokenizer=_FakeTokenizer(), context_window=5)
    context = [_make_hit("111", "T", "A rather long abstract with many words in it.")]

    with pytest.raises(PromptOverflowError, match="exceeding the configured context_window"):
        builder.build_prompt("A question with several words too", context)


def test_build_prompt_does_not_raise_under_budget():
    builder = PromptBuilder(tokenizer=_FakeTokenizer(), context_window=10_000)
    # Should not raise.
    builder.build_prompt("q", [_make_hit("111", "T", "A")])


def test_build_prompt_does_not_truncate_on_overflow():
    """Overflow must raise, never silently drop content."""
    builder = PromptBuilder(tokenizer=_FakeTokenizer(), context_window=1)
    with pytest.raises(PromptOverflowError):
        builder.build_prompt("q", [_make_hit("111", "T", "A")])

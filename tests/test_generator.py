"""Tests for generation.generator.Generator and make_transformers_generate_fn."""

from __future__ import annotations

import pytest

from generation.context_builder import PromptBuilder
from generation.generator import (
    EmptyGenerationError,
    Generator,
    make_transformers_generate_fn,
)
from schemas.corpus import CorpusDocument
from schemas.retrieval import HybridScoredDocument


def _make_context() -> list[HybridScoredDocument]:
    return [
        HybridScoredDocument(
            document=CorpusDocument(pmid="111", title="T", abstract="A"),
            sparse_score=1.0,
            dense_score=1.0,
            rrf_score=0.03,
        )
    ]


def test_generate_calls_context_builder_with_question_and_context():
    calls = []

    def fake_context_builder(question, context):
        calls.append((question, context))
        return "PROMPT"

    generator = Generator(
        model=object(),
        tokenizer=object(),
        context_builder=fake_context_builder,
        generate_fn=lambda prompt, tok, model: "an answer",
        model_name="fake-model",
    )

    context = _make_context()
    generator.generate("What is X?", context)

    assert calls == [("What is X?", context)]


def test_generate_returns_populated_generated_answer():
    generator = Generator(
        model=object(),
        tokenizer=object(),
        context_builder=lambda q, c: "PROMPT",
        generate_fn=lambda prompt, tok, model: "  the answer  ",
        model_name="fake-model",
    )

    context = _make_context()
    result = generator.generate("What is X?", context)

    assert result.question == "What is X?"
    assert result.answer_text == "the answer"  # stripped
    assert result.context == context
    assert result.model_name == "fake-model"


def test_generate_raises_on_empty_output():
    generator = Generator(
        model=object(),
        tokenizer=object(),
        context_builder=lambda q, c: "PROMPT",
        generate_fn=lambda prompt, tok, model: "",
        model_name="fake-model",
    )

    with pytest.raises(EmptyGenerationError):
        generator.generate("What is X?", _make_context())


def test_generate_raises_on_whitespace_only_output():
    generator = Generator(
        model=object(),
        tokenizer=object(),
        context_builder=lambda q, c: "PROMPT",
        generate_fn=lambda prompt, tok, model: "   \n\t  ",
        model_name="fake-model",
    )

    with pytest.raises(EmptyGenerationError):
        generator.generate("What is X?", _make_context())


def test_generate_passes_prompt_from_context_builder_to_generate_fn():
    received_prompts = []

    def fake_generate_fn(prompt, tokenizer, model):
        received_prompts.append(prompt)
        return "ok"

    generator = Generator(
        model=object(),
        tokenizer=object(),
        context_builder=lambda q, c: "A SPECIFIC PROMPT",
        generate_fn=fake_generate_fn,
        model_name="fake-model",
    )

    generator.generate("q", _make_context())
    assert received_prompts == ["A SPECIFIC PROMPT"]


class _FakeTokenizerForBuilder:
    def encode(self, text):
        return text.split()


def test_generator_accepts_prompt_builder_unmodified():
    """Proves ACR-002 resolved cleanly: PromptBuilder.build_prompt (a
    bound method) satisfies Generator's context_builder contract with
    zero changes to Generator itself.
    """
    builder = PromptBuilder(tokenizer=_FakeTokenizerForBuilder(), context_window=10_000)

    generator = Generator(
        model=object(),
        tokenizer=object(),
        context_builder=builder.build_prompt,
        generate_fn=lambda prompt, tok, model: "an answer",
        model_name="fake-model",
    )

    result = generator.generate("What is X?", _make_context())
    assert result.answer_text == "an answer"


class _FakeShapeArray(list):
    """Minimal stand-in for a tensor's .shape attribute access pattern."""

    @property
    def shape(self):
        return (len(self), len(self[0]) if self else 0)


class _FakeInputs(dict):
    def to(self, device):
        return self


class _FakeHFTokenizer:
    def __call__(self, prompt, return_tensors=None):
        return _FakeInputs(input_ids=_FakeShapeArray([[1, 2, 3]]))

    def decode(self, ids, skip_special_tokens=True):
        return "decoded output"


class _FakeHFModel:
    device = "cpu"

    def generate(self, **kwargs):
        # Simulate 3 prompt tokens + 2 newly generated tokens.
        return [[1, 2, 3, 4, 5]]


def test_transformers_generate_fn_calls_model_and_decodes():
    generate_fn = make_transformers_generate_fn(max_new_tokens=10, do_sample=False)
    result = generate_fn("some prompt", _FakeHFTokenizer(), _FakeHFModel())
    assert result == "decoded output"


def test_transformers_generate_fn_slices_off_prompt_tokens():
    captured = {}

    class _CapturingModel(_FakeHFModel):
        def generate(self, **kwargs):
            captured["max_new_tokens"] = kwargs["max_new_tokens"]
            captured["do_sample"] = kwargs["do_sample"]
            return [[1, 2, 3, 4, 5]]

    class _CapturingTokenizer(_FakeHFTokenizer):
        def decode(self, ids, skip_special_tokens=True):
            captured["decoded_ids"] = list(ids)
            return "ok"

    generate_fn = make_transformers_generate_fn(max_new_tokens=99, do_sample=True)
    generate_fn("prompt", _CapturingTokenizer(), _CapturingModel())

    assert captured["max_new_tokens"] == 99
    assert captured["do_sample"] is True
    # 3 prompt tokens sliced off the front of [1,2,3,4,5] -> [4,5]
    assert captured["decoded_ids"] == [4, 5]

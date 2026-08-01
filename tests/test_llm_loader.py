"""Tests for generation.llm_loader.load_generation_model.

Per the M3.3 testing strategy, this is deliberately thin: real model
loading is a manual smoke test, not a CI-gated unit test. These tests
mock the transformers calls to verify load_generation_model's own
logic (parameter wiring, error handling) without downloading any real
weights.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from generation.llm_loader import load_generation_model


def test_load_generation_model_returns_tokenizer_and_model():
    fake_tokenizer = MagicMock()
    fake_model = MagicMock()

    with (
        patch(
            "generation.llm_loader.AutoTokenizer.from_pretrained",
            return_value=fake_tokenizer,
        ) as mock_tok,
        patch(
            "generation.llm_loader.AutoModelForCausalLM.from_pretrained",
            return_value=fake_model,
        ) as mock_model,
    ):
        tokenizer, model = load_generation_model(
            model_name="fake/model",
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
            device_map="auto",
            hf_token="",
        )

    assert tokenizer is fake_tokenizer
    assert model is fake_model
    mock_tok.assert_called_once()
    mock_model.assert_called_once()


def test_load_generation_model_passes_none_token_when_empty_string():
    with (
        patch(
            "generation.llm_loader.AutoTokenizer.from_pretrained",
            return_value=MagicMock(),
        ) as mock_tok,
        patch(
            "generation.llm_loader.AutoModelForCausalLM.from_pretrained",
            return_value=MagicMock(),
        ),
    ):
        load_generation_model(
            model_name="fake/model",
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
            device_map="auto",
            hf_token="",
        )

    _, kwargs = mock_tok.call_args
    assert kwargs["token"] is None


def test_load_generation_model_raises_runtime_error_on_load_failure():
    with (
        patch(
            "generation.llm_loader.AutoTokenizer.from_pretrained",
            side_effect=OSError("network unreachable"),
        ),
        pytest.raises(RuntimeError, match="Failed to load generation model"),
    ):
        load_generation_model(
            model_name="fake/model",
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
            device_map="auto",
        )


def test_load_generation_model_raises_runtime_error_on_invalid_dtype():
    with pytest.raises(RuntimeError, match="Invalid bnb_4bit_compute_dtype"):
        load_generation_model(
            model_name="fake/model",
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="not_a_real_dtype",
            device_map="auto",
        )

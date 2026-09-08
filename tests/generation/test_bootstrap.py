"""Tests for the M3.3 generation bootstrap."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from config.settings import Settings
from generation.bootstrap import build_generator
from generation.generator import Generator


@pytest.fixture
def mock_settings() -> Settings:
    return Settings(
        llm_model_name="test-model",
        llm_load_in_4bit=True,
        llm_bnb_4bit_quant_type="nf4",
        llm_bnb_4bit_compute_dtype="float16",
        llm_device_map="auto",
        hf_token="test-token",
        llm_quantized_layers_gpu_resident=False,
        llm_max_new_tokens=100,
        llm_do_sample=False,
        llm_context_window=1024,
    )


@patch("generation.bootstrap.load_generation_model")
@patch("generation.bootstrap.make_transformers_generate_fn")
@patch("generation.bootstrap.PromptBuilder")
def test_build_generator(
    mock_prompt_builder_cls: Mock,
    mock_make_generate_fn: Mock,
    mock_load_model: Mock,
    mock_settings: Settings,
) -> None:
    mock_tokenizer = Mock()
    mock_model = Mock()
    mock_load_model.return_value = (mock_tokenizer, mock_model)

    mock_prompt_builder = Mock()
    mock_prompt_builder_cls.return_value = mock_prompt_builder

    mock_generate_fn = Mock()
    mock_make_generate_fn.return_value = mock_generate_fn

    generator = build_generator(mock_settings)

    assert isinstance(generator, Generator)

    mock_load_model.assert_called_once_with(
        model_name="test-model",
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        device_map="auto",
        hf_token="test-token",
        quantized_layers_gpu_resident=False,
    )

    mock_prompt_builder_cls.assert_called_once_with(
        tokenizer=mock_tokenizer,
        context_window=1024,
    )

    mock_make_generate_fn.assert_called_once_with(
        max_new_tokens=100,
        do_sample=False,
    )

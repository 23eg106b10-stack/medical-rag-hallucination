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


def test_load_generation_model_default_preserves_device_map_passthrough():
    """ACR-004: when quantized_layers_gpu_resident is not set (default
    False), device_map is passed through unchanged and the
    BitsAndBytesConfig CPU/disk-offload validation flag is NOT enabled —
    no regression to existing "auto" (or any other) placement behavior.
    """
    with (
        patch(
            "generation.llm_loader.AutoTokenizer.from_pretrained",
            return_value=MagicMock(),
        ),
        patch(
            "generation.llm_loader.AutoModelForCausalLM.from_pretrained",
            return_value=MagicMock(),
        ) as mock_model,
    ):
        load_generation_model(
            model_name="fake/model",
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
            device_map="auto",
            hf_token="",
        )

    _, kwargs = mock_model.call_args
    assert kwargs["device_map"] == "auto"
    assert kwargs["quantization_config"].llm_int8_enable_fp32_cpu_offload is False


def test_load_generation_model_gpu_resident_flag_passes_explicit_device_map():
    """ACR-004: quantized_layers_gpu_resident=True must pass the exact
    validated explicit device map to from_pretrained AND set
    llm_int8_enable_fp32_cpu_offload=True on BitsAndBytesConfig — these
    two behaviors are coupled and must never diverge. Without the flag,
    Transformers' quantizer validation rejects any device_map containing
    CPU/disk entries.
    """
    with (
        patch(
            "generation.llm_loader.AutoTokenizer.from_pretrained",
            return_value=MagicMock(),
        ),
        patch(
            "generation.llm_loader.AutoModelForCausalLM.from_pretrained",
            return_value=MagicMock(),
        ) as mock_model,
    ):
        load_generation_model(
            model_name="fake/model",
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
            device_map="auto",
            hf_token="",
            quantized_layers_gpu_resident=True,
        )

    _, kwargs = mock_model.call_args
    assert kwargs["device_map"] == {
        "model.embed_tokens": "cpu",
        "model.layers": "cuda:0",
        "model.norm": "cuda:0",
        "model.rotary_emb": "cuda:0",
        "lm_head": "cpu",
    }
    assert kwargs["quantization_config"].llm_int8_enable_fp32_cpu_offload is True


@pytest.mark.parametrize("gpu_resident", [True, False])
def test_load_generation_model_device_map_and_offload_flag_never_diverge(gpu_resident):
    """ACR-004: the explicit device map and the CPU/disk-offload
    validation flag are ONE coupled configuration. This test fails if a
    future change makes them independently settable — the map is in use
    if and only if the offload flag is True, for both flag values.
    """
    with (
        patch(
            "generation.llm_loader.AutoTokenizer.from_pretrained",
            return_value=MagicMock(),
        ),
        patch(
            "generation.llm_loader.AutoModelForCausalLM.from_pretrained",
            return_value=MagicMock(),
        ) as mock_model,
    ):
        load_generation_model(
            model_name="fake/model",
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
            device_map="auto",
            hf_token="",
            quantized_layers_gpu_resident=gpu_resident,
        )

    _, kwargs = mock_model.call_args
    explicit_map_used = kwargs["device_map"] != "auto"
    offload_flag_set = kwargs["quantization_config"].llm_int8_enable_fp32_cpu_offload

    assert explicit_map_used == offload_flag_set == gpu_resident

"""Unit tests for index.embedding_generator (M3.1.3)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from index.embedding_generator import embed_batch


def _fake_tokenizer(max_tokens: int = 8):
    """Fake tokenizer that truncates deterministically and pads to batch max,
    mirroring the real call signature: tokenizer(texts, truncation=True,
    padding=True, max_length=..., return_tensors="pt").
    """

    def tokenizer(texts, truncation=True, padding=True, max_length=None, return_tensors="pt"):
        limit = max_tokens if max_length is None else min(max_tokens, max_length)
        all_ids = []
        for text in texts:
            tokens = text.split()[:limit]
            ids = [hash(tok) % 1000 for tok in tokens] or [0]
            all_ids.append(ids)

        pad_len = max(len(ids) for ids in all_ids)
        padded = [ids + [0] * (pad_len - len(ids)) for ids in all_ids]
        return {"input_ids": torch.tensor(padded)}

    return tokenizer


def _fake_model(dim: int = 4):
    """Fake model whose output is a deterministic function of input_ids,
    with a per-row value so different documents in a batch produce
    different embeddings.
    """

    def model(**inputs):
        ids = inputs["input_ids"].float()
        batch_size, seq_len = ids.shape
        row_values = ids[:, :1]  # CLS-like first token, independent of padding
        hidden = row_values.unsqueeze(1).expand(batch_size, seq_len, dim).clone()
        return type("Output", (), {"last_hidden_state": hidden})()

    return model


class TestEmbedBatch:
    def test_returns_float32_array_of_expected_shape(self):
        texts = ["aspirin reduces inflammation", "metformin type 2 diabetes"]

        result = embed_batch(texts, _fake_tokenizer(), _fake_model(dim=4))

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (2, 4)

    def test_batch_order_matches_input_order(self):
        tokenizer, model = _fake_tokenizer(), _fake_model(dim=4)
        texts = ["alpha document text here", "beta completely different words entirely"]

        batch_result = embed_batch(texts, tokenizer, model)
        individual_a = embed_batch([texts[0]], tokenizer, model)
        individual_b = embed_batch([texts[1]], tokenizer, model)

        np.testing.assert_allclose(batch_result[0], individual_a[0], rtol=1e-5)
        np.testing.assert_allclose(batch_result[1], individual_b[0], rtol=1e-5)

    def test_deterministic_for_identical_batch(self):
        tokenizer, model = _fake_tokenizer(), _fake_model()
        texts = ["metformin type 2 diabetes", "aspirin reduces inflammation"]

        first = embed_batch(texts, tokenizer, model)
        second = embed_batch(texts, tokenizer, model)

        np.testing.assert_array_equal(first, second)

    def test_truncation_is_deterministic_for_long_input(self):
        tokenizer, model = _fake_tokenizer(max_tokens=8), _fake_model()
        long_text = " ".join(f"word{i}" for i in range(500))

        first = embed_batch([long_text], tokenizer, model)
        second = embed_batch([long_text], tokenizer, model)

        np.testing.assert_array_equal(first, second)

    def test_max_length_bound_is_applied(self):
        """embed_batch must pass max_length through to the tokenizer -
        this is the 512-token bound taken from NCBI's official usage
        example, not an arbitrary implementation detail.
        """
        received = {}

        def capturing_tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=None,
            return_tensors="pt",
        ):
            received["max_length"] = max_length
            return {"input_ids": torch.tensor([[1, 2, 3]])}

        embed_batch(["some text"], capturing_tokenizer, _fake_model())

        assert received["max_length"] == 512

    def test_raises_runtime_error_on_tokenizer_failure(self):
        def broken_tokenizer(*args, **kwargs):
            raise ValueError("tokenizer exploded")

        with pytest.raises(RuntimeError):
            embed_batch(["some text"], broken_tokenizer, _fake_model())

    def test_raises_runtime_error_on_model_failure(self):
        def broken_model(**inputs):
            raise RuntimeError("model exploded")

        with pytest.raises(RuntimeError):
            embed_batch(["some text"], _fake_tokenizer(), broken_model)

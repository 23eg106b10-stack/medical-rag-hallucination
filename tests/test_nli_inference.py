"""Unit tests for M5 batched NLI inference."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from verification.nli_inference import NLIInferenceError, run_nli_batch


class FakeTokenizer:
    """Lightweight fake tokenizer for testing tokenization calls."""

    def __init__(self) -> None:
        self.call_args_list: list[tuple[list[str], list[str]]] = []

    def __call__(
        self,
        premises: list[str],
        hypotheses: list[str],
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 512,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        self.call_args_list.append((list(premises), list(hypotheses)))
        batch_size = len(premises)
        return {
            "input_ids": torch.zeros((batch_size, 10), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, 10), dtype=torch.long),
        }


class FakeModel:
    """Lightweight fake model returning controlled logits."""

    def __init__(self, logits_generator=None) -> None:
        self.config = MagicMock()
        # Real PubMedBERT order: 0=contradiction, 1=entailment, 2=neutral
        self.config.id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}
        self._param = torch.nn.Parameter(torch.zeros(1))
        self._logits_generator = logits_generator
        self.forward_call_count = 0

    def parameters(self):
        yield self._param

    def __call__(self, **kwargs) -> MagicMock:
        self.forward_call_count += 1
        batch_size = kwargs["input_ids"].shape[0]
        if self._logits_generator:
            logits = self._logits_generator(batch_size, self.forward_call_count)
        else:
            # Default logits: high entailment (class 1), low contra (0), med neutral (2)
            # contra=0.0, entail=4.0, neutral=1.0
            row = torch.tensor([0.0, 4.0, 1.0])
            logits = row.repeat(batch_size, 1)

        mock_output = MagicMock()
        mock_output.logits = logits
        return mock_output


def test_run_nli_batch_empty_pairs() -> None:
    """Empty pair list returns empty list with zero forward passes."""
    tok = FakeTokenizer()
    model = FakeModel()
    scores = run_nli_batch([], tok, model, batch_size=32)
    assert scores == []
    assert model.forward_call_count == 0


def test_run_nli_batch_single_pair() -> None:
    """Single pair executes tokenization and produces an NLIScore."""
    tok = FakeTokenizer()
    model = FakeModel()
    pairs = [("Evidence text.", "Claim text.")]
    scores = run_nli_batch(pairs, tok, model, 32)

    assert len(scores) == 1
    assert scores[0].passage_pmid == ""
    assert scores[0].entailment_prob > 0.90
    assert scores[0].contradiction_prob < 0.05
    # Check premise and hypothesis order: premise = Evidence, hypothesis = Claim
    assert tok.call_args_list[0][0] == ["Evidence text."]
    assert tok.call_args_list[0][1] == ["Claim text."]


def test_run_nli_batch_chunking_and_order_preservation() -> None:
    """Chunks 5 pairs into batches of 2, preserving exact input ordering."""
    tok = FakeTokenizer()

    def generate_step_logits(batch_size: int, call_count: int) -> torch.Tensor:
        # Give distinct logits per call to verify order
        base = float(call_count)
        return torch.tensor([[base, base + 1.0, base + 2.0]]).repeat(batch_size, 1)

    model = FakeModel(logits_generator=generate_step_logits)
    pairs = [(f"Passage {i}", f"Claim {i}") for i in range(5)]
    scores = run_nli_batch(pairs, tok, model, 2)

    assert len(scores) == 5
    # Batch chunk sizes: 2, 2, 1 -> 3 forward calls
    assert model.forward_call_count == 3
    assert len(tok.call_args_list[0][0]) == 2
    assert len(tok.call_args_list[1][0]) == 2
    assert len(tok.call_args_list[2][0]) == 1

    assert [score.passage_pmid for score in scores] == ["" for _ in pairs]


def test_run_nli_batch_semantic_label_mapping_resolution() -> None:
    """Correctly assigns probabilities when class 0 is contradiction and class 1 is entailment."""
    tok = FakeTokenizer()

    # Logits: class 0 (contra) = 5.0, class 1 (entail) = 0.0, class 2 (neutral) = 0.0
    def contra_logits(batch_size: int, call_count: int) -> torch.Tensor:
        return torch.tensor([[5.0, 0.0, 0.0]]).repeat(batch_size, 1)

    model = FakeModel(logits_generator=contra_logits)
    scores = run_nli_batch([("Passage", "Claim")], tok, model, batch_size=1)

    assert len(scores) == 1
    assert scores[0].contradiction_prob > 0.95
    assert scores[0].entailment_prob < 0.05


def test_run_nli_batch_nan_logits_raises_nli_inference_error() -> None:
    """NaN logits raise NLIInferenceError."""
    tok = FakeTokenizer()

    def nan_logits(batch_size: int, call_count: int) -> torch.Tensor:
        return torch.tensor([[float("nan"), 1.0, 0.0]]).repeat(batch_size, 1)

    model = FakeModel(logits_generator=nan_logits)
    with pytest.raises(NLIInferenceError, match="NaN or Inf"):
        run_nli_batch([("Passage", "Claim")], tok, model, 32)


def test_run_nli_batch_inf_logits_raises_nli_inference_error() -> None:
    """Inf logits raise NLIInferenceError."""
    tok = FakeTokenizer()

    def inf_logits(batch_size: int, call_count: int) -> torch.Tensor:
        return torch.tensor([[float("inf"), 1.0, 0.0]]).repeat(batch_size, 1)

    model = FakeModel(logits_generator=inf_logits)
    with pytest.raises(NLIInferenceError, match="NaN or Inf"):
        run_nli_batch([("Passage", "Claim")], tok, model, 32)


def test_run_nli_batch_forward_crash_raises_nli_inference_error() -> None:
    """Runtime crash in forward pass raises NLIInferenceError."""
    tok = FakeTokenizer()
    crashing_generator = MagicMock(side_effect=RuntimeError("CUDA OOM"))
    model = FakeModel(logits_generator=crashing_generator)

    with pytest.raises(NLIInferenceError, match="Model forward pass failed"):
        run_nli_batch([("Passage", "Claim")], tok, model, 32)


def test_run_nli_batch_invalid_pair_structure_raises() -> None:
    """Non-tuple or non-2-element pairs raise NLIInferenceError."""
    tok = FakeTokenizer()
    model = FakeModel()
    with pytest.raises(NLIInferenceError, match="must be a 2-tuple"):
        run_nli_batch(["just a string"], tok, model, 32)  # type: ignore[list-item]

    with pytest.raises(NLIInferenceError, match="Pair elements at index 0 must be strings"):
        run_nli_batch([(123, "claim")], tok, model, 32)  # type: ignore[list-item]


def test_run_nli_batch_deterministic_repeated_calls() -> None:
    """Repeated calls produce identical floating point probabilities."""
    tok = FakeTokenizer()
    model = FakeModel()
    pairs = [("Evidence A", "Claim A"), ("Evidence B", "Claim B")]

    run1 = run_nli_batch(pairs, tok, model, batch_size=2)
    run2 = run_nli_batch(pairs, tok, model, batch_size=2)

    assert len(run1) == len(run2) == 2
    for s1, s2 in zip(run1, run2):
        assert s1.entailment_prob == s2.entailment_prob
        assert s1.neutral_prob == s2.neutral_prob
        assert s1.contradiction_prob == s2.contradiction_prob

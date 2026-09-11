"""Unit tests for M5 NLI model loader and label mapping validation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from config.settings import Settings
from verification.nli_loader import (
    VerifierLabelMappingError,
    load_nli_model,
    resolve_label_mapping,
)


def test_resolve_label_mapping_standard_pubmedbert_order() -> None:
    """PubMedBERT order (0: contradiction, 1: entailment, 2: neutral) resolves."""
    id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}
    mapping = resolve_label_mapping(id2label)
    assert mapping == {
        "contradiction": 0,
        "entailment": 1,
        "neutral": 2,
    }


def test_resolve_label_mapping_standard_mnli_order() -> None:
    """Standard MNLI order (0: entailment, 1: neutral, 2: contradiction) resolves correctly."""
    id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
    mapping = resolve_label_mapping(id2label)
    assert mapping == {
        "entailment": 0,
        "neutral": 1,
        "contradiction": 2,
    }


def test_resolve_label_mapping_case_and_whitespace_insensitive() -> None:
    """Resolving normalizes uppercase and leading/trailing whitespace."""
    id2label = {"0": " Contradiction ", "1": "ENTAILMENT", "2": "Neutral"}
    mapping = resolve_label_mapping(id2label)
    assert mapping == {
        "contradiction": 0,
        "entailment": 1,
        "neutral": 2,
    }


def test_resolve_label_mapping_missing_label_raises() -> None:
    """Missing any of the canonical three labels raises VerifierLabelMappingError."""
    id2label = {0: "entailment", 1: "neutral"}
    with pytest.raises(VerifierLabelMappingError, match="does not resolve to the canonical"):
        resolve_label_mapping(id2label)


def test_resolve_label_mapping_extra_unknown_labels_only_raises() -> None:
    """Unrecognized label set raises VerifierLabelMappingError."""
    id2label = {0: "positive", 1: "negative", 2: "neutral"}
    with pytest.raises(VerifierLabelMappingError):
        resolve_label_mapping(id2label)


def test_resolve_label_mapping_non_dict_raises() -> None:
    """Passing a non-dict to resolve_label_mapping raises VerifierLabelMappingError."""
    with pytest.raises(VerifierLabelMappingError, match="Expected id2label to be a dict"):
        resolve_label_mapping(["entailment", "neutral", "contradiction"])  # type: ignore[arg-type]


def test_resolve_label_mapping_invalid_index_raises() -> None:
    """Non-integer class index raises VerifierLabelMappingError."""
    id2label = {"abc": "entailment", 1: "neutral", 2: "contradiction"}
    with pytest.raises(VerifierLabelMappingError, match="Invalid class index"):
        resolve_label_mapping(id2label)


def test_load_nli_model_uses_settings_defaults() -> None:
    """load_nli_model consumes model name and CPU device from Settings."""
    mock_model = MagicMock()
    mock_model.config.id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}
    mock_tokenizer = MagicMock()

    with (
        patch(
            "verification.nli_loader.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        ) as mock_tok_from_pretrained,
        patch(
            "verification.nli_loader.AutoModelForSequenceClassification.from_pretrained",
            return_value=mock_model,
        ) as mock_model_from_pretrained,
        patch(
            "verification.nli_loader.get_settings",
            return_value=Settings(
                verifier_model_name="pritamdeka/PubMedBERT-MNLI-MedNLI",
                verifier_device="cpu",
                hf_token="",
            ),
        ),
    ):
        tokenizer, model, label_mapping = load_nli_model()

        assert tokenizer is mock_tokenizer
        assert model is mock_model
        assert label_mapping == {"contradiction": 0, "entailment": 1, "neutral": 2}
        mock_tok_from_pretrained.assert_called_once_with(
            "pritamdeka/PubMedBERT-MNLI-MedNLI",
            token=None,
        )
        mock_model_from_pretrained.assert_called_once_with(
            "pritamdeka/PubMedBERT-MNLI-MedNLI",
            token=None,
        )
        mock_model.eval.assert_called_once()
        mock_model.to.assert_called_once()


def test_load_nli_model_hf_token_passed_when_set() -> None:
    """load_nli_model forwards non-empty hf_token to transformers."""
    mock_model = MagicMock()
    mock_model.config.id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}
    mock_tokenizer = MagicMock()

    with (
        patch(
            "verification.nli_loader.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        ) as mock_tok_from_pretrained,
        patch(
            "verification.nli_loader.AutoModelForSequenceClassification.from_pretrained",
            return_value=mock_model,
        ) as mock_model_from_pretrained,
        patch(
            "verification.nli_loader.get_settings",
            return_value=Settings(hf_token="hf_secret_token"),
        ),
    ):
        load_nli_model()
        mock_tok_from_pretrained.assert_called_once_with(
            "pritamdeka/PubMedBERT-MNLI-MedNLI",
            token="hf_secret_token",
        )
        mock_model_from_pretrained.assert_called_once_with(
            "pritamdeka/PubMedBERT-MNLI-MedNLI",
            token="hf_secret_token",
        )


def test_load_nli_model_missing_id2label_raises() -> None:
    """Model lacking id2label raises VerifierLabelMappingError."""
    mock_model = MagicMock()
    mock_model.config.id2label = None
    mock_tokenizer = MagicMock()

    with (
        patch(
            "verification.nli_loader.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        ),
        patch(
            "verification.nli_loader.AutoModelForSequenceClassification.from_pretrained",
            return_value=mock_model,
        ),
    ):
        with pytest.raises(VerifierLabelMappingError, match="does not contain an id2label"):
            load_nli_model()


def test_load_nli_model_failure_raises_runtime_error() -> None:
    """Network or loading failure raises RuntimeError."""
    with patch(
        "verification.nli_loader.AutoTokenizer.from_pretrained",
        side_effect=OSError("Checkpoint not reachable"),
    ):
        with pytest.raises(RuntimeError, match="Failed to load NLI verifier model"):
            load_nli_model()

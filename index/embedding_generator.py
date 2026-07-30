"""MedCPT document embedding generation (M3.1.3).

Pure text -> vector conversion. Knows nothing about the corpus, FAISS, or
files on disk — that orchestration lives in ``faiss_pipeline``. Model
selection and batch size are owned by ``config.settings`` and resolved by
the caller (``build_faiss.py``); this module never reads settings itself.

Extraction strategy (CLS token from ``last_hidden_state``) and the
``max_length=512`` truncation bound match NCBI's official MedCPT usage
example for the article encoder — not an assumption. Truncation of
documents exceeding this length is deterministic (tokenizer-applied,
head truncation); no chunking or splitting is performed, per Design
Decision 8.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

_MAX_SEQUENCE_LENGTH = 512


def load_encoder(model_name: str) -> tuple[Any, Any]:
    """Load the MedCPT tokenizer and model for the given checkpoint.

    Args:
        model_name: Hugging Face model identifier (from
            ``settings.medcpt_model_name``).

    Returns:
        The ``(tokenizer, model)`` pair, with the model in eval mode.
    """
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def embed_batch(texts: list[str], tokenizer: Any, model: Any) -> np.ndarray:
    """Generate MedCPT embeddings for a batch of document texts.

    Uses the [CLS] token from ``last_hidden_state`` as the document
    representation, per NCBI's official MedCPT usage example (both the
    query and article encoders are used this way). Sequences longer than
    512 tokens are truncated deterministically by the tokenizer; no
    chunking or splitting is performed, per Design Decision 8.

    Args:
        texts: Batch of document texts (title + abstract per document,
            per Design Decision 2). Batch size is caller-controlled via
            ``settings.embedding_batch_size``.
        tokenizer: A loaded MedCPT tokenizer, from ``load_encoder``.
        model: A loaded MedCPT model, from ``load_encoder``.

    Returns:
        ``float32`` array of shape ``(len(texts), embedding_dim)``, in the
        same order as ``texts``.

    Raises:
        RuntimeError: if tokenization or model inference fails for the
            batch.
    """
    try:
        encoded = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=_MAX_SEQUENCE_LENGTH,
            return_tensors="pt",
        )
        with torch.no_grad():
            output = model(**encoded)
        embeddings = output.last_hidden_state[:, 0, :]
    except Exception as exc:
        raise RuntimeError(f"Batch embedding generation failed: {exc}") from exc

    return embeddings.detach().cpu().numpy().astype(np.float32)

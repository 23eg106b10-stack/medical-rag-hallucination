"""Hallucination verification orchestration layer (Milestone 5).

Evaluates extracted atomic claims against retrieved evidence context passages
using batched NLI inference and pure aggregation.

Enforces ADR-M5-001, ADR-M5-002, and ADR-M5-004:
- Context candidate set is the full GeneratedAnswer.context (no reranking/filtering).
- Evaluates PREMISE = passage.abstract against HYPOTHESIS = claim.claim_text.
- Delegates batched inference to `nli_inference.run_nli_batch`.
- Delegates verdict and evidence attribution to `evidence_aggregation.aggregate_evidence`.
- Returns self-contained `list[ClaimVerification]`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from config.settings import get_settings
from schemas.retrieval import HybridScoredDocument
from schemas.verification import ExtractedClaim
from schemas.verification_result import ClaimVerification, EvidenceAttribution, NLIScore
from verification.evidence_aggregation import aggregate_evidence
from verification.nli_inference import NLIInferenceError, run_nli_batch

logger = logging.getLogger(__name__)


class VerificationInputError(RuntimeError):
    """Raised when input to the ClaimVerifier fails structural or type validation."""


class ClaimVerifier:
    """Orchestrates NLI verification of atomic claims against retrieved context."""

    def __init__(
        self,
        model: Any = None,
        tokenizer: Any = None,
        batch_size: int | None = None,
        entailment_threshold: float | None = None,
        contradiction_threshold: float | None = None,
        nli_inference_fn: Callable[..., list[NLIScore]] | None = None,
    ) -> None:
        """Initialize ClaimVerifier with model, tokenizer, and runtime parameters.

        Args:
            model: Loaded sequence classification model.
            tokenizer: Matching Hugging Face tokenizer.
            batch_size: NLI inference batch size. If None, read from settings.
            entailment_threshold: Minimum prob for SUPPORTED. If None, from settings.
            contradiction_threshold: Minimum prob for CONTRADICTED. If None, from settings.
            nli_inference_fn: Optional custom inference function for testing/injection.
                Defaults to `run_nli_batch`.
        """
        settings = get_settings()

        self._model = model
        self._tokenizer = tokenizer
        self._batch_size = (
            batch_size if batch_size is not None else settings.verifier_nli_batch_size
        )
        self._entailment_threshold = (
            entailment_threshold
            if entailment_threshold is not None
            else settings.verifier_entailment_threshold
        )
        self._contradiction_threshold = (
            contradiction_threshold
            if contradiction_threshold is not None
            else settings.verifier_contradiction_threshold
        )
        self._nli_inference_fn = nli_inference_fn or run_nli_batch

    def verify_claims(
        self,
        claims: list[ExtractedClaim],
        context: list[HybridScoredDocument],
    ) -> list[ClaimVerification]:
        """Verify atomic claims against candidate evidence context passages.

        Args:
            claims: List of ExtractedClaim instances to verify.
            context: List of HybridScoredDocument context passages retrieved for
                the question.

        Returns:
            List of ClaimVerification objects containing verdict and evidence
            attribution for each claim, preserving claims order.

        Raises:
            VerificationInputError: If claims or context are malformed or invalid types.
            NLIInferenceError: If NLI scoring fails during evaluation.
        """
        # Validate inputs
        if not isinstance(claims, list):
            raise VerificationInputError(
                f"Expected claims to be a list, got {type(claims).__name__}."
            )
        for i, c in enumerate(claims):
            if not isinstance(c, ExtractedClaim):
                raise VerificationInputError(
                    f"Claim at index {i} is not an ExtractedClaim instance: {type(c).__name__}."
                )

        if not isinstance(context, list):
            raise VerificationInputError(
                f"Expected context to be a list, got {type(context).__name__}."
            )
        for j, doc in enumerate(context):
            if not isinstance(doc, HybridScoredDocument):
                raise VerificationInputError(
                    f"Context item at index {j} is not a HybridScoredDocument: "
                    f"{type(doc).__name__}."
                )

        # Failure boundary: Zero claims
        if not claims:
            return []

        # Failure boundary: Empty context -> All UNVERIFIABLE without NLI calls
        if not context:
            logger.warning(
                "ClaimVerifier received empty context; all %d claims marked UNVERIFIABLE "
                "with zero NLI calls.",
                len(claims),
            )
            return [
                ClaimVerification(
                    claim_text=claim.claim_text,
                    verdict="UNVERIFIABLE",
                    evidence=EvidenceAttribution(attributed_pmid=None, all_scores=[]),
                )
                for claim in claims
            ]

        # Construct (passage, claim) pairs across all claims and context documents
        # PREMISE = passage abstract
        # HYPOTHESIS = claim text
        all_pairs: list[tuple[str, str]] = []
        all_pmids: list[str] = []

        num_docs = len(context)
        for claim in claims:
            for doc in context:
                passage_text = doc.document.abstract
                claim_text = claim.claim_text
                all_pairs.append((passage_text, claim_text))
                all_pmids.append(doc.document.pmid)

        # Run batched NLI inference through the frozen four-argument interface.
        scores = self._nli_inference_fn(
            all_pairs,
            self._tokenizer,
            self._model,
            self._batch_size,
        )
        if len(scores) != len(all_pairs):
            raise NLIInferenceError(
                f"NLI returned {len(scores)} scores for {len(all_pairs)} input pairs."
            )
        for idx, score in enumerate(scores):
            score.passage_pmid = all_pmids[idx]

        # Aggregate evidence per claim
        results: list[ClaimVerification] = []
        for i, claim in enumerate(claims):
            start = i * num_docs
            end = start + num_docs
            claim_scores = scores[start:end]

            verdict, attribution = aggregate_evidence(
                claim_text=claim.claim_text,
                scores=claim_scores,
                entailment_threshold=self._entailment_threshold,
                contradiction_threshold=self._contradiction_threshold,
            )

            results.append(
                ClaimVerification(
                    claim_text=claim.claim_text,
                    verdict=verdict,
                    evidence=attribution,
                )
            )

        return results

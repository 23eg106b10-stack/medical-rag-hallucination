# Architectural Decision Records

This file records standalone architectural decisions — as distinct from
Architecture Change Requests, which amend already-frozen work (see
`ARCHITECTURE_CHANGE_REQUESTS.md`). Kept as a separate, append-only log
rather than folded into `ARCHITECTURE_BASELINE.md`, since the baseline is
rewritten (not appended) at each freeze and a growing decision log doesn't
fit that model — the same reasoning that already put ACRs in their own
file.

`ADR-001` predates this file's creation and is backfilled here from
`ARCHITECTURE_BASELINE.md` §5.3 for a single canonical source. `ADR-M3.3-001`
and `ADR-M3.3-002` were decided during M3.3 architecture review but were
never previously written to a durable document — backfilled here for the
same reason.

---

## ADR-001
**Title:** L2 Normalization for FAISS IndexFlatIP

**Affected milestone:**
M3.1.3

**Status:**
Accepted

**Decision:**
Embeddings are L2-normalized before insertion into `faiss.IndexFlatIP`,
enabling cosine similarity via inner product.

**Consequences:**
- Query vectors must be normalized identically at query time (see M3.2,
  `DenseRetriever.search`).
- `embedding_metadata.json` records `normalize_embeddings: true` for
  auditability.

---

## ADR-M3.3-001
**Title:** Inference Engine = Transformers + bitsandbytes

**Affected milestone:**
M3.3

**Status:**
Accepted

**Decision:**
Hugging Face Transformers + bitsandbytes (4-bit quantization) is the
mandatory inference backend for LLM generation. No other engine
(llama.cpp/GGUF, vLLM) is permitted.

**Rationale:**
- Matches Architecture v1.0 and the project blueprint.
- Same ecosystem as the frozen verifier checkpoint
  (`pritamdeka/PubMedBERT-MNLI-MedNLI`, also Transformers-based) —
  avoids running two separate inference ecosystems (llama.cpp for
  generation, Transformers for verification).
- Simplest reproducible stack; least cognitive overhead for a
  single-implementer research project.
- vLLM rejected outright: its continuous-batching, enterprise-serving
  design solves a problem this project doesn't have.

**Consequences:**
- `generation/llm_loader.py` is built against the Transformers
  `AutoModelForCausalLM` / `AutoTokenizer` / `BitsAndBytesConfig` API
  specifically; switching engines later is a breaking change to that
  module.

---

## ADR-M3.3-002
**Title:** Prompt Template Ownership

**Affected milestone:**
M3.3

**Status:**
Accepted

**Decision:**
`generation/prompts.py` contains only declarative template constants.
Never: string concatenation, formatting logic, context assembly, or
truncation. Those responsibilities belong exclusively to
`generation/context_builder.py`.

**Rationale:**
Keeps prompt templates declarative and inspectable independent of
assembly logic — mirrors the existing `fusion.py` (pure logic) /
`retriever.py` (orchestration) separation from M3.2.

**Consequences:**
- Any prompt-assembly logic accidentally added to `prompts.py` during
  future work is an architecture violation, not a style nitpick.
- See ACR-002 (`ARCHITECTURE_CHANGE_REQUESTS.md`) for how this ownership
  split was preserved when `build_prompt` was revised into
  `PromptBuilder`.

---

## ADR-M3.3-003
**Title:** Reserved

**Status:**
Reserved

**Note:**
Never assigned. Number intentionally unused after architecture review.
Kept reserved to preserve references if that number ever appeared in
discussions.

---

## ADR-M3.3-004
**Title:** Deterministic Generation Configuration

**Affected milestone:**
M3.3

**Status:**
Accepted

**Decision:**
Generation SHALL use `do_sample=False` by default.

**Rationale:**
- Deterministic outputs.
- Reproducible experiments.
- Consistent evaluation.
- Aligns with the project's research-reproducibility principle, already
  established for retrieval (deterministic tie-breaking, versioned
  queries, checksummed corpus).

**Consequences:**
- Generation is deterministic under default configuration.
- Temperature / top-p sampling remain future work, not currently
  exposed.
- Changing this default requires a future ADR — it is not a parameter to
  adjust silently via settings alone, despite `llm_do_sample` being a
  configurable field.

---

## ADR-M4-001

**Title:** Verification Unit = Atomic Medical Claim

**Affected milestone:**
M4

**Status:**
Accepted

**Decision:**
The verification unit throughout the verification pipeline SHALL be the atomic medical claim rather than the natural-language sentence.

**Rationale:**
- Natural-language sentence boundaries do not necessarily correspond to independent medical propositions.
- A single sentence may contain multiple claims requiring different verification outcomes.
- Natural Language Inference (NLI) models evaluate individual propositions; compound claims create ambiguous verification results.
- Claim-level verification requires one verification unit per proposition.

**Consequences:**
- Claim Extraction (M4) SHALL produce atomic medical claims as its output.
- Verification (M5) SHALL treat each extracted claim independently.
- This ADR defines the verification unit only.
- The extraction mechanism is intentionally not specified by this ADR.

**Out of Scope:**
This ADR does not define:
- claim extraction mechanism
- claim extraction schema
- repository integration
- implementation strategy
- transformation rules
- testing strategy

These decisions remain part of the M4 architecture review.

---

## ADR-M4-002
**Title:** Verification-Side Evidence Attribution

**Affected milestone:**
M4 / M5 boundary

**Status:**
Accepted

**Decision:**
Evidence attribution SHALL occur during M5 verification rather than
during M3.3 generation or M4 claim extraction. M3.3 `GeneratedAnswer`
remains unchanged, and M4 `ExtractedClaim` SHALL contain only
`claim_text`. For each extracted atomic claim, M5 SHALL evaluate the
claim against the retrieved evidence contained in
`GeneratedAnswer.context` and determine the evidence relevant to the
verification verdict. Generation-side inline citation markers are not
required.

**Rationale:**
- The original design's two apparent options were a false binary: verify
  every claim against the full unattributed top-5 context pool (weak
  evidence-grounding, undermines the project's core hallucination-
  detection thesis), or require frozen M3.3 to emit reliable inline
  `[1][2]` citation markers before M4 can proceed (reopens an
  already-frozen milestone and introduces a real reliability dependency
  on an 8B quantized instruct model correctly formatting exact citation
  tokens — a harder and more fragile problem than the evidence-
  attribution problem it was meant to solve).
- Evidence attribution is naturally a verification-side responsibility:
  M5 already exists to run NLI (`pritamdeka/PubMedBERT-MNLI-MedNLI`)
  between a claim and candidate evidence. Asking the dedicated NLI
  verifier to determine which passage(s) support or contradict a claim
  is the correct division of responsibility — asking a generation model
  to correctly self-report that same relationship via formatted markers
  is not.
- This preserves the frozen-milestone discipline already established
  elsewhere in this project (M3.1.2, M3.1.3, M3.2, M3.3): a defect or
  gap discovered after freeze is resolved via ACR against the milestone
  that owns it, or — as here — by resolving the dependency entirely on
  the non-frozen side of the boundary, not by silently reopening frozen
  work.
- This decision explicitly supersedes the "mandatory inline citation
  markers" intention recorded in `UNIFIED_ARCHITECTURE.md` §6 for this
  project's M4/M5 boundary — that document is supporting/historical
  material, not authoritative over `ARCHITECTURE_BASELINE.md`, and this
  ADR is the authoritative resolution going forward.

**Closed by this ADR:**
- `ExtractedClaim` remains exactly:
  ```python
  class ExtractedClaim(BaseModel):
      claim_text: str
  ```
- No `cited_passage_ids` field is added to `ExtractedClaim`.
- No claim ID / ordinal is added to `ExtractedClaim` (reaffirms the
  claim-identity decision already closed during M4 architecture review:
  Python list ordering is sufficient for the current in-process M4→M5
  boundary).
- No modification to `GeneratedAnswer` (`schemas/generation.py`).
- No inline citation-marker requirement is imposed on `generation/prompts.py`
  or any other M3.3 component.

**Left open by this ADR — belongs to M5 architecture review:**
- How the verifier selects/attributes evidence among the candidate
  passages in `GeneratedAnswer.context` for a given extracted claim.
- NLI threshold values.
- Number of evidence passages selected per claim (single strongest
  match vs. multiple).
- Entailment-vs-contradiction ranking/scoring formula.
- Multi-passage aggregation strategy, if any.
- M5's output schema (e.g. `{claim_text, verdict, raw_nli_scores}` per
  `UNIFIED_ARCHITECTURE.md` §7 is illustrative context only, not frozen
  by this ADR).

**Consequences:**

*Positive:*
- No reopening of frozen M3.3.
- No change to `GeneratedAnswer`.
- No change to `ExtractedClaim`.
- No fragile citation-format dependency on the generation model.
- M4 can proceed independently of any M3.3 change.
- M5 retains explicit, per-claim evidence attribution — stronger
  evidence-grounding than verifying against the full unattributed
  context pool.

*Negative:*
- M5 performs additional NLI evaluations per claim (up to one per
  candidate passage in `GeneratedAnswer.context`, rather than one
  evaluation against a single pre-identified passage).
- Evidence attribution becomes a verifier responsibility that must be
  designed carefully during M5 architecture review — it is not free,
  only deferred to the component best positioned to do it well.
- The exact attribution/ranking algorithm remains genuinely undecided
  until M5 architecture review; this ADR resolves the M4/M5 boundary
  contract, not M5's internals.

**Out of Scope:**
This ADR does not define:
- claim extraction mechanism (still governed by ADR-M4-001's "Out of
  Scope" and remains part of ongoing M4 architecture review)
- M5 evidence-attribution algorithm
- M5 NLI threshold or aggregation policy
- M5 output schema
- the retrieval minimum-similarity gate (`UNIFIED_ARCHITECTURE.md` §5.3)
  — a separate, still-open Retrieval→Generation boundary gap that does
  not block M4 or M5 and is out of scope for this ADR

---

## ADR-M4-003
**Title:** Atomic Claim Extraction Specification (LLM-Based Decomposition)

**Affected milestone:**
M4

**Status:**
Accepted

**Decision:**

This ADR specifies the claim extraction mechanism left open by ADR-M4-001
("Out of Scope: claim extraction mechanism, ... transformation rules").
ADR-M4-001 is not re-decided here; its verification-unit decision
(atomic medical claim) is the premise this ADR builds on.

1. **Mechanism.** Claim extraction is LLM-based decomposition: the
   already-loaded Llama-3.1-8B-Instruct generation model (loaded via
   the existing `generation.llm_loader.load_generation_model`, per
   ADR-M3.3-001) is prompted to decompose `GeneratedAnswer.answer_text`
   into one atomic medical claim per output line. No separate model,
   rules engine, or NLI-based decomposition is introduced.

2. **Determinism.** Extraction generation SHALL use `do_sample=False`
   (greedy decoding), matching ADR-M3.3-004's deterministic-generation
   principle. Extraction is a second, independent generation call
   against the same loaded model/tokenizer used for answer generation
   (or an equivalently configured one) — it does not reuse or mutate
   the `GeneratedAnswer` produced by M3.3.

3. **Prompt ownership and location.** Per ADR-M3.3-002's precedent, the
   extraction prompt template is a declarative-only constant, located
   at `verification/claim_extraction_prompts.py`. This is a new module,
   not an addition to `generation/prompts.py` — `generation/prompts.py`
   remains scoped by ADR-M3.3-002 to M3.3 answer generation only. Prompt
   assembly and output parsing are separate responsibilities, mirroring
   the `prompts.py` / `context_builder.py` split already established
   for M3.3: `verification/claim_extraction_prompts.py` holds only the
   template constants; a separate parser module consumes raw model
   output.

4. **Expected model output format.** One claim per line, plain text —
   explicitly NOT JSON, NOT a structured/delimited format. This is a
   deliberate low-fragility choice: JSON-mode extraction from an 8B
   quantized instruct model reintroduces exactly the format-reliability
   risk ADR-M4-002 rejected for inline citation markers.

5. **Parsing rules (deterministic, non-semantic):**
   - Split raw output on newlines.
   - Strip each line of leading numbering/bullet markers (e.g. `1.`,
     `-`, `*`) and surrounding whitespace.
   - Discard blank lines.
   - Strip inline PMID/reference markers (e.g. `[PMID: 12345678]`,
     `(PMID: 12345678)`, `(References: [PMID: ...])`) from the
     remaining `claim_text`. Citation stripping happens here, not in
     the LLM prompt — the prompt does not instruct the model to omit
     citations; the parser removes them post-hoc, since citation
     format varies (per calibration §2: bracketed, parenthetical, and
     combined-reference forms) and relying on the model to reliably
     omit them is the same fragile dependency already rejected in
     ADR-M4-002.
   - Discard incomplete fragments under an observable, string-level
     contract — no grammatical parsing is performed. A line is
     discarded as an incomplete fragment if either:
     (a) it does not end in terminal sentence punctuation (`.`, `!`,
     `?`), or a closing quote/parenthesis immediately following one; or
     (b) it contains an unbalanced bracket or parenthesis pair (an
     opening `(` or `[` — most commonly from a citation marker — with
     no matching close). This directly matches the calibration-observed
     failure shapes (`Therefore, it`, `pharmacogenomics biom`,
     `(PMID:`), all of which fail (a) and/or (b), and requires no
     grammar or completeness judgment beyond these two string checks.
   - Discard lines that are pure conversational/generation-process
     commentary — narrowly: statements about the model's own
     response, behavior, or the exchange itself (offers to rephrase,
     disclaimers addressed to the user, apologies, self-referential
     framing such as "Revised rephrased answer:"). This is narrower
     than "meta-commentary" in general: a line expressing evidentiary
     uncertainty about the medical question itself (e.g. "the evidence
     does not support a clear conclusion regarding X") is a substantive
     claim about the evidence and is NOT discarded under this rule —
     only commentary about the model/conversation is. (Calibration
     Sample 6 illustrates the distinction: its 4 retained unique claims
     were evidentiary statements; only the repeated "Revised rephrased
     answer:" framing around them was process commentary.)
   - Deduplicate remaining lines using **string-level normalization
     only**: lowercase, strip punctuation, collapse whitespace, then
     exact-match comparison. No stemming, no lemmatization, no
     embedding/semantic similarity.

6. **Atomicity and anaphora resolution are an extraction-generation
   contract, not a parser responsibility.** Per rules 1 and 4, the
   extraction prompt requires the model to emit already-atomic,
   already-resolved claims — one independently verifiable medical
   proposition per line, with anaphoric references (e.g. "these risk
   factors") grounded to explicit entities. This is a generation-side
   contract enforced by prompt instruction, not by downstream
   validation. The parser performs no semantic decomposition and no
   semantic atomicity detection: it does not split compound lines, does
   not resolve references, and does not evaluate whether a line is
   "really" atomic. The parser's rule 5 checks (fragment completeness,
   process-commentary exclusion, normalization dedup) are the only
   filtering it performs, and none of them detect compound or
   unresolved content. Whether the model's output actually satisfies the
   atomicity/resolution contract is therefore not independently checked
   by this ADR's pipeline.

7. **Claim-count gate.**
   - The gate is evaluated on the **post-filter, post-dedup unique
     valid claim count** — i.e. after numbering/whitespace stripping,
     blank-line removal, citation stripping, incomplete-fragment
     filtering, process-commentary filtering, and normalization dedup —
     not on raw output line count. This matches the metric the
     calibration analysis (§4) actually calibrated against ("Unique
     Atomic Claims": mean 12.30, max 24), not "Total Candidate Claims"
     (max 32), which is dominated by greedy-decoding repetition loops
     observed in 9/10 calibration samples and is not a proxy for
     genuine claim density.
   - `0` unique valid claims is a valid result (e.g. an answer that is
     entirely process commentary).
   - `1..35` unique valid claims is a valid result.
   - `>35` unique valid claims raises `ClaimExtractionError`. No silent
     truncation to the first 35 is performed.

8. **`MAX_CLAIMS = 35`.** Adopted per calibration §4.4 Option B: +11
   claims above the observed maximum of 24 (N=10, +45.8%). Empirical
   distribution: mean 12.30, median 10.5, σ=6.85; μ+2σ=26.00,
   μ+3σ=32.85. 35 sits above μ+3σ. No population-coverage percentage
   (e.g. "covers X% of the distribution") is asserted from this — N=10
   is too small to support a normality-based coverage claim, and this
   ADR relies only on the descriptive statistics and the direct
   comparison to the observed maximum, not on a distributional coverage
   inference. `MAX_CLAIMS` is a named constant in the M4 extraction
   module (not a `config.settings` field), consistent with
   ADR-M3.3-004's precedent that behavior-defining constants tied to a
   specific ADR are not silently adjustable via settings.

9. **Malformed-output failure boundary.** Raw model output is malformed
   if, after line-splitting (rule 5's first step) and blank-line
   discard, zero candidate lines remain — e.g. output that is
   empty/whitespace-only. In that case, `ClaimExtractionError` is
   raised, and no silent fallback to treating the whole, unparsed
   `answer_text` as a single claim is performed. A single valid claim
   line — with or without a trailing newline — is not malformed: one
   candidate line after splitting is sufficient to proceed to the rest
   of rule 5's filtering. This is distinct from rule 7's zero-claims
   case: output that produces at least one candidate line but *filters
   down* to zero valid claims (e.g. all lines filtered as process
   commentary) is valid; output that produces *no candidate lines at
   all* is malformed and errors. `ClaimExtractionError` is a single
   exception type covering both the >35 gate (rule 7) and
   malformed/unparseable output (this rule), per the specification
   driving this ADR — unlike `EmptyGenerationError` in `generator.py`,
   which is scoped to one specific failure mode. A future ADR may split
   this if callers need to distinguish the two programmatically.

10. **Schema boundary (unchanged).** `ExtractedClaim` remains exactly:
```python
    class ExtractedClaim(BaseModel):
        claim_text: str
```
    No claim ID, ordinal, or `cited_passage_ids` field is added
    (reaffirms ADR-M4-002's closed decision). `GeneratedAnswer`
    (`schemas/generation.py`) is not modified. Evidence attribution
    remains an M5 verification-side responsibility per ADR-M4-002 and
    is not designed, extended, or touched by this ADR.

**Rationale:**
- Calibration evidence (N=10 real E2E samples,
  `m4_calibration_reconciliation.md`) empirically disproved the
  provisional token-density heuristic (~3 claims/100 tokens ⟹ max ~25);
  Sample 2 alone produced 24 unique claims from 514 tokens. `MAX_CLAIMS`
  is set from the observed distribution and its μ/σ, not a theoretical
  bound or an unsupported coverage-percentage inference from N=10.
- One-claim-per-line plain-text output (not JSON) is the lower-fragility
  choice for an 8B quantized instruct model, consistent with the
  reasoning in ADR-M4-002 that rejected relying on this same model
  class to reliably emit a stricter structured format.
- Deterministic, string-only deduplication keeps the extraction
  pipeline free of a second model dependency and keeps its failure
  modes fully inspectable, given 9/10 calibration samples exhibited
  greedy-decoding repetition loops that a purely mechanical dedup step
  must absorb.
- The gate applies to the unique valid claim count rather than the raw
  candidate count because raw-candidate count is dominated by decoding
  repetition artifacts, not claim density (calibration §4.1: raw mean
  20.20/σ8.85 vs. unique mean 12.30/σ6.85).
- The incomplete-fragment and process-commentary filters are defined as
  observable, string-level checks specifically so they don't smuggle in
  semantic judgment through the back door of "obviously" filtering —
  each calibration-observed failure case (truncated tokens, unbalanced
  citation brackets, self-referential rephrasing language) maps to one
  of the two explicit checks in rule 5, not to a discretionary quality
  read of the line.
- Atomicity and anaphora resolution are placed on the generation side
  (rule 6) rather than given a parser-side enforcement mechanism because
  no such mechanism is specified by this ADR and none is introduced by
  it — inventing one here would exceed this ADR's scope and add an
  unreviewed heuristic to a fail-loud pipeline.
- This ADR does not touch `generation/`, `schemas/generation.py`, or
  `ARCHITECTURE_BASELINE.md` directly (baseline correction, if any,
  follows the existing freeze-time convention per §11, same as
  ACR-003/ACR-004).

**Consequences:**

*Positive:*
- M4 has a fully specified extraction mechanism, output contract, and
  two explicit fail-loud failure boundaries (malformed / over-limit),
  closing the "Out of Scope" gap left by ADR-M4-001.
- No new schema surface; `ExtractedClaim` and `GeneratedAnswer` remain
  exactly as ADR-M4-002 closed them.
- `MAX_CLAIMS=35` is empirically grounded against the calibrated
  metric, with a plain, re-checkable arithmetic justification (margin
  above observed max, μ/σ) rather than a statistical coverage claim
  that N=10 couldn't actually support.

*Negative:*
- A second deterministic generation call is required per question, in
  addition to answer generation, increasing per-question latency and
  GPU time on the ACR-004 hardware-constrained model.
- Atomicity and anaphora resolution (rule 6) are a generation-prompt
  contract with no independent downstream check. If the model emits a
  still-compound or still-unresolved line, this ADR's pipeline has no
  mechanism to catch it before it becomes an `ExtractedClaim` — this is
  an accepted gap, not an oversight, since closing it would require
  introducing a validator this ADR is explicitly not authorized to add.
- `ClaimExtractionError` conflating two distinct failure causes
  (over-limit vs. malformed) into one exception type means callers
  cannot distinguish "the model produced too much" from "the model
  produced garbage" from exception type alone.

**Out of Scope:**
This ADR does not define:
- M5 NLI verification, evidence attribution, or thresholding (remains
  governed by ADR-M4-002's open items).
- Any mechanism for detecting non-atomic or unresolved-reference lines
  downstream of generation — rule 6 fixes atomicity/resolution as a
  generation-side contract only; no detection or enforcement mechanism
  is introduced or specified by this ADR.
- Any change to `generation/`, `schemas/generation.py`, or the frozen
  M3.3 architecture.
- Retry/backoff behavior on `ClaimExtractionError` (whether the caller
  re-prompts, fails the question, or does something else is a
  verification-pipeline orchestration decision, not this ADR's).

---

## ADR-M5-001
**Title:** Evidence Candidate Set and Semantic Direction

**Affected milestone:**
M5

**Status:**
Accepted

**Decision:**
1. The evidence candidate set for claim verification is the complete retrieved context from generation:
   `GeneratedAnswer.context` (list of `HybridScoredDocument`).
2. No secondary reranker, pre-filter, or relevance thresholding is applied prior to NLI verification. Every passage in `GeneratedAnswer.context` is evaluated against every extracted claim.
3. The NLI semantic direction is:
   - `PREMISE` = evidence passage abstract (`doc.document.abstract`)
   - `HYPOTHESIS` = extracted atomic claim (`claim.claim_text`)
   - Pair representation: `(passage_text, claim_text)`.

**Rationale:**
- Preserves complete alignment with what the generation model saw as evidence.
- Eliminates upstream retrieval bias or cascading false negatives from a second heuristic filter.
- Formulates verification strictly as whether the cited/retrieved evidence entails or contradicts the atomic claim.

---

## ADR-M5-002
**Title:** Evidence Aggregation Policy and Contradiction Priority

**Affected milestone:**
M5

**Status:**
Accepted

**Decision:**
1. Verification evaluates candidate passage NLI probabilities via max-pooling across candidate evidence passages per claim:
   - `max_contra = max(s.contradiction_prob for s in candidate_scores)`
   - `max_entail = max(s.entailment_prob for s in candidate_scores)`
2. Contradiction priority: Contradiction is evaluated first. If `max_contra >= contradiction_threshold` (default 0.5), the verdict is `CONTRADICTED`.
3. Entailment evaluation: If not contradicted and `max_entail >= entailment_threshold` (default 0.5), the verdict is `SUPPORTED`.
4. Fallback verdict: If neither threshold is crossed, the verdict is `UNVERIFIABLE`.
5. Winning evidence attribution:
   - For `CONTRADICTED`: PMID of the candidate passage with the highest contradiction probability.
   - For `SUPPORTED`: PMID of the candidate passage with the highest entailment probability.
   - For `UNVERIFIABLE`: `attributed_pmid = None`.
6. Audit trail preservation: All evaluated passage scores are retained in `EvidenceAttribution.all_scores`.

**Rationale:**
- In clinical medical question answering, direct contradiction of a factual claim by any retrieved passage indicates severe risk/hallucination that overrides partial entailment elsewhere.
- Retaining all candidate scores preserves a full audit trail for downstream confidence calibration and error analysis.

---

## ADR-M5-003
**Title:** NLI Cross-Encoder Inference and Public API Contract

**Affected milestone:**
M5

**Status:**
Accepted

**Decision:**
1. Verifier model checkpoint: `pritamdeka/PubMedBERT-MNLI-MedNLI` on CPU by default.
2. Dynamic label mapping: Semantic labels (`"entailment"`, `"neutral"`, `"contradiction"`) are resolved dynamically from `model.config.id2label` at runtime. Numeric label indices are never hardcoded.
3. Public NLI API contract:
   ```python
   def run_nli_batch(
       pairs: list[tuple[str, str]],
       tokenizer: Any,
       model: Any,
       batch_size: int,
   ) -> list[NLIScore]:
   ```
4. Batching and chunking: `run_nli_batch` owns internal chunking by `batch_size` and guarantees deterministic, 1-to-1 ordered output preserving `results[i] <-> pairs[i]`.
5. Pure raw probabilities: `run_nli_batch` returns raw softmax probabilities (`entailment_prob`, `neutral_prob`, `contradiction_prob`) under `torch.inference_mode()`. It applies no thresholding or verdict logic.

**Rationale:**
- Keeps the public inference API minimal, decoupled, and focused purely on batched tensor execution.
- Dynamic label mapping prevents silent misclassification across different NLI checkpoints or revisions.

---

## ADR-M5-004
**Title:** Verification Orchestration and PMID Ownership

**Affected milestone:**
M5

**Status:**
Accepted

**Decision:**
1. `ClaimVerifier` (`verification/verifier.py`) is the central orchestrator for Milestone 5.
2. `ClaimVerifier` constructs `(passage_text, claim_text)` pairs across all claims and context documents, delegates scoring to `run_nli_batch`, and owns PMID association:
   - It attaches `doc.document.pmid` to the returned `NLIScore` objects using input ordering.
3. `ClaimVerifier` slices scores per claim and delegates aggregation to `evidence_aggregation.aggregate_evidence`.
4. Output packaging: Returns a list of `ClaimVerification` objects, preserving input claim ordering.
5. Error boundaries:
   - Non-list or mistyped claims/context raise `VerificationInputError`.
   - Zero claims returns empty list `[]` without NLI execution.
   - Empty context returns all claims as `UNVERIFIABLE` with zero NLI execution.
   - NLI execution failure raises `NLIInferenceError`.

---

## ADR-M6-001
**Title:** Contradiction Hard-Ceiling Policy

**Affected milestone:**
M6

**Status:**
Accepted

**Decision:**
1. Affirmative contradiction signal: Any `CONTRADICTED` claim in `verifications` triggers the contradiction ceiling mechanism.
2. If `contradicted_count > 0`, the final score is capped: `score = min(base_score, contradiction_ceiling)`.
3. The flag `contradiction_ceiling_applied` is defined strictly as `contradicted_count > 0 and base_score > contradiction_ceiling`. It evaluates to `True` only when the ceiling actively constrained the numerical score. If `base_score <= contradiction_ceiling`, `contradiction_ceiling_applied` is `False`.
4. The hard-ceiling mechanism is frozen. The default numeric ceiling value of `0.2` is provisional and uncalibrated pending ground-truth evaluation.

**Rationale:**
Extends M5's contradiction-priority policy (`ADR-M5-002`) to the answer level. In clinical medical QA, factual contradiction of an atomic claim by retrieved literature indicates high hallucination risk that overrides factual support on other claims.

---

## ADR-M6-002
**Title:** Unverifiable Claims via Denominator Dilution

**Affected milestone:**
M6

**Status:**
Accepted

**Decision:**
1. `UNVERIFIABLE` claims contribute to total claims `n` (the denominator of `base_score = s / n`), but do not contribute to the numerator `s`.
2. No explicit additional penalty or multiplier is applied to unverifiable claims.
3. `UNVERIFIABLE` claims do not activate the contradiction hard ceiling.

**Rationale:**
An unverifiable claim represents a retrieval-coverage gap or epistemic uncertainty, not an affirmative hallucination or proven error. Treating retrieval gaps as harshly as contradicted claims would penalize answers for literature gaps identically to factual falsehoods.

---

## ADR-M6-003
**Title:** Verdict-Count-Based Scoring; NLI Probabilities Excluded from Frozen Formula

**Affected milestone:**
M6

**Status:**
Accepted

**Decision:**
1. Base score calculation is strictly count-based: `base_score = supported_count / total_claims`.
2. Softmax probabilities (`entailment_prob`, `neutral_prob`, `contradiction_prob`) from M5's `NLIScore` are excluded from the M6 scoring formula.
3. M5 probability data remains available as an evidence and audit trail within `ClaimVerification.evidence.all_scores`, preserved read-only.
4. M6 loads no model, tokenizer, or inference pipeline; it is a deterministic pure function over already-computed M5 verification results.

**Rationale:**
Cross-encoder softmax outputs are uncalibrated as answer-level probabilities; incorporating their raw continuous magnitudes into the headline confidence score would introduce false precision without empirical grounding.

---

## ADR-M6-004
**Title:** M6 Output Schema and Zero-Claim Sentinel Semantics

**Affected milestone:**
M6

**Status:**
Accepted

**Decision:**
1. Introduce `schemas/confidence_result.py` with `ConfidenceResult(BaseModel)`:
   - `score: float | None`
   - `level: Literal["HIGH", "MEDIUM", "LOW", "NOT_APPLICABLE"]`
   - `total_claims: int`
   - `supported_count: int`
   - `contradicted_count: int`
   - `unverifiable_count: int`
   - `contradiction_ceiling_applied: bool`
2. Invariants (frozen):
   - `total_claims == supported_count + contradicted_count + unverifiable_count`
   - All counts `>= 0`
   - `score is None` $\iff$ `total_claims == 0` $\iff$ `level == "NOT_APPLICABLE"`
   - `total_claims == 0` $\implies$ `contradiction_ceiling_applied == False`
   - `total_claims > 0` $\implies$ `score` $\in [0.0, 1.0]$ and `level` $\in \{"\text{HIGH}", "\text{MEDIUM}", "\text{LOW}"\}$
3. Zero-claim sentinel: When `verifications == []`, return `score = None`, `level = "NOT_APPLICABLE"`, all counts `0`, and `contradiction_ceiling_applied = False`. Zero claims signifies that no verifiable claims were available for evaluation, not that the answer is completely contradicted (`0.0`).
4. Diagnostic counts only: No claim texts or passage texts are duplicated into `ConfidenceResult`.

**Rationale:**
Preserves clean schema boundaries and explicit sentinel semantics for empty extractions, preventing downstream systems from conflating "no claims extracted" with "completely false answer".

---

## ADR-M6-005
**Title:** Confidence Thresholds Provisional Pending Ground-Truth Evaluation

**Affected milestone:**
M6

**Status:**
Accepted

**Decision:**
1. The architectural mechanisms (verdict-count scoring, contradiction hard ceiling, denominator dilution, ordered categorical boundary checks) are frozen.
2. The numeric defaults:
   - `confidence_contradiction_ceiling = 0.2`
   - `confidence_level_high_threshold = 0.8`
   - `confidence_level_medium_threshold = 0.5`
   are explicitly designated as **provisional and uncalibrated**.
3. Small-sample calibration passes (e.g. N=10) are explicitly rejected for M6. Meaningful calibration requires evaluation against ground-truth answer correctness on a representative dataset in M7.

**Rationale:**
Prevents provisional heuristic values from being treated as empirically or clinically validated thresholds prior to systematic ground-truth evaluation.


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

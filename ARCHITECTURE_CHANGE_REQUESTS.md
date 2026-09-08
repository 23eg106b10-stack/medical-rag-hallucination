# Architecture Change Requests

## ACR-001
**Title:** Persist FAISS PMID Mapping

**Affected milestone:** 
M3.1.3

**Reason:** 
FAISS artifacts lacked deterministic vector-position → PMID mapping.

**Decision:** 
Generate `faiss_pmids.json` alongside `faiss_index.bin`.

**Status:** 
Approved

---

## ACR-002
**Title:** Prompt Builder Contract Revision

**Affected milestone:**
M3.3

**Reason:**
The frozen `build_prompt(question, context) -> str` free-function contract could not satisfy the mandatory "raise on context overflow" failure boundary. Enforcing a token budget requires counting tokens against a real tokenizer, but the function's signature had no parameter through which to receive one — a genuine contradiction between two frozen requirements, discovered during M3.3 implementation, not an implementation mistake.

**Decision:**
Replace the module-level `build_prompt` free function with an injectable `PromptBuilder` class. `tokenizer` and `context_window` are supplied via constructor dependency injection; `build_prompt` becomes an instance method.

**Status:**
Closed

**Implementation:**
Implemented

**Implementation Summary:**
- `PromptBuilder` replaced the free function `build_prompt`.
- Hidden tokenizer/context-window dependencies removed — both are now explicit constructor parameters, no longer implicit assumptions.
- `Generator`'s dependency-injection contract was preserved unchanged: a bound `PromptBuilder.build_prompt` method satisfies the existing `Callable[[str, list[HybridScoredDocument]], str]` signature with zero modification to `generator.py`.
- `PromptOverflowError` introduced; raised when the assembled prompt exceeds `context_window` tokens.
- Zero prompt truncation policy: overflow is always treated as an error, never as a signal to silently drop or shorten retrieved evidence.

Full architecture detail: [`ARCHITECTURE_BASELINE.md`, §5.5](./ARCHITECTURE_BASELINE.md#55-m33--llm-generation)

---

## ACR-003
**Title:** Correct BM25 Artifact Filename (`bm25_index.pkl`, not `.json`)

**Affected milestone:**
M3.1.2 (frozen)

**Reason:**
A repository audit found that the frozen `ARCHITECTURE_BASELINE.md` names the
BM25 index artifact `data/indexes/bm25_index.json` in §5.2 and §6, while
describing that same artifact's contents as a pickled payload —
`{"model": BM25Okapi, "pmids": [...]}`. A `.json` extension over a pickle
binary is a contradiction in the frozen document, not a stale-but-valid
description.

The actual `index/bm25_pipeline.py` implementation was already correct: it
serializes via `pickle.dump` and the real production artifact on disk is
`data/indexes/bm25_index.pkl` (confirmed against the live 171,858-document
corpus build and the repository tree). `config/settings.py` had inherited
the architecture's incorrect filename (`bm25_index_filename =
"bm25_index.json"`), and `index/build_bm25.py` had silently diverged from
that setting by hardcoding the literal `"bm25_index.pkl"` instead of
consuming `settings.bm25_index_filename` — itself a separate violation of
the project's configuration-singleton rule (`ARCHITECTURE_BASELINE.md`
§4, §8.1: all runtime configuration must flow through `get_settings()`).

Full investigation detail: see repository audit findings on the BM25
artifact naming/format discrepancy.

**Decision:**
`bm25_index.pkl` is the canonical BM25 artifact filename and pickle
remains the canonical serialization format. No serialization or artifact
format change is made — only the filename declared in configuration, and
the path-construction logic that had bypassed configuration, are
corrected to match the artifact format that was already correct.

Concretely:
- `config/settings.py`: `bm25_index_filename` default changed from
  `"bm25_index.json"` to `"bm25_index.pkl"`.
- `index/build_bm25.py`: the hardcoded `"bm25_index.pkl"` literal is
  replaced with `settings.indexes_dir / settings.bm25_index_filename`, so
  the production path is actually derived from configuration rather than
  duplicating it.
- `tests/test_bm25_pipeline.py`: a regression test is added asserting
  that the build path is derived from `settings.bm25_index_filename`,
  specifically to catch any future reintroduction of a hardcoded filename
  literal in the build entry point.
- `ARCHITECTURE_BASELINE.md` is **not** corrected by this ACR. Per §11 of
  that document, it is rewritten only at the next milestone freeze; this
  ACR records the correction to be folded in at that time.
- `index/bm25_pipeline.py` and `retrieval/sparse.py` are unchanged: both
  already treat the artifact as a pickle and required no correction.

**Rationale:**
- The artifact's actual, already-shipped format is pickle — changing
  format to match the erroneous `.json` extension would mean converting a
  real production artifact and its consumer (`BM25Retriever.from_pickle`)
  to satisfy a documentation error, which is backwards.
- Correcting the filename in configuration without also wiring
  `build_bm25.py` to consume it would leave the dead-config-field problem
  in place; both defects share one root cause (the architecture's
  original naming error) and are corrected together.
- This is a mid-milestone correction to already-frozen M3.1.2 output, not
  a new architectural decision — it belongs in the ACR log, per the
  ADR/ACR split already established in `DECISIONS_LOG.md`.

**Status:**
Approved

**Implementation:**
Implemented

**Affected files:**
- `config/settings.py`
- `index/build_bm25.py`
- `tests/test_bm25_pipeline.py`

**Explicitly not modified:**
- `ARCHITECTURE_BASELINE.md` (corrected at next freeze, per §11)
- `DECISIONS_LOG.md`
- `index/bm25_pipeline.py`
- `retrieval/sparse.py`
- `PROJECT_STATE.md`

**Validation requirements:**
- Existing BM25 test suite passes unmodified.
- New regression test (`tests/test_bm25_pipeline.py`) passes and fails if
  `index/build_bm25.py` reverts to a hardcoded filename.
- Full pytest suite, ruff, black, and compileall all pass clean.
- `settings.indexes_dir / settings.bm25_index_filename` resolves to
  `data/indexes/bm25_index.pkl`.
- No production module hardcodes `"bm25_index.pkl"` where
  `settings.bm25_index_filename` should be consumed instead.
- No existing runtime artifact (`data/indexes/bm25_index.pkl`, the live
  corpus, or FAISS artifacts) is regenerated, moved, or renamed by this
  change — the artifact was already correct; only the code path that
  names it was wrong.

---

## ACR-004
**Title:** Explicit GPU-Resident Placement for Quantized Decoder Layers (Llama-3.1-8B-Instruct, 4-bit NF4)

**Affected milestone:**
M3.3 (frozen)

**Reason:**
The frozen loader's `device_map="auto"` was found, on the project's
actual RTX 3050 6GB target hardware, to dispatch some `Linear4bit`
(4-bit quantized) decoder layers to CPU. At forward time, Accelerate's
`AlignDevicesHook` attempts to move such a module's `QuantState` to
CUDA, which fails with `NotImplementedError: Cannot copy out of meta
tensor; no data!`. This occurred on Transformers 5.16.1, Accelerate
1.14.0, and bitsandbytes 0.50.2, and was confirmed independent of
`low_cpu_mem_usage`, which was tested directly and produced the
identical failure. No `Linear4bit` module can be placed on a non-CUDA
device under this stack; this constraint was never explicit in the
frozen architecture, only implicitly assumed to be satisfied by `"auto"`
placement — an assumption this hardware disproves.

An explicit device map was independently validated on the target
hardware and confirmed to: load the model successfully; place 224/224
`Linear4bit` modules on CUDA and 0 on CPU; complete a direct forward
pass; produce identical output across two deterministic short
generations; and successfully complete a full generation using the
configured `llm_max_new_tokens` value (validated at 512 tokens).

**Decision:**
`config/settings.py` gains one new field,
`llm_quantized_layers_gpu_resident: bool = False`, describing the
required behavior — all quantized decoder layers remain GPU-resident —
rather than exposing any device-map mechanism through configuration.
`generation.llm_loader.load_generation_model` gains a corresponding
`quantized_layers_gpu_resident: bool = False` parameter; when `True`, it
constructs and applies the single validated explicit device map
internally for `Llama-3.1-8B-Instruct` / `LlamaForCausalLM`, overriding
the passed `device_map` string for that call only. When `False` (the
default), behavior is unchanged from the current frozen implementation.

`generation.llm_loader.load_generation_model` also sets
`llm_int8_enable_fp32_cpu_offload=True` on the `BitsAndBytesConfig`
passed to `from_pretrained` whenever
`quantized_layers_gpu_resident=True`. This is a Transformers-level
validation gate — confirmed by direct source inspection of the installed
Transformers 5.16.1 (`transformers/quantizers/quantizer_bnb_4bit.py`,
`Bnb4BitHfQuantizer.validate_environment`) — required to permit any
CPU/disk entry in a device_map at all under this dependency stack.
**This does not change the project's quantization scheme.** The model
remains 4-bit NF4 throughout; `BitsAndBytesConfig.quantization_method()`
derives quantization mode independently from `load_in_4bit` /
`bnb_4bit_quant_type`, confirmed by direct source inspection to be
unaffected by this flag. The flag's name is a legacy artifact of its
original int8-only implementation, reused for 4-bit CPU-offload
validation without a differently-named 4-bit-specific equivalent
existing in this Transformers version. The explicit device map and this
flag are one coupled configuration, driven by a single parameter, and
must never become independently configurable.

The validated explicit map:

```python
{
    "model.embed_tokens": "cpu",
    "model.layers": "cuda:0",
    "model.norm": "cuda:0",
    "model.rotary_emb": "cuda:0",
    "lm_head": "cpu",
}
```

No change is made to: the inference backend (Transformers +
bitsandbytes), the quantization scheme (4-bit NF4), compute dtype
(float16), the model (`meta-llama/Llama-3.1-8B-Instruct`), deterministic
generation (`do_sample=False`), or any `Generator` / `GeneratedAnswer` /
`PromptBuilder` contract.

**Rationale:**
- The problem is specifically CPU dispatch of quantized `Linear4bit`
  modules, not device placement in general — `model.embed_tokens` and
  `lm_head` are plain, unquantized tensors and can be safely
  CPU-resident without triggering the `QuantState`/meta-tensor failure.
- The Settings field names the architectural guarantee being enforced
  (quantized layers remain GPU-resident), not the placement mechanism
  used to enforce it, keeping the configuration surface narrow and
  intentional rather than a generalized device-map system.
- This is a mid-milestone correction to an underspecified device-
  placement gap in already-frozen M3.3 output, not a new architectural
  decision — the backend, quantization scheme, and model are all
  unchanged, consistent with the ACR/ADR split already established in
  `DECISIONS_LOG.md` and with the precedent of ACR-002 and ACR-003.
- The explicit map is coupled to `LlamaForCausalLM`'s specific module
  names and must not be presented as a universal placement abstraction
  for other model architectures.
- The original implementation of this ACR carried forward the explicit
  device map but omitted this required flag, causing real-hardware
  validation to fail with the identical error the pre-ACR-004
  configuration produced. This was corrected after a read-only source
  inspection of the exact installed Transformers version established,
  with certainty, the mechanism and scope of the required flag.

**Status:**
Approved

**Implementation:**
Implemented

**Affected files:**
- `config/settings.py`
- `generation/llm_loader.py`
- `tests/test_llm_loader.py`
- `tests/test_settings.py`

**Explicitly not modified:**
- `ARCHITECTURE_BASELINE.md` (corrected at next freeze, per §11)
- `DECISIONS_LOG.md`
- `generation/generator.py`
- `generation/context_builder.py`
- `generation/prompts.py`
- `schemas/generation.py`
- `PROJECT_STATE.md`

**Operational constraint (recorded, non-gating):**
The validated configuration completed a full generation at the
configured `llm_max_new_tokens` value (512 tokens validated) with a
tight remaining VRAM margin on the RTX 3050 6GB target. Future increases
to `llm_context_window`, `llm_max_new_tokens`, or the number of
retrieved documents could exhaust this margin and reintroduce placement
failure. This is an operational characteristic of the current
configuration on this hardware, not a defect in this correction, and is
not part of the acceptance criteria below.

**Acceptance criteria:**
- Model constructs successfully with `quantized_layers_gpu_resident=True`.
- 0 of 224 `Linear4bit` modules report a non-CUDA device.
- Direct forward pass succeeds.
- A short deterministic generation (`do_sample=False`) produces identical
  output across two consecutive runs.
- Successful completion of a full generation using the configured
  `llm_max_new_tokens` value (validated at 512 tokens).
- All criteria validated on the actual target environment; not a CI-gated
  requirement.

**Explicitly out of scope:**
- Automatic VRAM detection.
- A generalized or configurable device-map mechanism.
- Support for model architectures other than the validated
  `LlamaForCausalLM` structure.
- Any change to inference backend, model identity, quantization scheme,
  or deterministic generation behavior.

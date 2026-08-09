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

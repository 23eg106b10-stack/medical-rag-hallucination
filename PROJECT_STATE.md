# PROJECT_STATE.md

> **Purpose:** Session bootstrap and live development-state tracker.  
> Unlike [`ARCHITECTURE_BASELINE.md`](./ARCHITECTURE_BASELINE.md) — which is rewritten once per frozen milestone and captures *what the architecture is* — this document captures *where development is right now* and is updated continuously throughout a session.

---

## Current Milestone

**M3.2 Hybrid Retrieval — Runtime Bring-Up**
Status: 🔄 Validation / integration bring-up
Objective: Validate the already-frozen M3.2 Hybrid Retrieval architecture against real, live-built artifacts (corpus, BM25 index, FAISS index) rather than the fakes/mocks used at freeze time. No architectural change is in scope — this is runtime verification of frozen M3.2 work.

**M4 — Claim Extraction (paused)**
Status: ⏸️ Paused pending real retrieval/runtime validation
Objective (on resume): Implement structured claim extraction from `GeneratedAnswer.answer_text`, producing a list of atomic, verifiable claims as input for the hallucination verifier (M5). ADR-M4-001 (verification unit = atomic medical claim) remains accepted and unaffected by the pause.

---

## Last Frozen / Completed Milestones

**ACR-003 — BM25 Artifact Filename & Configuration Correction**  
Implementation: Complete | Validation: Complete | Commit: Pending  
Details: Corrected `bm25_index_filename` default to `bm25_index.pkl` in `config/settings.py` and updated `index/build_bm25.py` to consume `settings.bm25_index_filename`. Validation clean (125 tests passing, ruff, black, compileall). Not yet committed — see Repository Health below.

**M3.1.3 Live Execution — FAISS Dense Index Building**  
Completed at: 2026-08-06  
Output: `data/indexes/faiss_index.bin` (171,858 vectors, dim=768) + `data/indexes/embedding_metadata.json` + `data/indexes/faiss_pmids.json`

**M3.1.2 Live Execution — BM25 Lexical Index Building**  
Completed at: 2026-08-06  
Output: `data/indexes/bm25_index.pkl` (171,858 documents indexed) + `data/indexes/bm25_metadata.json`

**M3.1.1 Live Execution — Corpus Construction**  
Completed at: 2026-08-06  
Output: `data/corpus/corpus.jsonl` (171,858 documents, 295 MB) + `data/corpus/corpus.meta.json`  
Corpus Version: `1.0`  
Checksum: `b96c906133f72e60b62b9efe590b1aae083aef05a747eed46f154fc88427d35e`

**M3.3 — LLM Generation**  
Git tag: `m3.3-frozen`  
Frozen at: 2026-08-02

---

## Milestone History

| Tag | Milestone | Commit / Status |
|---|---|---|
| *(untagged)* | ACR-003 BM25 Config Correction | 🔄 Implementation + validation complete; commit pending |
| *(untagged)* | M3.1.3 FAISS Dense Index Build | ✅ Complete (171,858 vectors, dim=768, `faiss_index.bin`) |
| *(untagged)* | M3.1.2 BM25 Lexical Index Build | ✅ Complete (171,858 documents, `bm25_index.pkl`) |
| *(untagged)* | M3.1.1 Live Corpus Build | ✅ Complete (171,858 documents, Checksum `b96c9061...`) |
| `m3.3-frozen` | M3.3 — LLM Generation | `ad70a1a` |
| `m3.2-frozen` | M3.2 — Hybrid Retrieval | `87249cf` |
| *(untagged)* | M2 — Infrastructure Foundation | `a5eb8ed` |
| *(untagged)* | M1 — Initial Scaffold | `3314ac0` |

---

## Validation Status

| Layer | Status | Notes |
|---|---|---|
| `config/` (settings, exceptions) | ✅ Passing | `test_config.py`, `test_settings.py`, `test_exceptions.py` |
| `schemas/` | ✅ Passing | `test_schemas.py` |
| `index/` BM25 pipeline & build | ✅ Passing | `test_bm25_pipeline.py` (including ACR-003 regression tests), `test_bm25_tokenizer.py` |
| `index/` FAISS pipeline | ✅ Passing | `test_faiss_pipeline.py`, `test_embedding_generator.py` |
| `index/` FAISS pmids (ACR-001) | ✅ Passing | `tests/index/test_faiss_pipeline_acr001.py` |
| `index/` Corpus Construction (M3.1.1) | ✅ Verified Live | Successfully generated `corpus.jsonl` (171,858 docs) |
| `retrieval/` dense retriever | ✅ Passing | `tests/retrieval/test_dense.py` |
| `generation/` LLM loader | ✅ Passing | `tests/generation/test_llm_loader.py` |
| `generation/` context builder | ✅ Passing | `tests/generation/test_context_builder.py` |
| `generation/` generator | ✅ Passing | `tests/generator/test_generator.py` |
| `verification/` | ⏸️ Paused | ADR-M4-001 accepted; claim extraction design work paused pending M3.2 runtime bring-up |
| `evaluation/` | ⬜ Not yet built | Stubs only |
| `app/` | ⬜ Not yet built | Stub only |

---

## Repository Health

| Check | Status |
|---|---|
| Working tree | ⚠️ **Not clean** — pending commit (see below) |
| Lint (ruff) | ✅ Passing (`ruff check .` clean) |
| Formatting (black) | ✅ Passing (`black --check .` clean) |
| Test suite | ✅ Passing (125/125 tests passing) |
| Compilation | ✅ Passing (`python -m compileall .` clean) |
| Frozen artifacts | ✅ **Corpus & Indexes Generated!** `corpus.jsonl` (171,858 records), `bm25_index.pkl` (171,858 docs), and `faiss_index.bin` (171,858 vectors, dim=768) present. |

**Pending working-tree changes (uncommitted):**

*Intentional — ACR-003 implementation:*
- `ARCHITECTURE_CHANGE_REQUESTS.md`
- `config/settings.py`
- `index/build_bm25.py`
- `tests/test_bm25_pipeline.py`

*Untracked:*
- `PROJECT_STATE.md`
- `UNIFIED_ARCHITECTURE.md`
- `schemas/verification.py`

None of the above have been committed. Repository cleanup/commit is outstanding before the working tree can be described as clean.

---

## Open ACRs

| ID | Title | Status |
|---|---|---|
| ACR-003 | BM25 Artifact Filename & Configuration Correction | 🔄 Implementation: Complete · Validation: Complete · Commit: Pending |

ACR-001 and ACR-002 remain closed.

---

## Open ADRs

| ID | Title | Status |
|---|---|---|
| ADR-001 | L2 Normalization for FAISS `IndexFlatIP` | ✅ Accepted |
| ADR-M3.3-001 | Inference Engine = Transformers + bitsandbytes | ✅ Accepted |
| ADR-M3.3-002 | Prompt Template Ownership | ✅ Accepted |
| ADR-M3.3-003 | Reserved — never assigned | Reserved |
| ADR-M3.3-004 | Deterministic Generation (`do_sample=False`) | ✅ Accepted |
| ADR-M4-001 | Verification Unit = Atomic Medical Claim | ✅ Accepted |

---

## Resolved & Active Known Limitations

### Resolved in recent sessions
1. **Corpus construction failure on live NCBI data resolved.**
   - Real corpus generation completed successfully (171,858 validated documents, SHA-256 checksum recorded).
2. **BM25 and FAISS indexes generated on live corpus.**
   - BM25 lexical index built on 171,858 documents (`data/indexes/bm25_index.pkl`).
   - FAISS dense index built on 171,858 vectors (dim=768, `data/indexes/faiss_index.bin`).
3. **ACR-003 BM25 configuration correction implemented and validated (commit pending).**
   - Default filename corrected to `bm25_index.pkl`, build path wired to `settings.bm25_index_filename`, and regression test suite added. Validation clean (125/125 tests, ruff, black, compileall). Not yet committed — see Repository Health.

### Active limitations entering M3.2 runtime bring-up
1. **No end-to-end integration path yet.** `HybridRetriever`, `Generator`, and `Verifier` require a runnable wiring/pipeline orchestrator.
2. **`PromptBuilder` context budget is prompt-only.** Token budgeting does not reserve headroom for `llm_max_new_tokens`.
3. **No real model weights loaded yet.** Generation and dense retrieval have been tested using fakes; live model runs require GPU environment and model downloads.
4. **M4 (Claim Extraction) is paused**, not progressing, pending completion of M3.2 runtime/integration validation against real corpus and index artifacts.

---

*Update this file at the start and end of every session, and immediately after any freeze.*

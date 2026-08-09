# Retrieval-Augmented Medical QA with Hallucination Detection
## Unified Architecture Specification — v2.0

*B.Tech AI Mini Project · Department of Artificial Intelligence, Anurag University · Supervisor: Dr. K. Basava Raju*

**Status:** Reconciliation and modernization pass over Architecture v1.0 (Blueprint3, the corrected/authoritative revision) and the M3.1.1 Corpus Construction component spec (Blueprint3_v2). Supersedes both. Does **not** supersede anything already frozen and implemented through M3.3 (`ARCHITECTURE_BASELINE.md`) — see the Reconciliation Notes below for what this document changes versus what it merely documents.

---

## 0. Reconciliation Notes (read this first)

This section exists because a from-scratch "best possible version" rewrite, done blindly, would contradict months of already-frozen, already-implemented, already-tested work. Every change below is tagged by what it actually costs:

- 🟢 **Documentation-only** — corrects the document to match reality; zero implementation impact.
- 🟡 **Open decision** — a real gap or question with no implementation cost yet, but a decision is owed.
- 🔴 **Requires an ACR** — would mean reopening and modifying already-frozen, already-implemented architecture (M3.1.1–M3.3).

| Finding | Tier | Resolution in this document |
|---|---|---|
| Blueprint1 and Blueprint2 are content-identical, superseded drafts carrying an unverified verifier checkpoint | 🟢 | Discarded as sources; Blueprint3's verified checkpoint and 20/80 split used throughout |
| §2 module ownership assigns Query Processing/Retrieval to Thaluka and Claim Extraction/Verification/Confidence to Deshagoni Sai Kiran, who has left the team | 🟢 | Ownership table corrected to current reality (§2) |
| Platform fixed as "Google Colab (free-tier GPU)" | 🟢 | Corrected to local Windows / VS Code, matching the actual environment change |
| Minimum-similarity gate + `INSUFFICIENT_EVIDENCE` early exit, specified as part of Retrieval's frozen contract | 🔴 | **Not implemented.** `HybridScoredDocument`/`FusionResult` (M3.2, frozen) carry no gate status field. Kept in this spec as intended design (§5.3), flagged as an implementation gap requiring either an ACR against M3.2 or a formal descope decision — not resolved unilaterally here. |
| Mandatory inline citation markers `[1][2]` + exact refusal token in generated answers | 🔴 | **Not implemented.** `GeneratedAnswer`/`generation/prompts.py` (M3.3, frozen) have neither. Same treatment as above (§6). |
| Corpus never run against live NCBI data; `data/corpus/`, `data/indexes/` still empty | 🟡 | Elevated from a buried risk-table row to its own top-level risk (§13) — carried unresolved across three frozen milestones |
| 5-baseline (this document) vs. 6-baseline (review slides) evaluation protocol mismatch | 🟡 | Documented explicitly in §10, not silently picked one way |
| MedMCQA referenced throughout (§5.1, §9, folder structure, baselines, roadmap) but absent from the actual implemented corpus pipeline (PubMedQA + NCBI MeSH only) | 🟡 | Flagged in §5.1 and §9.3 — likely superseded scope, not silently kept or silently cut |

Everything else below is genuine technical modernization, tagged the same way, embedded inline where it applies rather than listed separately — a reviewer reading straight through sees the recommendation and its cost together.

---

## 1. Purpose

A clinical question is answered only from evidence the system itself retrieved, and every claim in that answer is checked against the specific evidence cited for it before anything is shown to the user. Retrieval, not verification, is the first line of defense against unsupported answers — no downstream stage is allowed to "undo" a decision an upstream stage should have made.

## 2. System Architecture and Module Ownership

```
Clinical Question
    → [1] Query Processing
    → [2] Evidence Retrieval (BM25 + MedCPT dense, RRF fusion)
        → min-similarity gate → INSUFFICIENT_EVIDENCE (early exit) or continue   🔴 not yet implemented
    → [3] LLM Generation (evidence-grounded, inline citation markers [1][2]…)
        → hard refusal token if ungrounded                                       🔴 not yet implemented
    → [4] Claim Extraction (sentence-level split of the generated answer)        ⬜ planned, M3.4
    → [5] Evidence Verification (per-claim NLI vs. its cited passage only)       ⬜ planned, M5
        → SUPPORTED / CONTRADICTED / UNVERIFIABLE per claim
    → [6] Confidence Scoring (calibrated NLI softmax, per claim + overall)       ⬜ planned, M6
    → [7] Streamlit UI (answer, citations, per-claim verdicts, confidence)       ⬜ planned, M8
```

| Module | Responsibility | Owner | Status |
|---|---|---|---|
| Query Processing | Clean/normalize the input question; no retrieval logic lives here | Bingi Meghamsh | Not yet built |
| Evidence Retrieval | BM25 + MedCPT dense search over the frozen PubMed index; RRF fusion; top-5 return; early-exit gate on low similarity | Bingi Meghamsh | **Implemented and frozen (M3.2)** — gate not implemented, see §5.3 |
| LLM Generation | Prompt an 8B open instruct model with question + numbered evidence; citation markers and refusal token per original design | Bingi Meghamsh | **Implemented and frozen (M3.3)** — citation/refusal not implemented, see §6 |
| Claim Extraction | Sentence-split the generated answer into atomic claims; map each claim to the citation marker(s) it uses | Bingi Meghamsh | Not yet built (M3.4, current focus) |
| Evidence Verification | Per-claim NLI (Entailment/Neutral/Contradiction) against only the cited passage(s) for that claim | Bingi Meghamsh | Not yet built (M5) |
| Confidence Scoring | Temperature-scale the verifier's softmax output on a held-out calibration split; expose per-claim and overall confidence | Bingi Meghamsh | Not yet built (M6) |
| Evaluation Harness | Run all baselines, compute accuracy/F1/detection metrics/ECE, produce the comparison table | Bingi Meghamsh (technical); Thaluka Jyoshna (annotation, analysis support) | Not yet built (M7) |
| UI (Streamlit) | Render the fixed output contract: answer, citations, per-claim verdict, confidence. No new logic | Bingi Meghamsh | Not yet built (M8) |

*Ownership corrected to current team reality — see Reconciliation Notes. Thaluka Jyoshna's contribution (literature review, evaluation-dataset hand-annotation, documentation, presentation/defense preparation) is real and load-bearing but non-technical; this table reflects that honestly rather than either fabricating a technical role or omitting the contribution.*

## 3. Data Flow Contracts

| From → To | Payload |
|---|---|
| Query Processing → Retrieval | `{ clean_question: str }` |
| Retrieval → Generation | `{ passages: [HybridScoredDocument], gate_status: SUFFICIENT \| INSUFFICIENT }` 🔴 |
| Generation → Claim Extraction | `{ answer_text: str (with inline [n] markers), refused: bool }` 🔴 |
| Claim Extraction → Verification | `{ claims: [{claim_text, cited_passage_ids: [n]}] }` |
| Verification → Confidence | `{ claims: [{claim_text, verdict, raw_nli_scores}] }` |
| Confidence → UI | `{ claims: [{claim_text, verdict, calibrated_confidence}], overall_confidence, citations }` |

🔴-marked rows describe intended contracts not yet reflected in the actual frozen `GeneratedAnswer` / `HybridScoredDocument` schemas — see §0.

## 4. Technology Stack

| Layer | Choice | Rationale |
|---|---|---|
| Sparse retrieval | BM25 (`rank_bm25`) | Cheap, competitive baseline on biomedical text. **Implemented (M3.1.2).** |
| Dense retrieval | MedCPT (query + article encoder pair) | Purpose-built for query→PubMed-abstract retrieval. **Implemented (M3.1.3).** |
| Fusion | Reciprocal Rank Fusion, no learned reranker | Cross-encoder reranking rejected — marginal gain at this corpus scale. **Implemented (M3.2).** 🟡 *Modernization note: this call was defensible in 2024–25; at ~25k documents it remains defensible today too — a cross-encoder reranker adds real latency and a tuning surface for a single-semester project with no evidence the current top-5 quality is the bottleneck. Not recommending a change.* |
| Vector store | FAISS, flat index (`IndexFlatIP`) | Corpus small enough that approximate search buys nothing. **Implemented (ADR-001).** |
| Generation LLM | Llama-3.1-8B-Instruct, 4-bit (bitsandbytes) | **Implemented (ADR-M3.3-001).** 🟡 *Modernization note: this remains a reasonable choice for a locally-hosted, gated-repo-accessible, instruction-following 8B model. Newer small instruct models exist, but replacing a frozen, tested, working choice for marginal quality gain — with no retraining/re-evaluation budget implied — is not worth reopening. Not recommending a change.* |
| Claim extraction | Sentence splitting (spaCy/nltk) | LLM-based decomposition rejected — doubles inference cost, adds its own hallucination risk. Not yet built (M3.4). |
| Verifier | `pritamdeka/PubMedBERT-MNLI-MedNLI` (fixed, verified) | 86.67% MedNLI accuracy; non-sequential `id2label` — must be set explicitly. Not yet built (M5). |
| Confidence | Temperature-scaled softmax from the fixed verifier, 20/80 calibration split | Not yet built (M6). |
| UI | Streamlit | Demo-layer decision, not reopened. Not yet built (M8). |
| **Platform** | **Local (Windows, VS Code, Python venv)** | 🟢 *Corrected — Blueprint3 fixed this as Google Colab. That constraint no longer applies; local execution is the actual current environment.* |

## 5. Corpus Construction (M3.1.1 — implemented and frozen)

*Merged from the separate M3.1.1 component specification (Blueprint3_v2). That document's status line read "Proposed (Not Implemented)" — it has since been implemented and frozen; this section reflects that.*

**Responsibilities (exactly five):** obtain PubMed abstracts; normalize metadata; deduplicate by PMID; validate documents; export a frozen, versioned corpus. Nothing else — no BM25, no FAISS, no embeddings, no retrieval, no scoring, no generation.

### 5.1 Sources
- **Source A — PubMedQA:** guarantees corpus coverage for PubMedQA evaluation. Output: PMID list.
- **Source B — NCBI E-utilities:** broader coverage via a fixed, version-controlled MeSH query specification (`queries.yaml`), consumed but never modified by this component.

🟡 *Both this document and Blueprint3 reference MedMCQA subject-area coverage for Source B's query design. The actual implemented `corpus_sources.py`/`corpus_pipeline.py` sources only PubMedQA + NCBI MeSH — no MedMCQA integration exists anywhere in the built pipeline. Flagging rather than resolving: either MedMCQA was deliberately dropped from scope at some point without this being recorded, or it's still intended and hasn't been built. Confirm before this document is treated as final.*

### 5.2 Deduplication and validation
PMID is the sole dedup key — no fuzzy or title/abstract similarity matching. On collision, Source B (direct NCBI) is authoritative over Source A (PubMedQA's bundled copy), since NCBI is the system of record. Every document requires a non-empty PMID, non-empty title, and a "meaningful" abstract (non-placeholder, above a minimum token length) — the exact detection mechanism is an implementation decision, not frozen here. Invalid records are skipped; no automatic repair.

### 5.3 Retrieval query-time flow (M3.2)
BM25 top-20 and MedCPT-dense top-20, computed independently, merged via RRF (`k=60`, frozen per ADR), truncated to top-5.

🔴 **Gap:** the original design specifies a minimum-similarity gate on the fused top-5 — if nothing clears threshold, the pipeline exits early with `INSUFFICIENT_EVIDENCE` rather than letting generation attempt an answer on weak evidence. This is not implemented in the frozen `HybridRetriever`. This is arguably the single highest-value missing piece relative to the project's stated thesis (hallucination *prevention*, not just detection after the fact) — recommend prioritizing a decision on this before M3.4 architecture review, via ACR against M3.2.

**Recall@k reporting requirement (unchanged, still correct):** must be reported separately for the PubMedQA-sourced portion of the corpus versus the MeSH-pulled portion — the PubMedQA-sourced portion is near-closed-book by construction, so an averaged number overstates genuine open-retrieval difficulty.

## 6. Generation Pipeline (M3.3 — implemented and frozen)

Fixed prompt structure: System → Evidence → Question → Answer (per ADR-M3.3-002/ACR-002). `PromptBuilder` enforces a hard token budget and raises rather than silently truncating evidence (ACR-002) — this no-silent-truncation principle is a genuine strength worth calling out; it's a stronger hallucination-prevention property than either original blueprint specified.

🔴 **Gap:** the original design requires mandatory inline citation markers (`[1][2]…`) so verification can check each claim against the specific passage it drew from, plus an exact, programmatically-detectable refusal token (`INSUFFICIENT EVIDENCE`) rather than free-text hedging. Neither exists in `generation/prompts.py` or `GeneratedAnswer` as built. Without citation markers, Claim Extraction (M3.4) has no clean way to map a claim back to a specific source passage — it would have to verify against the full top-5 context pool rather than the passage that actually grounds each claim, which is a real quality regression from the original design, not a cosmetic gap. Recommend resolving this at M3.4 architecture review, since M3.4 depends on it directly.

Model revision/commit hash pinning for reproducibility: not yet recorded anywhere. 🟡 Add to `config/settings.py` or `DECISIONS_LOG.md` before Review-2.

## 7. Verification Pipeline (planned, M5)

Per-claim NLI against only the cited passage(s) for that claim — not the full retrieved set. Three-way output: SUPPORTED / CONTRADICTED / UNVERIFIABLE, with UNVERIFIABLE surfaced distinctly rather than collapsed into SUPPORTED. Verifier is fixed, no implementation-time model selection: `pritamdeka/PubMedBERT-MNLI-MedNLI`. Its `id2label` mapping is non-sequential (`{1: entailment, 0: contradiction, 2: neutral}`) — must be set explicitly in code, with a unit test asserting known entailment/contradiction examples classify correctly, before wiring it into the pipeline.

*This section depends on §6's citation-marker gap being resolved first — per-claim verification against "only the cited passage" requires claims to actually carry citation references.*

## 8. Confidence Scoring (planned, M6)

Temperature-scaled softmax from the fixed verifier's own output — no separate model, no extra inference calls. Calibration (20 items) and test reporting (80 items) use disjoint data from the same ~100-item hand-annotated set; overall answer confidence is the **minimum** calibrated confidence across all verified claims (a single weak claim caps the whole answer), fixed — no averaging alternative.

## 9. Evaluation Protocol

### 9.1 Annotation
~100 generated answers, one primary annotator (Thaluka Jyoshna) labeling every claim SUPPORTED/CONTRADICTED/UNVERIFIABLE; a second annotator independently labels a 20-answer overlap subset for Cohen's kappa as a label-quality check. 20/80 calibration/test split, no overlap.

**Limitation to state explicitly, not hide:** annotators are B.Tech students, not clinicians. The project guide spot-checking a slice of disagreements does not fully resolve the construct-validity gap.

### 9.2 Metrics
Answer accuracy (exact-match/F1 for PubMedQA yes/no/maybe; accuracy for MedMCQA if retained — see §9.3); hallucination detection precision/recall/F1 against the annotated 80-item held-out set only; retrieval recall@5/MRR reported separately by corpus segment (§5.1); calibration via ECE.

### 9.3 MedMCQA — unresolved scope question
🟡 Same flag as §5.1: MedMCQA appears throughout the original evaluation protocol (forced free-text justification for claim extraction, accuracy-only fallback if unreliable) but doesn't exist in the implemented corpus. This decision — was MedMCQA dropped, or still pending — needs to be made and recorded, not rediscovered during M3.4 or M7 planning.

## 10. Baselines

| # | Baseline | Purpose |
|---|---|---|
| 1 | Baseline LLM, no retrieval | Hallucination-rate floor with zero grounding |
| 2 | Closed-book RAG (PubMedQA gold context) | Isolates generation quality from retrieval quality |
| 3 | Standard RAG, no verification | Isolates retrieval's contribution alone |
| 4 | RAG + rule-based lexical overlap | Weak verification baseline; motivates why NLI is needed |
| 5 | RAG + NLI verification (proposed system) | Full system |

🟡 **This document specifies 5 baselines.** Earlier project tracking flagged a discrepancy with review slides showing 6 baselines, never reconciled. Recorded here explicitly rather than silently picking one — resolve before `evaluation/` (M7) is built, since the evaluation harness's shape depends on the answer.

## 11. Repository Structure

*Reflects the actual current tree, not the original blueprint's proposed layout (which omitted `schemas/`, `utils/`, `scripts/`, and included the now-deleted `retrieval/config.yaml`).*

```
medical-rag-hallucination/
├── config/            # Pydantic Settings — sole env-var access point
├── data/              # corpus/, pubmedqa/, annotations/ — corpus/ and indexes/ currently empty, see §13
├── index/             # build_corpus.py, build_bm25.py, build_faiss.py
├── schemas/           # corpus, retrieval, generation, bm25, embedding — inter-module contracts
├── retrieval/          # BM25 + dense + RRF fusion (frozen, M3.2)
├── generation/         # PromptBuilder, Generator, llm_loader (frozen, M3.3)
├── verification/        # claim_extraction.py, verifier.py, confidence.py — stubs
├── evaluation/           # run_baselines.py, metrics.py — stubs
├── app/                  # streamlit_app.py — stub
├── utils/                # centralized logging
├── scripts/              # reset_indexes.py
├── tests/
├── docs/                 # ARCHITECTURE_BASELINE.md, PROJECT_STATE.md, DECISIONS_LOG.md, ARCHITECTURE_CHANGE_REQUESTS.md
└── README.md
```

## 12. Development Roadmap

Reflects actual milestone numbering (`M1`–`M8`, matching `PROJECT_STATE.md`), not the original blueprint's week-based plan, which predates the M3.x restructuring and the loss of a team member.

| Milestone | Status |
|---|---|
| M1 — Initial scaffold | ✅ |
| M2 — Infrastructure foundation | ✅ |
| M3.1.1 — Corpus construction | ✅ Frozen — **but never run against live NCBI data**, see §13 |
| M3.1.2 — BM25 index | ✅ Frozen |
| M3.1.3 — FAISS index (+ ACR-001) | ✅ Frozen |
| M3.2 — Hybrid retrieval | ✅ Frozen — similarity gate not implemented, see §5.3 |
| M3.3 — LLM generation | 🔶 Implemented, freeze not formally declared — citation/refusal not implemented, see §6 |
| M3.4 — Claim extraction | ⬜ Current focus |
| M5 — Verification | ⬜ |
| M6 — Confidence scoring | ⬜ |
| M7 — Evaluation & baselines | ⬜ |
| M8 — Streamlit app | ⬜ |

## 13. Risks and Limitations

**Elevated to top priority — carried unresolved across three frozen milestones:**
1. **Corpus has never been generated against live NCBI data.** `data/corpus/` and `data/indexes/` are empty. Every downstream milestone (BM25, FAISS, Hybrid Retrieval, Generation) has been built and tested exclusively against synthetic/injected data. The ~25k-abstract target is architecturally frozen but empirically unconfirmed.
2. **Similarity gate and citation/refusal mechanisms specified in the original architecture are not implemented** — see §5.3, §6. This is a gap between what's documented as the system's core hallucination-prevention design and what actually exists in frozen code.

**Carried from the original blueprint, still applicable:**
3. Annotators are non-clinicians — ground truth reflects student judgment, named as a limitation, not hidden.
4. Calibration/test contamination risk if the same annotated sample were reused for both — mitigated by the fixed 20/80 split with no overlap.
5. Verifier's non-sequential `id2label` mapping could be silently misassigned if implemented from habit instead of the model card — mitigate with an explicit unit test before wiring the verifier in.
6. 4-bit quantized inference is not always bit-identical run to run — pin exact model revisions for reproducibility.

**New, surfaced by this reconciliation:**
7. 5-vs-6 baseline evaluation protocol discrepancy, unresolved (§10).
8. MedMCQA scope ambiguity — referenced throughout planning documents, absent from implementation (§5.1, §9.3).
9. No real model has been loaded end-to-end — `llm_loader.py` and `retrieval/dense.py` both depend on real model weights; all current tests use injected fakes.

## 14. Future Work

- Extend the PubMed index beyond ~25k abstracts if coverage gaps are found post-submission — contingent on first actually running corpus construction against live data (§13).
- Replace sentence-split claim extraction with LLM-based decomposition if compute budget allows in a follow-on project.
- Clinician-reviewed annotation set, addressing the construct-validity limitation named in §13.
- Full-text article retrieval (beyond abstracts), if chunking strategy is revisited.
- Resolve the similarity-gate and citation-marker implementation gaps (§5.3, §6) — the highest-leverage future work item, since it closes the gap between documented and actual hallucination-prevention behavior.

---

## Change Control

This document reconciles Architecture v1.0 (Blueprint3) and the M3.1.1 component specification (Blueprint3_v2) against actual project history through M3.3. It does not itself freeze or unfreeze anything — 🔴-tagged items require a formal ACR before any code changes; 🟡-tagged items require an explicit decision recorded in `DECISIONS_LOG.md`; 🟢-tagged items are already true and require no further action beyond this document existing.

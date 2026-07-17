# Milestone 3 — Component M3.1.1

## Corpus Construction Architecture Specification

**Status:** Frozen (Approved by Chief Architect)  
**Revision:** 2 (incorporates accepted architecture review changes)  
**Project:** Medical RAG Hallucination Detection (Architecture v1.0)

---

## Purpose

Construct a reproducible, frozen biomedical corpus that serves as the sole retrieval source for the entire project.

This component is not responsible for retrieval. It only creates the corpus. Once generated, the corpus becomes immutable.

---

## Responsibilities

This component has exactly five responsibilities.

1. Obtain PubMed abstracts.
2. Normalize retrieved metadata.
3. Remove duplicate papers.
4. Validate documents against semantic and structural requirements.
5. Export a frozen, versioned corpus.

**Nothing else.**

Specifically it will **not**:

- build BM25
- build FAISS
- compute embeddings
- rerank
- retrieve
- score
- evaluate
- generate answers

---

## Inputs

### Source A — PubMedQA

**Purpose:** Guarantee corpus coverage for PubMedQA evaluation.  
**Output:** List of PMIDs.

### Source B — NCBI E-utilities Search

**Purpose:** Broader biomedical coverage using MeSH-based searches matching MedMCQA subject areas.  
**Output:** PMIDs + abstracts.

**Query determinism:** The MeSH query set used for Source B is not left to implementer judgment. It is defined in a fixed, version-controlled query specification (e.g. `queries.yaml`) that is itself a versioned artifact of the project, external to this document. This component consumes that specification as input; it does not define or infer MeSH terms at runtime. Any change to the query specification is a new version of the specification and, transitively, invalidates the reproducibility guarantee of any corpus generated from the prior version.

This exactly follows Architecture v1.0.

---

## Query Specification Contract

`queries.yaml` contains the fixed MeSH-based search queries used for corpus construction.

- This component consumes the file but never modifies it.
- The file is version-controlled.
- Any modification to this file constitutes a new corpus specification version and therefore changes the resulting corpus.

The architecture intentionally does not specify the internal YAML structure or parsing logic, since those belong to implementation.

---

## Output

A single frozen corpus, accompanied by a corpus metadata record.

### Corpus document — contains only

- PMID
- Title
- Abstract

No embeddings. No vectors. No metadata beyond what retrieval requires, at the document level.

### Corpus metadata record — accompanies the corpus

To prevent silent corpus drift and allow downstream components to verify they are indexing the exact corpus used during evaluation, the frozen output is accompanied by a metadata record containing:

- corpus version
- generation timestamp
- document count
- SHA-256 checksum of `corpus.jsonl`

This metadata record is itself part of the frozen output. It is generated, not hand-maintained.

---

## Canonical Document Schema

Internally every document should have:

```python
@dataclass
class CorpusDocument:
    pmid: str
    title: str
    abstract: str
```

Nothing else. No score. No embedding. No ranking fields. Those belong to later milestones.

---

## Pipeline

```
PubMedQA
        \
         \
          ---> PMID Collection
         /
NCBI Search (per queries.yaml)
          |
          v
Download Metadata
          |
          v
Normalize
          |
          v
Deduplicate (PMID)
          |
          v
Validate
          |
          v
Write Corpus + Metadata Record
          |
          v
Frozen Corpus
```

---

## Deduplication Rule

**Primary key:** PMID

If two records share the same PMID, keep exactly one. Discard the other in full — its title and abstract are not merged or compared field-by-field with the retained record.

**Source precedence:** When the same PMID is returned by both Source A and Source B, the record retrieved directly from NCBI E-utilities (Source B) is authoritative and is the one retained. NCBI is the origin system of record for PubMed metadata; PubMedQA's bundled copy is treated as a coverage signal ("this PMID must be present"), not as a metadata source of truth.

No fuzzy matching. No title similarity. No abstract similarity. PMID is authoritative.

---

## Validation Rules

Every document must satisfy:

- **PMID** — non-empty
- **Title** — non-empty
- **Abstract** — non-empty, and meaningful

"Meaningful" is an architectural principle, not a numeric threshold: an abstract must represent actual scientific content, not an empty placeholder, a non-content string (e.g. "No abstract available"), or degenerate text. The concrete detection mechanism — minimum token length, placeholder-string matching, language checks, or some combination — is an implementation and configuration decision made at build time, not frozen into this architecture.

Invalid records are skipped. Do not attempt automatic repair.

---

## Folder Structure

Architecture v1.0 already defines the long-term layout. For this component, only the corpus-related portion is in scope.

```
data/
    corpus/
        corpus.jsonl
        corpus.meta.json
```

`corpus.meta.json` holds the corpus metadata record (version, timestamp, document count, checksum) described above. Future components may read this file. This component never reads indexes.

---

## Public Contract

### Input

- PubMedQA
- NCBI
- `queries.yaml` (versioned MeSH query specification)

### Output

- `corpus.jsonl`
- `corpus.meta.json`

### Consumers

Future:

- BM25 Builder
- Dense Index Builder
- Evaluation

No other component should depend on internal implementation details.

---

## Error Handling

Expected failures:

- Missing PMID → Skip
- Missing or non-meaningful abstract → Skip
- Network interruption → Raise an exception and stop.

Partial corpus generation is not considered successful. Corpus generation either completes and produces a validated `corpus.jsonl` plus `corpus.meta.json`, or it fails and produces no output. The retry, backoff, and resumability strategy used to reduce the likelihood of reaching the failure path is an implementation detail and is intentionally not specified here.

---

## Logging

Use only:

```python
logger = logging.getLogger(__name__)
```

Report:

```
Downloading PMIDs...
Downloading abstracts...
Normalizing corpus...
Removing duplicates...
Validating corpus...
Writing corpus...
Corpus complete.
Documents written: XXXXX
Corpus version: XXXXX
Checksum: XXXXX
```

Never configure logging. Infrastructure already owns logging.

---

## Configuration

This component should receive configuration only through the existing frozen settings mechanism.

Hard-coded values are limited to invariant validation rules (e.g. "PMID must be non-empty", "abstract must be meaningful"). API endpoints, file paths, the MeSH query specification location, and output locations belong in configuration, preserving backward compatibility with the frozen infrastructure.

---

## Dependencies

**Consumes:**
- Existing logging infrastructure (`utils/logging.py`)
- Existing settings infrastructure (`config/settings.py`)
- Existing schemas where applicable (`schemas/`)
- Versioned MeSH query specification (`queries.yaml`)

**Produces:**
- Frozen, versioned corpus (`corpus.jsonl` + `corpus.meta.json`)

**Consumed later by:**
- BM25 Index Builder
- FAISS Index Builder
- Retrieval Pipeline

---

## Non-Goals

This component intentionally does **not**:

- Build BM25 indexes
- Build FAISS indexes
- Generate MedCPT embeddings
- Perform retrieval
- Calculate similarity scores
- Apply Reciprocal Rank Fusion
- Implement early-exit similarity gates
- Load LLMs
- Perform verification
- Compute evaluation metrics
- Define retry/backoff/resumability strategy (implementation concern)

Those belong to later Milestone 3 components, or to the implementation layer.
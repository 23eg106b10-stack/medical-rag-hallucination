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

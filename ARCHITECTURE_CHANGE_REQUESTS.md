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

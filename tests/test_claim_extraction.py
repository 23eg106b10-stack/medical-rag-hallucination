"""Unit tests for M4 claim extraction (verification.claim_extraction).

Covers all frozen contracts defined in ADR-M4-001, ADR-M4-002, and ADR-M4-003:
1. Empty/whitespace-only raw output -> ClaimExtractionError
2. One valid claim
3. Multiple claims
4. Numbered lines
5. Bullet-prefixed lines
6. PMID/reference marker stripping
7. Incomplete fragment filtering
8. Balanced/unbalanced bracket/parenthesis behavior
9. Pure process-commentary filtering
10. Substantive uncertainty retained
11. Exact normalized deduplication
12. Non-semantic near-duplicates are NOT deduplicated
13. Zero valid claims after filtering is allowed
14. Exactly 35 unique valid claims is allowed
15. 36 unique valid claims raises ClaimExtractionError
16. No truncation occurs
17. Single claim without trailing newline is accepted
18. ExtractedClaim schema remains exactly claim_text: str
19. Deterministic generation configuration uses do_sample=False
20. Existing generation/model infrastructure is reused rather than duplicated
"""

from __future__ import annotations

import pytest

from schemas.generation import GeneratedAnswer
from schemas.verification import ExtractedClaim
from verification.claim_extraction import (
    MAX_CLAIMS,
    ClaimExtractionError,
    ClaimExtractor,
    build_claim_extraction_prompt,
    make_extraction_generate_fn,
    parse_claims,
)
from verification.claim_extraction_prompts import (
    CLAIM_EXTRACTION_SYSTEM_PROMPT,
    CLAIM_EXTRACTION_USER_TEMPLATE,
)


# 1. Empty/whitespace-only raw output -> ClaimExtractionError
def test_empty_or_whitespace_output_raises():
    with pytest.raises(ClaimExtractionError, match="empty or whitespace"):
        parse_claims("")

    with pytest.raises(ClaimExtractionError, match="empty or whitespace"):
        parse_claims("   \n\t   \n  ")


# 2. One valid claim
def test_one_valid_claim():
    raw = "Metformin lowers blood glucose levels in type 2 diabetes."
    claims = parse_claims(raw)
    assert len(claims) == 1
    assert isinstance(claims[0], ExtractedClaim)
    assert claims[0].claim_text == "Metformin lowers blood glucose levels in type 2 diabetes."


# 3. Multiple claims
def test_multiple_valid_claims():
    raw = (
        "Aspirin reduces platelet aggregation.\n"
        "Lisinopril is an ACE inhibitor.\n"
        "Atorvastatin inhibits HMG-CoA reductase.\n"
    )
    claims = parse_claims(raw)
    assert len(claims) == 3
    assert [c.claim_text for c in claims] == [
        "Aspirin reduces platelet aggregation.",
        "Lisinopril is an ACE inhibitor.",
        "Atorvastatin inhibits HMG-CoA reductase.",
    ]


# 4. Numbered lines
def test_numbered_lines_stripping():
    raw = (
        "1. Metformin is used for diabetes.\n"
        "2) Insulin stimulates glucose uptake.\n"
        "(3) Glipizide promotes insulin secretion.\n"
        "[4] Empagliflozin is an SGLT2 inhibitor.\n"
    )
    claims = parse_claims(raw)
    assert len(claims) == 4
    assert [c.claim_text for c in claims] == [
        "Metformin is used for diabetes.",
        "Insulin stimulates glucose uptake.",
        "Glipizide promotes insulin secretion.",
        "Empagliflozin is an SGLT2 inhibitor.",
    ]


def test_leading_decimal_quantity_preserved():
    raw = "1.5 mg of dexamethasone reduces mortality in hospitalized patients."
    claims = parse_claims(raw)
    assert len(claims) == 1
    assert (
        claims[0].claim_text
        == "1.5 mg of dexamethasone reduces mortality in hospitalized patients."
    )


def test_genuine_numbered_list_stripping_with_leading_decimal():
    raw = (
        "1. 1.5 mg of dexamethasone reduces mortality.\n"
        "2. 0.9% normal saline is infused for resuscitation.\n"
    )
    claims = parse_claims(raw)
    assert len(claims) == 2
    assert [c.claim_text for c in claims] == [
        "1.5 mg of dexamethasone reduces mortality.",
        "0.9% normal saline is infused for resuscitation.",
    ]


# 5. Bullet-prefixed lines
def test_bullet_prefixed_lines_stripping():
    raw = (
        "- Metformin is first-line therapy.\n"
        "* Sulfonylureas can cause hypoglycemia.\n"
        "• GLP-1 agonists promote weight loss.\n"
        "+ DPP-4 inhibitors are weight neutral.\n"
    )
    claims = parse_claims(raw)
    assert len(claims) == 4
    assert [c.claim_text for c in claims] == [
        "Metformin is first-line therapy.",
        "Sulfonylureas can cause hypoglycemia.",
        "GLP-1 agonists promote weight loss.",
        "DPP-4 inhibitors are weight neutral.",
    ]


# 6. PMID/reference marker stripping
def test_pmid_and_reference_marker_stripping():
    raw = (
        "Metformin reduces HbA1c [PMID: 12345678].\n"
        "Lisinopril lowers blood pressure (PMID: 98765432).\n"
        "Aspirin inhibits COX-1 (References: [PMID: 11111], [PMID: 22222]).\n"
        "Statins reduce LDL-C (as reported in PMID: 33333).\n"
    )
    claims = parse_claims(raw)
    assert len(claims) == 4
    assert [c.claim_text for c in claims] == [
        "Metformin reduces HbA1c.",
        "Lisinopril lowers blood pressure.",
        "Aspirin inhibits COX-1.",
        "Statins reduce LDL-C.",
    ]


# 7. Incomplete fragment filtering (lacking terminal punctuation)
def test_incomplete_fragment_lacking_terminal_punctuation_discarded():
    raw = (
        "Therefore, it\n"
        "Metformin reduces HbA1c in patients with diabetes.\n"
        "pharmacogenomics biom\n"
    )
    claims = parse_claims(raw)
    assert len(claims) == 1
    assert claims[0].claim_text == "Metformin reduces HbA1c in patients with diabetes."


# 8. Balanced/unbalanced bracket/parenthesis behavior
def test_unbalanced_brackets_discarded():
    raw = (
        "Statins reduce cardiovascular events (PMID: 1234.\n"
        "Metformin is effective in adults (as seen in clinical trials).\n"
        "Aspirin prevents stroke [unclosed bracket.\n"
        "Lisinopril is safe [studied extensively].\n"
    )
    claims = parse_claims(raw)
    assert len(claims) == 2
    assert [c.claim_text for c in claims] == [
        "Metformin is effective in adults (as seen in clinical trials).",
        "Lisinopril is safe [studied extensively].",
    ]


# 9. Pure process-commentary filtering
def test_pure_process_commentary_filtered():
    raw = (
        "Here are the extracted atomic claims:\n"
        "Metformin lowers blood glucose.\n"
        "Please let me know if you would like me to rephrase.\n"
        "If you have any further questions, feel free to ask.\n"
        "I cannot provide a definitive answer.\n"
        "As an AI, I suggest consulting a physician.\n"
        "Note: the answer is based on current guidelines.\n"
        "Insulin regulates carbohydrate metabolism.\n"
    )
    claims = parse_claims(raw)
    assert len(claims) == 2
    assert [c.claim_text for c in claims] == [
        "Metformin lowers blood glucose.",
        "Insulin regulates carbohydrate metabolism.",
    ]


# 10. Substantive uncertainty retained
def test_substantive_medical_uncertainty_retained():
    raw = (
        "Statin therapy may reduce cardiovascular risk in elderly patients.\n"
        "The exact biological mechanism might be related to endothelial function.\n"
        "Evidence for high-dose vitamin C could be considered uncertain.\n"
        "The optimal duration of dual antiplatelet therapy remains unclear.\n"
    )
    claims = parse_claims(raw)
    assert len(claims) == 4
    assert [c.claim_text for c in claims] == [
        "Statin therapy may reduce cardiovascular risk in elderly patients.",
        "The exact biological mechanism might be related to endothelial function.",
        "Evidence for high-dose vitamin C could be considered uncertain.",
        "The optimal duration of dual antiplatelet therapy remains unclear.",
    ]


# 11. Exact normalized deduplication
def test_exact_normalized_deduplication():
    raw = (
        "Metformin lowers blood glucose.\n"
        "metformin lowers blood glucose\n"
        "  METFORMIN   LOWERS   BLOOD   GLUCOSE.  \n"
        "Insulin regulates glycogen storage.\n"
    )
    claims = parse_claims(raw)
    assert len(claims) == 2
    # Preserves the first encountered surface form
    assert claims[0].claim_text == "Metformin lowers blood glucose."
    assert claims[1].claim_text == "Insulin regulates glycogen storage."


# 12. Non-semantic near-duplicates are NOT deduplicated
def test_non_semantic_near_duplicates_not_deduplicated():
    raw = (
        "Metformin decreases blood glucose levels.\n"
        "Metformin lowers blood glucose levels.\n"
        "Metformin reduces glucose concentrations.\n"
    )
    claims = parse_claims(raw)
    # Exact normalized equality only: different lexical words are kept
    assert len(claims) == 3


# 13. Zero valid claims after filtering is allowed
def test_zero_valid_claims_allowed_when_all_filtered():
    raw = (
        "Please let me know if you need more clarification.\n"
        "I cannot provide a definitive answer.\n"
        "Incomplete fragment without punctuation\n"
    )
    claims = parse_claims(raw)
    assert claims == []


# 14. Exactly 35 unique valid claims is allowed
def test_exactly_35_claims_allowed():
    lines = [f"Claim number {i} is medically valid." for i in range(1, 36)]
    raw = "\n".join(lines)
    claims = parse_claims(raw, max_claims=MAX_CLAIMS)
    assert len(claims) == 35


# 15. 36 unique valid claims raises ClaimExtractionError
def test_36_claims_raises_claim_extraction_error():
    lines = [f"Claim number {i} is medically valid." for i in range(1, 37)]
    raw = "\n".join(lines)
    with pytest.raises(ClaimExtractionError, match="exceeds the maximum permitted limit of 35"):
        parse_claims(raw, max_claims=MAX_CLAIMS)


# 16. No truncation occurs (fail-loud instead of truncating)
def test_no_truncation_on_overflow():
    lines = [f"Claim number {i} is medically valid." for i in range(1, 40)]
    raw = "\n".join(lines)
    with pytest.raises(ClaimExtractionError):
        parse_claims(raw, max_claims=MAX_CLAIMS)


# 17. Single claim without trailing newline is accepted
def test_single_claim_without_trailing_newline_accepted():
    raw = "Metformin improves insulin sensitivity."
    assert not raw.endswith("\n")
    claims = parse_claims(raw)
    assert len(claims) == 1
    assert claims[0].claim_text == "Metformin improves insulin sensitivity."


# 18. ExtractedClaim schema remains exactly claim_text: str
def test_extracted_claim_schema_fields():
    claim = ExtractedClaim(claim_text="Metformin reduces HbA1c.")
    assert claim.claim_text == "Metformin reduces HbA1c."
    assert list(ExtractedClaim.model_fields.keys()) == ["claim_text"]
    assert ExtractedClaim.model_fields["claim_text"].annotation is str


# 19. Deterministic generation configuration uses do_sample=False
def test_deterministic_generation_configuration_do_sample_false():
    # Calling make_extraction_generate_fn with do_sample=True must be forbidden
    with pytest.raises(ValueError, match="do_sample=False"):
        make_extraction_generate_fn(do_sample=True)

    # Valid call returns callable
    gen_fn = make_extraction_generate_fn(max_new_tokens=256, do_sample=False)
    assert callable(gen_fn)


# 20. Existing generation/model infrastructure is reused rather than duplicated
def test_claim_extractor_reuses_existing_infrastructure():
    recorded_prompts = []

    def fake_generate_fn(prompt: str, tokenizer: object, model: object) -> str:
        recorded_prompts.append(prompt)
        return (
            "Metformin reduces hepatic glucose production.\n"
            "Metformin improves peripheral insulin sensitivity.\n"
        )

    fake_model = object()
    fake_tokenizer = object()

    extractor = ClaimExtractor(
        model=fake_model,
        tokenizer=fake_tokenizer,
        generate_fn=fake_generate_fn,
        max_claims=35,
    )

    # Test with GeneratedAnswer instance
    answer = GeneratedAnswer(
        question="What does metformin do?",
        answer_text=(
            "Metformin reduces hepatic glucose production and improves insulin sensitivity."
        ),
        context=[],
        model_name="fake-llama",
    )

    claims = extractor.extract_claims(answer)
    assert len(claims) == 2
    assert claims[0].claim_text == "Metformin reduces hepatic glucose production."
    assert claims[1].claim_text == "Metformin improves peripheral insulin sensitivity."

    assert len(recorded_prompts) == 1
    assert CLAIM_EXTRACTION_SYSTEM_PROMPT in recorded_prompts[0]
    assert answer.answer_text in recorded_prompts[0]

    # Test with raw string
    claims_str = extractor.extract_claims("Simple answer text.")
    assert len(claims_str) == 2
    assert len(recorded_prompts) == 2


def test_build_claim_extraction_prompt_structure():
    prompt = build_claim_extraction_prompt("Medical answer text.")
    assert prompt.startswith(CLAIM_EXTRACTION_SYSTEM_PROMPT)
    assert "Medical answer text." in prompt
    assert CLAIM_EXTRACTION_USER_TEMPLATE.format(answer_text="Medical answer text.") in prompt

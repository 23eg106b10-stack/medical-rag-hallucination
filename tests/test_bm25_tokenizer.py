"""Tests for the deterministic BM25 tokenizer (M3.1.2).

Verifies the contract: lowercase, strip punctuation, whitespace split,
discard empty tokens. No stemming, lemmatization, or stopword removal.
"""

from __future__ import annotations

from index.bm25_tokenizer import tokenize


class TestTokenizeLowercasing:
    """All tokens must be lowercase."""

    def test_uppercase_becomes_lowercase(self) -> None:
        assert tokenize("HELLO") == ["hello"]

    def test_mixed_case_becomes_lowercase(self) -> None:
        assert tokenize("Hello World") == ["hello", "world"]


class TestTokenizePunctuationStripping:
    """Punctuation is removed and replaced by whitespace."""

    def test_comma_removed(self) -> None:
        tokens = tokenize("hello, world")
        assert tokens == ["hello", "world"]

    def test_period_removed(self) -> None:
        tokens = tokenize("end of sentence.")
        assert tokens == ["end", "of", "sentence"]

    def test_question_mark_removed(self) -> None:
        tokens = tokenize("What is this?")
        assert tokens == ["what", "is", "this"]

    def test_parentheses_removed(self) -> None:
        tokens = tokenize("text (with) parens")
        assert tokens == ["text", "with", "parens"]

    def test_unicode_punctuation_removed(self) -> None:
        # Unicode punctuation should also be stripped
        tokens = tokenize("café — restaurant")
        assert tokens == ["café", "restaurant"]


class TestTokenizeWhitespaceSplitting:
    """Tokens are separated by any whitespace."""

    def test_multiple_spaces_collapsed(self) -> None:
        tokens = tokenize("hello   world")
        assert tokens == ["hello", "world"]

    def test_tab_and_newline_split(self) -> None:
        tokens = tokenize("hello\tworld\nfoo")
        assert tokens == ["hello", "world", "foo"]


class TestTokenizeEmptyAndWhitespace:
    """Empty strings and whitespace-only strings yield no tokens."""

    def test_empty_string_returns_empty(self) -> None:
        assert tokenize("") == []

    def test_whitespace_only_returns_empty(self) -> None:
        assert tokenize("   \t\n  ") == []


class TestTokenizeOrderPreservation:
    """Token order in the result matches original text order."""

    def test_order_preserved(self) -> None:
        tokens = tokenize("alpha beta gamma")
        assert tokens == ["alpha", "beta", "gamma"]


class TestTokenizeMixedPunctuationAndCasing:
    """Realistic titles and abstracts tokenize correctly."""

    def test_realistic_title(self) -> None:
        title = "A Systematic Review of COVID-19 Treatments"
        tokens = tokenize(title)
        assert tokens == [
            "a",
            "systematic",
            "review",
            "of",
            "covid",
            "19",
            "treatments",
        ]

    def test_realistic_abstract_snippet(self) -> None:
        abstract = (
            "Background: Diabetes is a growing concern. Methods: We searched "
            "PubMed and extracted 50 studies. Results: HbA1c improved (p<0.05)."
        )
        tokens = tokenize(abstract)
        assert "background" in tokens
        assert "diabetes" in tokens
        assert "methods" in tokens
        assert "results" in tokens
        # numeric tokens preserved
        assert "50" in tokens
        assert "0" in tokens


class TestTokenizeNoStemmingOrStopwords:
    """Tokenizer is intentionally naive — no stemming or stopword removal."""

    def test_no_stemming_running(self) -> None:
        # "running" stays as-is; no Porter/Snowball stemming
        assert tokenize("running") == ["running"]

    def test_stopwords_preserved(self) -> None:
        tokens = tokenize("the quick brown fox")
        assert "the" in tokens
        assert "a" not in tokens  # 'a' is not in the input

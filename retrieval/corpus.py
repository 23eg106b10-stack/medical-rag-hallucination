"""In-memory corpus lookup for hybrid retrieval (M3.2).

Loads the frozen ``corpus.jsonl`` once and serves PMID -> CorpusDocument
lookups. ``corpus.jsonl`` has no native random-access structure (it is a
flat JSON Lines file), so this trades startup memory for O(1) lookup, per
the M3.2 repository design decision (in-memory dict chosen over
byte-offset seeking or a SQLite conversion).
"""

from __future__ import annotations

import logging
from pathlib import Path

from schemas.corpus import CorpusDocument

logger = logging.getLogger(__name__)


class MissingDocumentError(KeyError):
    """Raised when a requested PMID is not present in the loaded corpus.

    This indicates retrieval indices (BM25/FAISS) and the corpus have
    drifted out of sync — it is not a recoverable, silently-skippable
    condition.
    """


class InMemoryCorpusReader:
    """Serves PMID -> CorpusDocument lookups from an in-memory dict.

    Per the M3.2 contract: ``get_documents`` preserves exact input order
    and raises rather than silently omitting missing PMIDs. Callers must
    never rely on positional zipping against a possibly-shortened result.
    """

    def __init__(self, documents_by_pmid: dict[str, CorpusDocument]) -> None:
        self._documents_by_pmid = documents_by_pmid

    @classmethod
    def from_jsonl(cls, corpus_path: Path) -> InMemoryCorpusReader:
        """Build a reader by parsing ``corpus.jsonl`` into an in-memory dict.

        Args:
            corpus_path: Path to the frozen ``corpus.jsonl`` (M3.1.1 output).

        Returns:
            A populated ``InMemoryCorpusReader``.

        Raises:
            FileNotFoundError: if ``corpus_path`` does not exist.
        """
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus not found: {corpus_path}")

        documents_by_pmid: dict[str, CorpusDocument] = {}
        with corpus_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    doc = CorpusDocument.model_validate_json(line)
                    documents_by_pmid[doc.pmid] = doc

        logger.info(
            "Loaded %d documents into memory from %s",
            len(documents_by_pmid),
            corpus_path,
        )
        return cls(documents_by_pmid)

    def get_documents(self, pmids: list[str]) -> list[CorpusDocument]:
        """Resolve a list of PMIDs to their corpus documents.

        Args:
            pmids: PMIDs to resolve, in the desired output order.

        Returns:
            Documents in the exact same order as ``pmids`` — never
            shortened, never reordered.

        Raises:
            MissingDocumentError: if any PMID is not present in the corpus.
        """
        results: list[CorpusDocument] = []
        for pmid in pmids:
            doc = self._documents_by_pmid.get(pmid)
            if doc is None:
                raise MissingDocumentError(
                    f"PMID {pmid!r} not found in corpus — retrieval indices "
                    "and corpus are out of sync."
                )
            results.append(doc)
        return results

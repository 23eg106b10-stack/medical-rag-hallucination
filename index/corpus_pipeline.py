"""Corpus construction pipeline: normalize, dedupe, validate, write.

Pure/local logic only — no network or filesystem access to external
datasets. All I/O against PubMedQA/NCBI lives in ``corpus_sources``; this
module receives already-fetched raw records and turns them into the frozen
corpus, per the M3.1.1 architecture pipeline stages.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

from schemas.corpus import CorpusDocument, CorpusMetadata

logger = logging.getLogger(__name__)

_MIN_ABSTRACT_TOKENS = 20
_PLACEHOLDER_ABSTRACTS = {
    "no abstract available",
    "no abstract",
    "abstract not available",
    "n/a",
}


def normalize(raw: dict[str, str]) -> CorpusDocument | None:
    """Normalize a raw record into a ``CorpusDocument``.

    Args:
        raw: Raw record with (at minimum) ``pmid``, ``title``, ``abstract``
            keys, as produced by ``corpus_sources``.

    Returns:
        A ``CorpusDocument`` with whitespace-normalized fields, or ``None``
        if the record is missing its PMID (uncorrectable — skipped by the
        caller rather than repaired, per the architecture's no-repair rule).
    """
    pmid = (raw.get("pmid") or "").strip()
    if not pmid:
        return None

    title = (raw.get("title") or "").strip()
    abstract = (raw.get("abstract") or "").strip()

    return CorpusDocument(pmid=pmid, title=title, abstract=abstract)


def deduplicate(documents: list[tuple[CorpusDocument, bool]]) -> list[CorpusDocument]:
    """Deduplicate documents by PMID, applying source precedence.

    Each input pair carries its provenance (``is_authoritative``) alongside
    the document itself, captured by the caller at fetch time — before
    normalization would otherwise discard it. ``CorpusDocument`` itself
    never carries a source field; provenance is only ever handled here,
    internally, and is discarded once precedence has been applied.

    When the same PMID appears more than once, the record where
    ``is_authoritative`` is ``True`` (i.e. retrieved to satisfy Source B /
    NCBI E-utilities coverage) wins outright over one that is only present
    to satisfy Source A / PubMedQA PMID coverage. The loser is discarded in
    full — no field-level merging or similarity comparison is performed,
    per the M3.1.1 source precedence rule.

    Args:
        documents: ``(document, is_authoritative)`` pairs from all sources,
            in any order.

    Returns:
        One document per unique PMID.
    """
    by_pmid: dict[str, tuple[CorpusDocument, bool]] = {}

    for doc, is_authoritative in documents:
        existing = by_pmid.get(doc.pmid)
        if existing is None:
            by_pmid[doc.pmid] = (doc, is_authoritative)
            continue

        _, existing_authoritative = existing
        if is_authoritative and not existing_authoritative:
            by_pmid[doc.pmid] = (doc, is_authoritative)
        # Otherwise keep whichever is already stored — do not overwrite an
        # authoritative record with a non-authoritative one.

    logger.info("Deduplicated %d documents -> %d unique PMIDs", len(documents), len(by_pmid))
    return [doc for doc, _ in by_pmid.values()]


def is_valid(document: CorpusDocument) -> bool:
    """Check a document against the M3.1.1 validation rules.

    PMID and title must be non-empty; the abstract must be non-empty and
    "meaningful" — not a placeholder string and not degenerately short.
    The concrete thresholds here are an implementation decision, per the
    architecture's validation principle (not frozen at the architecture
    level).

    Args:
        document: The normalized document to check.

    Returns:
        ``True`` if the document satisfies all validation rules.
    """
    if not document.pmid or not document.title:
        return False

    abstract = document.abstract.strip()
    if not abstract:
        return False
    if abstract.lower() in _PLACEHOLDER_ABSTRACTS:
        return False
    if len(abstract.split()) < _MIN_ABSTRACT_TOKENS:
        return False

    return True


def write_corpus(
    documents: list[CorpusDocument], corpus_dir: Path, corpus_version: str
) -> CorpusMetadata:
    """Write the frozen corpus and its accompanying metadata record.

    Args:
        documents: Validated, deduplicated documents to write.
        corpus_dir: Directory to write ``corpus.jsonl`` and
            ``corpus.meta.json`` into. Created if it does not exist.
        corpus_version: Version identifier for this corpus generation run.

    Returns:
        The ``CorpusMetadata`` record that was also written to disk.
    """
    corpus_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = corpus_dir / "corpus.jsonl"
    meta_path = corpus_dir / "corpus.meta.json"

    with corpus_path.open("w", encoding="utf-8") as f:
        for doc in documents:
            f.write(doc.model_dump_json())
            f.write("\n")

    checksum = hashlib.sha256(corpus_path.read_bytes()).hexdigest()

    metadata = CorpusMetadata(
        corpus_version=corpus_version,
        generated_at=datetime.now(timezone.utc).isoformat(),
        document_count=len(documents),
        checksum_sha256=checksum,
    )

    with meta_path.open("w", encoding="utf-8") as f:
        f.write(metadata.model_dump_json(indent=2))

    logger.info(
        "Wrote corpus: documents=%d, version=%s, checksum=%s",
        metadata.document_count,
        metadata.corpus_version,
        metadata.checksum_sha256,
    )
    return metadata

"""BM25 index construction pipeline (M3.1.2).

Local file I/O only (reading corpus.jsonl, writing the index and its
metadata) — no network access, no retrieval, scoring, or ranking. Each
stage is independently callable and deterministic given the same corpus
and tokenizer, per the M3.1.2 architecture.
"""

from __future__ import annotations

import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from rank_bm25 import BM25Okapi

from schemas.bm25 import BM25Metadata
from schemas.corpus import CorpusDocument, CorpusMetadata

logger = logging.getLogger(__name__)

_BM25_VERSION = "1.0"
_TOKENIZER_NAME = "simple"
_ALGORITHM = "BM25Okapi"
_LIBRARY = "rank-bm25"


def load_corpus(corpus_path: Path) -> list[CorpusDocument]:
    """Load and parse the frozen corpus.

    Args:
        corpus_path: Path to ``corpus.jsonl`` produced by M3.1.1.

    Returns:
        Parsed documents, in file order.

    Raises:
        FileNotFoundError: if ``corpus_path`` does not exist.
    """
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    documents: list[CorpusDocument] = []
    with corpus_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                documents.append(CorpusDocument.model_validate_json(line))

    logger.info("Loaded %d documents from %s", len(documents), corpus_path)
    return documents


def load_corpus_version(corpus_dir: Path) -> str:
    """Read the corpus version out of the M3.1.1 metadata record.

    Args:
        corpus_dir: Directory containing ``corpus.meta.json``.

    Returns:
        The ``corpus_version`` field from that record.

    Raises:
        FileNotFoundError: if ``corpus.meta.json`` does not exist.
    """
    meta_path = corpus_dir / "corpus.meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Corpus metadata not found: {meta_path}")

    metadata = CorpusMetadata.model_validate_json(meta_path.read_text(encoding="utf-8"))
    return metadata.corpus_version


def validate_corpus(documents: list[CorpusDocument]) -> None:
    """Validate corpus integrity before indexing.

    Args:
        documents: Documents loaded from ``corpus.jsonl``.

    Raises:
        ValueError: if the corpus is empty, or any document is missing a
            PMID, title, or abstract.
    """
    if not documents:
        raise ValueError("Corpus is empty — refusing to build a BM25 index.")

    for doc in documents:
        if not doc.pmid or not doc.title or not doc.abstract:
            raise ValueError(f"Corpus document is missing a required field: pmid={doc.pmid!r}")


def extract_document_text(document: CorpusDocument) -> str:
    """Extract the text to index for a single document.

    Title and abstract are concatenated, since M3.1.2 does not specify
    field weighting and BM25 here is a single flat lexical index.

    Args:
        document: The corpus document.

    Returns:
        Combined title + abstract text.
    """
    return f"{document.title} {document.abstract}"


def build_index(
    documents: list[CorpusDocument],
    tokenizer: Callable[[str], list[str]],
) -> tuple[BM25Okapi, list[str]]:
    """Tokenize documents and build the BM25 model.

    Args:
        documents: Validated corpus documents.
        tokenizer: Tokenizer function to apply to each document's text.

    Returns:
        The fitted ``BM25Okapi`` model, and the list of PMIDs in the same
        order as the model's internal document positions.

    Raises:
        ValueError: if any document tokenizes to an empty token list.
    """
    corpus_tokens: list[list[str]] = []
    pmids: list[str] = []

    for doc in documents:
        tokens = tokenizer(extract_document_text(doc))
        if not tokens:
            raise ValueError(f"Document {doc.pmid} tokenized to an empty token list.")
        corpus_tokens.append(tokens)
        pmids.append(doc.pmid)

    model = BM25Okapi(corpus_tokens)
    logger.info("Built BM25Okapi model over %d documents", len(pmids))
    return model, pmids


def serialize_index(model: BM25Okapi, pmids: list[str], index_path: Path) -> None:
    """Serialize the BM25 model and its PMID ordering to disk.

    The pickled payload is a dict of ``{"model": BM25Okapi, "pmids": [...]}``
    — the PMID list is embedded here rather than as a separate output file,
    since M3.1.2 names only ``bm25_index.pkl`` and ``bm25_metadata.json`` as
    outputs, and the index is unusable without a way to map result
    positions back to PMIDs.

    Args:
        model: The fitted BM25 model.
        pmids: PMIDs in the same order as the model's document positions.
        index_path: Destination path for the pickle file.

    Raises:
        OSError: if the index directory cannot be created.
        pickle.PicklingError: if serialization fails.
    """
    index_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with index_path.open("wb") as f:
            pickle.dump({"model": model, "pmids": pmids}, f)
    except pickle.PicklingError:
        logger.error("Failed to serialize BM25 index to %s", index_path)
        raise

    logger.info("Serialized BM25 index to %s", index_path)


def write_metadata(document_count: int, corpus_version: str, metadata_path: Path) -> BM25Metadata:
    """Write the BM25 index metadata record.

    Args:
        document_count: Number of documents in the index.
        corpus_version: Version of the source corpus (from M3.1.1 metadata).
        metadata_path: Destination path for ``bm25_metadata.json``.

    Returns:
        The ``BM25Metadata`` record that was written.
    """
    metadata = BM25Metadata(
        bm25_version=_BM25_VERSION,
        corpus_version=corpus_version,
        document_count=document_count,
        tokenizer=_TOKENIZER_NAME,
        algorithm=_ALGORITHM,
        library=_LIBRARY,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")

    logger.info("Wrote BM25 metadata to %s", metadata_path)
    return metadata

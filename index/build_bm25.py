"""Build the BM25 lexical index from the frozen corpus (M3.1.2).

Entry point only: wires ``bm25_pipeline`` stages together. Contains no
business logic of its own, per the same single-responsibility pattern
established in M3.1.1\'s ``build_corpus.py``.

Run as: python -m index.build_bm25
"""

from __future__ import annotations

import logging

from config.settings import get_settings
from index.bm25_pipeline import (
    build_index,
    load_corpus,
    load_corpus_version,
    serialize_index,
    validate_corpus,
    write_metadata,
)
from index.bm25_tokenizer import tokenize
from utils.logging import setup_logging

logger = logging.getLogger(__name__)


def build_bm25_index() -> None:
    """Run the full M3.1.2 BM25 index construction pipeline end to end.

    Raises:
        FileNotFoundError: if the corpus or its metadata record is missing.
        ValueError: if the corpus is empty, malformed, or any document
            tokenizes to an empty token list.
        pickle.PicklingError: if index serialization fails.
    """
    settings = get_settings()
    corpus_path = settings.corpus_dir / "corpus.jsonl"
    index_path = settings.indexes_dir / "bm25_index.pkl"
    metadata_path = settings.indexes_dir / "bm25_metadata.json"

    logger.info("Reading corpus...")
    documents = load_corpus(corpus_path)
    corpus_version = load_corpus_version(settings.corpus_dir)

    logger.info("Validating corpus...")
    validate_corpus(documents)

    logger.info("Tokenizing and building BM25 index...")
    model, pmids = build_index(documents, tokenizer=tokenize)

    logger.info("Serializing index...")
    serialize_index(model, pmids, index_path)

    logger.info("Writing metadata...")
    metadata = write_metadata(len(pmids), corpus_version, metadata_path)

    logger.info("BM25 index complete.")
    logger.info("Documents indexed: %d", metadata.document_count)
    logger.info("Corpus version: %s", metadata.corpus_version)
    logger.info("BM25 version: %s", metadata.bm25_version)


if __name__ == "__main__":
    setup_logging(log_filename="build_bm25.log")
    build_bm25_index()
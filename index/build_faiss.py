"""Build the dense (FAISS) vector index from the frozen corpus (M3.1.3).

Entry point only: wires ``embedding_generator`` and ``faiss_pipeline``
stages together. Contains no business logic of its own, per the same
single-responsibility pattern established in M3.1.1's ``build_corpus.py``
and M3.1.2's ``build_bm25.py``.

Run as: python -m index.build_faiss
"""

from __future__ import annotations

import logging

from config.settings import get_settings
from index.embedding_generator import load_encoder
from index.faiss_pipeline import (
    build_index,
    generate_embeddings,
    load_corpus,
    load_corpus_metadata,
    normalize_embeddings,
    validate_corpus,
    write_index_and_metadata,
)
from utils.logging import setup_logging

logger = logging.getLogger(__name__)


def build_faiss_index() -> None:
    """Run the full M3.1.3 dense index construction pipeline end to end.

    Raises:
        FileNotFoundError: if the corpus or its metadata record is missing.
        ValueError: if the corpus is empty or malformed.
        RuntimeError: if embedding generation fails for any batch - the
            pipeline aborts immediately and no index is written, per the
            M3.1.3 error boundary decision (no partial index).
    """
    settings = get_settings()
    corpus_path = settings.corpus_dir / "corpus.jsonl"
    index_path = settings.indexes_dir / settings.faiss_index_filename
    metadata_path = settings.indexes_dir / "embedding_metadata.json"
    pmids_path = settings.indexes_dir / "faiss_pmids.json"

    logger.info("Reading corpus...")
    documents = load_corpus(corpus_path)
    corpus_metadata = load_corpus_metadata(settings.corpus_dir)

    logger.info("Validating corpus...")
    validate_corpus(documents)

    logger.info("Loading MedCPT encoder...")
    tokenizer, model = load_encoder(settings.medcpt_model_name)

    logger.info("Generating embeddings...")
    vectors = generate_embeddings(documents, tokenizer, model, settings.embedding_batch_size)

    logger.info("Normalizing embeddings...")
    vectors = normalize_embeddings(vectors)

    logger.info("Building FAISS index...")
    index = build_index(vectors)

    logger.info("Writing index and metadata...")
    metadata = write_index_and_metadata(
        index=index,
        pmids=[doc.pmid for doc in documents],
        document_count=len(documents),
        embedding_dimension=vectors.shape[1],
        corpus_version=corpus_metadata.corpus_version,
        corpus_checksum=corpus_metadata.checksum_sha256,
        model_name=settings.medcpt_model_name,
        index_path=index_path,
        metadata_path=metadata_path,
        pmids_path=pmids_path,
    )

    logger.info("FAISS index complete.")
    logger.info("Documents indexed: %d", metadata.document_count)
    logger.info("Corpus version: %s", metadata.corpus_version)
    logger.info("Embedding version: %s", metadata.embedding_version)


if __name__ == "__main__":
    setup_logging(log_filename="build_faiss.log")
    build_faiss_index()

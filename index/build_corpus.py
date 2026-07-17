"""Build the retrieval corpus from raw medical datasets (M3.1.1).

Entry point only: wires ``corpus_sources`` (external I/O) to
``corpus_pipeline`` (normalize/dedupe/validate/write). Contains no business
logic of its own, per the M3.1.1 architecture's single-responsibility
requirement for this component.

Run as: python -m index.build_corpus
"""

from __future__ import annotations

import logging

import httpx

from config.settings import get_settings
from index.corpus_pipeline import deduplicate, is_valid, normalize, write_corpus
from index.corpus_sources import (
    fetch_ncbi_abstracts,
    load_pubmedqa_pmids,
    load_query_specification,
    search_ncbi_pmids,
)
from schemas.corpus import CorpusDocument
from utils.logging import setup_logging

logger = logging.getLogger(__name__)


def build_corpus() -> None:
    """Run the full M3.1.1 corpus construction pipeline end to end.

    Raises:
        FileNotFoundError: if required input files (PubMedQA data, the
            query specification) are missing.
        httpx.HTTPError: on NCBI network interruption — corpus generation
            stops and produces no output, per the architecture's error
            handling rule.
    """
    settings = get_settings()

    logger.info("Downloading PMIDs...")
    pubmedqa_pmids = load_pubmedqa_pmids(settings)
    query_spec = load_query_specification(settings)

    # Each raw record is paired with its provenance at the point of fetch —
    # before normalization would otherwise discard it. True = retrieved to
    # satisfy Source B (NCBI) coverage; False = retrieved only to satisfy
    # Source A (PubMedQA) PMID coverage. See corpus_pipeline.deduplicate.
    raw_records: list[tuple[dict[str, str], bool]] = []
    ncbi_pmids: set[str] = set()

    with httpx.Client(timeout=30.0) as client:
        for entry in query_spec["queries"]:
            mesh_pmids = search_ncbi_pmids(entry["mesh_term"], settings, client)
            ncbi_pmids.update(mesh_pmids)

        logger.info("Downloading abstracts...")
        ncbi_records = fetch_ncbi_abstracts(sorted(ncbi_pmids), settings, client)
        raw_records.extend((record, True) for record in ncbi_records)

        pubmedqa_only = sorted(set(pubmedqa_pmids) - ncbi_pmids)
        if pubmedqa_only:
            pubmedqa_records = fetch_ncbi_abstracts(pubmedqa_only, settings, client)
            raw_records.extend((record, False) for record in pubmedqa_records)

    logger.info("Normalizing corpus...")
    documents: list[tuple[CorpusDocument, bool]] = []
    for raw, is_authoritative in raw_records:
        doc = normalize(raw)
        if doc is not None:
            documents.append((doc, is_authoritative))

    logger.info("Removing duplicates...")
    deduped_documents = deduplicate(documents)

    logger.info("Validating corpus...")
    valid_documents = [doc for doc in deduped_documents if is_valid(doc)]
    logger.info("Validation: %d/%d documents kept", len(valid_documents), len(deduped_documents))

    logger.info("Writing corpus...")
    # corpus_version is intentionally derived from the query specification's
    # own version, not set independently: per the M3.1.1 Query Specification
    # Contract, any change to queries.yaml constitutes a new corpus
    # specification version, so tying corpus_version to it directly is what
    # keeps the two from silently drifting out of sync.
    metadata = write_corpus(
        valid_documents,
        corpus_dir=settings.corpus_dir,
        corpus_version=query_spec["version"],
    )

    logger.info("Corpus complete.")
    logger.info("Documents written: %d", metadata.document_count)
    logger.info("Corpus version: %s", metadata.corpus_version)
    logger.info("Checksum: %s", metadata.checksum_sha256)


if __name__ == "__main__":
    setup_logging(log_filename="build_corpus.log")
    build_corpus()

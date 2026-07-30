"""External data sources for corpus construction (M3.1.1, Source A / B).

This module owns every network and dataset-file access made by corpus
construction. Nothing else in the pipeline talks to PubMedQA files or the
NCBI API directly — that isolation is what lets ``corpus_pipeline`` be
tested without network access.

Raw records returned here are untyped dicts, not ``CorpusDocument``:
normalization into the canonical schema happens in ``corpus_pipeline``, per
the M3.1.1 responsibility split (obtain vs. normalize).
"""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import httpx
import yaml

from config.settings import Settings

logger = logging.getLogger(__name__)

_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_EFETCH_BATCH_SIZE = 200


def load_pubmedqa_pmids(settings: Settings) -> list[str]:
    """Load the list of PMIDs required for PubMedQA coverage (Source A).

    Expects the PubMedQA release file named by ``settings.pubmedqa_filename``
    (default ``ori_pqal.json``) under ``settings.pubmedqa_dir``, whose
    top-level keys are PMIDs.

    Args:
        settings: Application settings, used to locate the PubMedQA data
            directory and filename.

    Returns:
        List of PMIDs (as strings) required for PubMedQA coverage.

    Raises:
        FileNotFoundError: if the expected PubMedQA file is not present.
    """
    path = settings.pubmedqa_dir / settings.pubmedqa_filename
    if not path.exists():
        raise FileNotFoundError(f"PubMedQA source file not found: {path}")

    with path.open(encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    pmids = list(data.keys())
    logger.info("Loaded %d PMIDs from PubMedQA (Source A)", len(pmids))
    return pmids


def load_query_specification(settings: Settings) -> dict[str, Any]:
    """Load the frozen MeSH query specification (Query Specification Contract).

    This component consumes ``queries.yaml`` and never modifies it.

    Args:
        settings: Application settings, used to locate ``queries_path``.

    Returns:
        The parsed specification, with ``version`` (str) and ``queries``
        (list of ``{"subject_area": ..., "mesh_term": ...}``) keys.

    Raises:
        FileNotFoundError: if ``queries_path`` does not exist.
        ValueError: if the file is missing the expected ``version`` or
            ``queries`` keys.
    """
    path: Path = settings.queries_path
    if not path.exists():
        raise FileNotFoundError(f"Query specification not found: {path}")

    with path.open(encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    if not isinstance(spec, dict) or "queries" not in spec or "version" not in spec:
        raise ValueError(f"Malformed query specification: {path}")

    if not spec["queries"]:
        raise ValueError(
            f"Query specification at {path} has no queries — it is blocked "
            "pending the project's approved MeSH query list and must not be "
            "used to run corpus construction."
        )

    logger.info(
        "Loaded query specification version=%s (%d queries)",
        spec["version"],
        len(spec["queries"]),
    )
    return spec


def _eutils_params(settings: Settings) -> dict[str, str]:
    params = {"tool": "medical-rag-hallucination", "email": settings.ncbi_email}
    if settings.ncbi_api_key:
        params["api_key"] = settings.ncbi_api_key
    return params


def search_ncbi_pmids(mesh_term: str, settings: Settings, client: httpx.Client) -> list[str]:
    """Run one NCBI ESearch query and return matching PMIDs.

    Args:
        mesh_term: A single MeSH-qualified query term from the query
            specification.
        settings: Application settings, used for E-utilities contact info.
        client: Shared ``httpx.Client`` for connection reuse across calls.

    Returns:
        List of matching PMIDs (as strings).

    Raises:
        httpx.HTTPError: on network interruption or non-2xx response, per
            the M3.1.1 error handling rule (raise and stop).
    """
    response = client.get(
        f"{_EUTILS_BASE}/esearch.fcgi",
        params={
            **_eutils_params(settings),
            "db": "pubmed",
            "term": mesh_term,
            "retmax": "10000",
        },
    )
    response.raise_for_status()

    root = ET.fromstring(response.text)
    pmids = [el.text for el in root.findall(".//IdList/Id") if el.text]
    logger.info("NCBI ESearch %r -> %d PMIDs", mesh_term, len(pmids))
    return pmids


def fetch_ncbi_abstracts(
    pmids: list[str], settings: Settings, client: httpx.Client
) -> list[dict[str, str]]:
    """Fetch title/abstract metadata for a batch of PMIDs via NCBI EFetch.

    Args:
        pmids: PMIDs to fetch, deduplicated by the caller.
        settings: Application settings, used for E-utilities contact info.
        client: Shared ``httpx.Client`` for connection reuse across calls.

    Returns:
        List of raw records with keys ``pmid``, ``title``, ``abstract``.
        A PMID with no abstract text in the response is omitted here and
        will be caught by validation as a missing-abstract skip.

    Raises:
        httpx.HTTPError: on network interruption or non-2xx response, per
            the M3.1.1 error handling rule (raise and stop).
    """
    records: list[dict[str, str]] = []

    for start in range(0, len(pmids), _EFETCH_BATCH_SIZE):
        batch = pmids[start : start + _EFETCH_BATCH_SIZE]
        response = client.get(
            f"{_EUTILS_BASE}/efetch.fcgi",
            params={
                **_eutils_params(settings),
                "db": "pubmed",
                "id": ",".join(batch),
                "retmode": "xml",
            },
        )
        response.raise_for_status()
        records.extend(_parse_efetch_xml(response.text))

    logger.info(
        "NCBI EFetch retrieved %d records for %d requested PMIDs",
        len(records),
        len(pmids),
    )
    return records


def _parse_efetch_xml(xml_text: str) -> list[dict[str, str]]:
    """Parse a PubmedArticleSet EFetch response into raw records."""
    root = ET.fromstring(xml_text)
    records: list[dict[str, str]] = []

    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//MedlineCitation/PMID")
        title_el = article.find(".//Article/ArticleTitle")
        abstract_els = article.findall(".//Article/Abstract/AbstractText")

        if pmid_el is None or pmid_el.text is None:
            continue

        title = title_el.text if title_el is not None and title_el.text else ""
        abstract = " ".join(el.text for el in abstract_els if el.text)

        records.append({"pmid": pmid_el.text, "title": title, "abstract": abstract})

    return records

"""Dense vector index construction pipeline (M3.1.3).

Local file I/O only (reading corpus.jsonl/corpus.meta.json, writing the
FAISS index and its metadata) plus orchestration of ``embedding_generator``
— no query encoding, search, or retrieval logic, per the M3.1.3
architecture. Each stage is independently callable; the pipeline as a
whole aborts on any embedding failure rather than producing a partial
index (no partial index is a valid output).

Owns corpus loading, validation, and document text extraction locally
rather than importing from ``index.bm25_pipeline`` — BM25 and FAISS are
sibling pipelines over the same corpus and must not depend on one
another, even though the logic is small and currently identical.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from index.embedding_generator import embed_batch
from schemas.corpus import CorpusDocument, CorpusMetadata
from schemas.embedding import EmbeddingMetadata

logger = logging.getLogger(__name__)

_EMBEDDING_VERSION = "1.0"
_FAISS_INDEX_TYPE = "IndexFlatIP"
_EMBEDDING_DTYPE = "float32"


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


def load_corpus_metadata(corpus_dir: Path) -> CorpusMetadata:
    """Read the M3.1.1 corpus metadata record, including its checksum.

    Args:
        corpus_dir: Directory containing ``corpus.meta.json``.

    Returns:
        The parsed ``CorpusMetadata`` record.

    Raises:
        FileNotFoundError: if ``corpus.meta.json`` does not exist.
    """
    meta_path = corpus_dir / "corpus.meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Corpus metadata not found: {meta_path}")

    return CorpusMetadata.model_validate_json(meta_path.read_text(encoding="utf-8"))


def validate_corpus(documents: list[CorpusDocument]) -> None:
    """Validate corpus integrity before embedding.

    Args:
        documents: Documents loaded from ``corpus.jsonl``.

    Raises:
        ValueError: if the corpus is empty, or any document is missing a
            PMID, title, or abstract.
    """
    if not documents:
        raise ValueError("Corpus is empty — refusing to build a FAISS index.")

    for doc in documents:
        if not doc.pmid or not doc.title or not doc.abstract:
            raise ValueError(f"Corpus document is missing a required field: pmid={doc.pmid!r}")


def extract_document_text(document: CorpusDocument) -> str:
    """Extract the text to embed for a single document.

    Title and abstract are concatenated, matching the sparse (BM25)
    representation exactly, per Design Decision 2.

    Args:
        document: The corpus document.

    Returns:
        Combined title + abstract text.
    """
    return f"{document.title} {document.abstract}"


def generate_embeddings(
    documents: list[CorpusDocument],
    tokenizer: Any,
    model: Any,
    batch_size: int,
) -> np.ndarray:
    """Generate MedCPT embeddings for all documents, in corpus order.

    Documents are encoded in batches of ``batch_size``. Any failure aborts
    the entire pipeline immediately rather than producing a partial index,
    per the M3.1.3 error boundary decision.

    Args:
        documents: Validated corpus documents, in corpus file order.
        tokenizer: A loaded MedCPT tokenizer, from ``embedding_generator.load_encoder``.
        model: A loaded MedCPT model, from ``embedding_generator.load_encoder``.
        batch_size: Number of documents per forward pass (from
            ``settings.embedding_batch_size``).

    Returns:
        ``float32`` array of shape ``(len(documents), embedding_dim)``, in
        the same order as ``documents``.

    Raises:
        RuntimeError: if embedding generation fails for any batch.
    """
    texts = [extract_document_text(doc) for doc in documents]
    batches: list[np.ndarray] = []

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        try:
            batches.append(embed_batch(batch_texts, tokenizer, model))
        except RuntimeError:
            batch_pmids = [d.pmid for d in documents[start : start + batch_size]]
            logger.error("Embedding generation failed for batch PMIDs: %s", batch_pmids)
            raise

    array = np.vstack(batches).astype(np.float32)
    logger.info("Generated %d embeddings (dim=%d)", array.shape[0], array.shape[1])
    return array


def normalize_embeddings(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize embeddings row-wise, per M3.1.3 ADR-001.

    Args:
        vectors: ``float32`` array of shape ``(n, dim)``.

    Returns:
        The same array, L2-normalized in place.
    """
    faiss.normalize_L2(vectors)
    return vectors


def build_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    """Build a FAISS ``IndexFlatIP`` over the given vectors.

    Args:
        vectors: ``float32`` array of shape ``(n, dim)``. Callers are
            responsible for normalizing beforehand if cosine similarity is
            required — this function performs no normalization itself.

    Returns:
        The populated ``IndexFlatIP``.
    """
    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)
    logger.info("Built IndexFlatIP with %d vectors (dim=%d)", index.ntotal, dimension)
    return index


def _metadata_from_inputs(
    document_count: int,
    embedding_dimension: int,
    corpus_version: str,
    corpus_checksum: str,
    model_name: str,
) -> EmbeddingMetadata:
    return EmbeddingMetadata(
        embedding_version=_EMBEDDING_VERSION,
        corpus_version=corpus_version,
        corpus_checksum=corpus_checksum,
        model_name=model_name,
        embedding_dimension=embedding_dimension,
        embedding_dtype=_EMBEDDING_DTYPE,
        faiss_index_type=_FAISS_INDEX_TYPE,
        normalize_embeddings=True,
        document_count=document_count,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def write_index_and_metadata(
    index: faiss.IndexFlatIP,
    pmids: list[str],
    document_count: int,
    embedding_dimension: int,
    corpus_version: str,
    corpus_checksum: str,
    model_name: str,
    index_path: Path,
    metadata_path: Path,
    pmids_path: Path,
) -> EmbeddingMetadata:
    """Atomically write the FAISS index, its metadata, and its PMID ordering.

    Per ACR-001 (M3.1.3 architecture change request, raised during M3.2
    implementation): the FAISS index alone has no way to map a result
    vector position back to a PMID. ``pmids[i]`` must always correspond
    to FAISS vector position ``i`` — this is the contract downstream
    retrieval code (``DenseRetriever``) relies on, and it must not be
    reconstructed by depending on the BM25 artifact or by assuming
    ``corpus.jsonl`` file order is stable across runs.

    All three files are written to temporary paths first, then renamed
    into place only after every write succeeds. If any write fails, all
    temp files (and any previously-renamed final files from this call)
    are removed — an index, metadata record, or PMID ordering file
    without its two matching counterparts is exactly the partial-output
    state the M3.1.3 error boundary forbids.

    Args:
        index: The populated FAISS index.
        pmids: PMIDs in the exact same order as the vectors were added to
            ``index`` — i.e. ``pmids[i]`` is the PMID embedded at vector
            position ``i``. Callers must pass the same document ordering
            used to build ``index``.
        document_count: Number of documents/vectors in the index.
        embedding_dimension: Dimensionality of each embedding vector.
        corpus_version: Version of the source corpus (M3.1.1 metadata).
        corpus_checksum: SHA-256 checksum of the source corpus, copied
            verbatim from ``corpus.meta.json``, per the M3.1.3 corpus/index
            integrity contract.
        model_name: MedCPT checkpoint identifier used to generate the
            embeddings (from ``settings.medcpt_model_name``).
        index_path: Final destination path for ``faiss_index.bin``.
        metadata_path: Final destination path for ``embedding_metadata.json``.
        pmids_path: Final destination path for ``faiss_pmids.json``.

    Returns:
        The ``EmbeddingMetadata`` record that was written.

    Raises:
        ValueError: if ``len(pmids) != document_count``.
        OSError: if the index directory cannot be created or a temp file
            cannot be written.
    """
    if len(pmids) != document_count:
        raise ValueError(
            f"pmids length ({len(pmids)}) does not match document_count "
            f"({document_count}) — refusing to write a mismatched FAISS "
            "index/pmids pair."
        )

    index_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    pmids_path.parent.mkdir(parents=True, exist_ok=True)

    index_tmp = index_path.with_suffix(index_path.suffix + ".tmp")
    metadata_tmp = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    pmids_tmp = pmids_path.with_suffix(pmids_path.suffix + ".tmp")
    metadata = _metadata_from_inputs(
        document_count=document_count,
        embedding_dimension=embedding_dimension,
        corpus_version=corpus_version,
        corpus_checksum=corpus_checksum,
        model_name=model_name,
    )

    try:
        faiss.write_index(index, str(index_tmp))
        metadata_tmp.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")
        pmids_tmp.write_text(json.dumps(pmids, indent=2), encoding="utf-8")
        metadata_tmp.replace(metadata_path)
        pmids_tmp.replace(pmids_path)
        index_tmp.replace(index_path)
    except Exception:
        logger.error("Failed to write FAISS index, metadata, and pmids atomically")
        index_tmp.unlink(missing_ok=True)
        metadata_tmp.unlink(missing_ok=True)
        pmids_tmp.unlink(missing_ok=True)
        if index_path.exists():
            index_path.unlink()
        if metadata_path.exists():
            metadata_path.unlink()
        if pmids_path.exists():
            pmids_path.unlink()
        raise

    logger.info("Wrote FAISS index to %s", index_path)
    logger.info("Wrote embedding metadata to %s", metadata_path)
    logger.info("Wrote FAISS pmids ordering to %s", pmids_path)
    return metadata

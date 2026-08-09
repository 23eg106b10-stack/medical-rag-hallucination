"""M3.2 composition root — constructs a real, runnable HybridRetriever.

This module is the single place in the repository allowed to combine
``config.settings`` with the M3.2 retrieval components. Every component
it wires — ``BM25Retriever``, ``DenseRetriever``, ``InMemoryCorpusReader``,
``HybridRetriever`` — remains dependency-injected and settings-agnostic
by design; none of them import ``config.settings`` themselves, and this
module does not change that. It exists purely to do the one thing none
of those components are allowed to do on their own: read configuration,
load real artifacts and the MedCPT query encoder from disk, and assemble
a ``HybridRetriever`` instance ready to serve real queries.

No model, index, or corpus loading happens at import time. All
construction is deferred to ``build_hybrid_retriever()``, called
explicitly by whoever needs a live retriever (a script, a future
application entry point, or a test).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from config.settings import Settings, get_settings
from index.bm25_tokenizer import tokenize
from index.embedding_generator import embed_batch, load_encoder
from retrieval.corpus import InMemoryCorpusReader
from retrieval.dense import DenseRetriever
from retrieval.retriever import HybridRetriever
from retrieval.sparse import BM25Retriever

# Filenames not exposed as settings fields — same convention already
# established by index/build_bm25.py for bm25_metadata.json: these are
# fixed output names of the M3.1.3 FAISS pipeline
# (index.faiss_pipeline.write_index_and_metadata), not configuration.
_EMBEDDING_METADATA_FILENAME = "embedding_metadata.json"
_FAISS_PMIDS_FILENAME = "faiss_pmids.json"

# Corpus filename, matching the convention already used by
# index.bm25_pipeline.build_bm25_index and index.faiss_pipeline.
_CORPUS_FILENAME = "corpus.jsonl"


def _load_bm25_retriever(settings: Settings) -> BM25Retriever:
    """Load the frozen BM25 index into a ready-to-query retriever."""
    index_path = settings.indexes_dir / settings.bm25_index_filename
    return BM25Retriever.from_pickle(index_path, tokenizer=tokenize)


def _make_query_encode_fn() -> Callable[[str, Any, Any], np.ndarray]:
    """Build the single-query encode_fn required by DenseRetriever.

    Explicitly adapts the existing, unmodified ``embed_batch`` (which
    operates on a list of texts and returns a 2D array) to the single-text
    ``(query, tokenizer, model) -> np.ndarray`` contract DenseRetriever
    requires — rather than relying on embed_batch's undocumented
    tolerance for being called with a bare string. Preserves embed_batch's
    existing MedCPT embedding behavior (CLS-token extraction, 512-token
    truncation) unchanged; this wrapper only adapts batch shape.
    """

    def encode_fn(query: str, tokenizer: Any, model: Any) -> np.ndarray:
        vectors = embed_batch([query], tokenizer, model)
        return vectors[0]

    return encode_fn


def _load_dense_retriever(settings: Settings) -> DenseRetriever:
    """Load the MedCPT query encoder and the frozen FAISS artifacts."""
    tokenizer, model = load_encoder(settings.medcpt_query_encoder_model_name)

    index_path = settings.indexes_dir / settings.faiss_index_filename
    metadata_path = settings.indexes_dir / _EMBEDDING_METADATA_FILENAME
    pmids_path = settings.indexes_dir / _FAISS_PMIDS_FILENAME

    return DenseRetriever.from_disk(
        index_path=index_path,
        metadata_path=metadata_path,
        pmids_path=pmids_path,
        tokenizer=tokenizer,
        model=model,
        encode_fn=_make_query_encode_fn(),
    )


def _load_corpus_reader(settings: Settings) -> InMemoryCorpusReader:
    """Load the frozen corpus for PMID -> CorpusDocument resolution."""
    corpus_path = settings.corpus_dir / _CORPUS_FILENAME
    return InMemoryCorpusReader.from_jsonl(corpus_path)


def build_hybrid_retriever(settings: Settings | None = None) -> HybridRetriever:
    """Construct a real, runnable HybridRetriever from settings and artifacts.

    Loads the frozen BM25 pickle, the MedCPT query encoder and FAISS
    artifacts, and the corpus, then wires them into a HybridRetriever via
    dependency injection — mirroring exactly the constructor contract
    HybridRetriever already exposes. No retrieval component is modified
    or reimplemented here.

    Args:
        settings: Settings instance to use. Defaults to
            ``config.settings.get_settings()`` when omitted; accepting it
            explicitly keeps this function testable against injected,
            temporary settings without touching real artifacts.

    Returns:
        A HybridRetriever ready to serve ``.search(query)`` calls.

    Raises:
        FileNotFoundError: if the BM25, FAISS, or corpus artifacts are
            missing, propagated unchanged from the underlying component
            loaders.
        ValueError: if the FAISS artifact set is internally inconsistent,
            propagated unchanged from DenseRetriever.from_disk.
    """
    settings = settings or get_settings()

    sparse_retriever = _load_bm25_retriever(settings)
    dense_retriever = _load_dense_retriever(settings)
    corpus_reader = _load_corpus_reader(settings)

    return HybridRetriever(
        sparse_retriever=sparse_retriever,
        dense_retriever=dense_retriever,
        corpus_reader=corpus_reader,
    )

"""Minimal E2E generation runner.

Runs the complete question -> retrieval -> generation pipeline.
"""

from __future__ import annotations

import argparse
import time

import psutil
import torch

from config.settings import get_settings
from generation.bootstrap import build_generator
from retrieval.bootstrap import build_hybrid_retriever
from schemas.generation import GeneratedAnswer


def main() -> GeneratedAnswer | None:
    parser = argparse.ArgumentParser(description="Run the E2E RAG generation pipeline.")
    parser.add_argument("question", type=str, help="The medical question to answer.")
    args = parser.parse_args()

    settings = get_settings()

    t0_model = time.perf_counter()
    print("Building Generator...")
    generator = build_generator(settings)
    t_model = time.perf_counter() - t0_model
    print(f"Generator built in {t_model:.2f}s")

    t0_retriever = time.perf_counter()
    print("\nBuilding HybridRetriever...")
    retriever = build_hybrid_retriever(settings)
    t_retriever = time.perf_counter() - t0_retriever
    print(f"HybridRetriever built in {t_retriever:.2f}s")

    print(f"\nQuestion: {args.question}")

    print("\nRetrieving context...")
    t0_search = time.perf_counter()
    context = retriever.search(args.question)
    t_search = time.perf_counter() - t0_search

    print("\nRetrieved documents:")
    for i, doc in enumerate(context):
        print(f"[{i+1}] PMID: {doc.document.pmid} (RRF Score: {doc.rrf_score:.4f})")

    if not context:
        print("No context retrieved. Stopping.")
        return None

    print("\nGenerating answer...")
    t0_gen = time.perf_counter()
    answer = generator.generate(args.question, context)
    t_gen = time.perf_counter() - t0_gen

    print("\nGenerated Answer:")
    print("-" * 80)
    print(answer.answer_text)
    print("-" * 80)

    # Performance & resource metrics
    tokenizer = getattr(generator, "_tokenizer", None)
    prompt_builder = getattr(generator, "_context_builder", None)
    if tokenizer and prompt_builder:
        prompt_str = prompt_builder(args.question, context)
        prompt_tokens: int | str = len(tokenizer.encode(prompt_str))
    else:
        prompt_tokens = "N/A"

    new_tokens: int | str = len(tokenizer.encode(answer.answer_text)) if tokenizer else "N/A"
    if isinstance(new_tokens, int) and t_gen > 0:
        throughput: str = f"{new_tokens / t_gen:.2f} tok/s"
    else:
        throughput = "N/A"

    vm = psutil.virtual_memory()
    proc = psutil.Process()
    vram_alloc = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
    vram_res = torch.cuda.memory_reserved() / (1024 * 1024) if torch.cuda.is_available() else 0.0

    print("\nExecution Metrics:")
    print(f"  Model Load Time:        {t_model:.2f}s")
    print(f"  Retriever Load Time:    {t_retriever:.2f}s")
    print(f"  Retrieval Time:         {t_search:.2f}s")
    print(f"  Generation Time:        {t_gen:.2f}s")
    print(f"  Prompt Tokens:          {prompt_tokens}")
    print(f"  New Tokens:             {new_tokens}")
    print(f"  Generation Throughput:  {throughput}")
    print(f"  Process RSS:            {proc.memory_info().rss / (1024 * 1024):.1f} MB")
    print(
        f"  Host RAM Available:     {vm.available / (1024**3):.2f} GB / "
        f"{vm.total / (1024**3):.2f} GB ({vm.percent}% used)"
    )
    print(f"  GPU VRAM Alloc / Res:   {vram_alloc:.1f} MB / {vram_res:.1f} MB")

    return answer


if __name__ == "__main__":
    main()

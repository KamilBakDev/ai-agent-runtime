"""Offline RAG evaluation: retrieval hit-rate + keyword-overlap faithfulness proxy.

Deliberately LLM-free so it runs in CI with zero cost/network dependency. Two
metrics per the spec's "faithfulness, relevance" requirement:

- retrieval_hit_rate: did the expected source document show up in the top-k results?
  (relevance proxy)
- keyword_faithfulness_rate: do all expected keywords appear somewhere in the
  retrieved text? (a cheap stand-in for "the answer would be grounded in retrieved
  context" -- a real LLM-as-judge could replace this once a real LLM is configured)
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from rag.retriever import HybridRetriever

DATASET_PATH = Path(__file__).parent / "qa_dataset.json"


async def evaluate(top_k: int = 3) -> dict:
    dataset = json.loads(DATASET_PATH.read_text())
    retriever = HybridRetriever()
    results = []
    try:
        for item in dataset:
            chunks = await retriever.retrieve(item["question"], top_k=top_k)
            sources = [c.source for c in chunks]
            retrieved_text = " ".join(c.text.lower() for c in chunks)
            source_hit = item["expected_source"] in sources
            keyword_hit = all(
                kw.lower() in retrieved_text for kw in item.get("expected_keywords", [])
            )
            results.append(
                {
                    "question": item["question"],
                    "source_hit": source_hit,
                    "keyword_hit": keyword_hit,
                    "retrieved_sources": sources,
                }
            )
    finally:
        await retriever.close()

    n = len(dataset) or 1
    return {
        "num_questions": len(dataset),
        "retrieval_hit_rate": sum(r["source_hit"] for r in results) / n,
        "keyword_faithfulness_rate": sum(r["keyword_hit"] for r in results) / n,
        "results": results,
    }


def _main() -> None:
    summary = asyncio.run(evaluate())
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "results"}, indent=2
        )
    )
    for r in summary["results"]:
        status = "OK  " if r["source_hit"] and r["keyword_hit"] else "MISS"
        print(f"[{status}] {r['question']}  -> {r['retrieved_sources']}")


if __name__ == "__main__":
    _main()

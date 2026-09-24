"""RAG pipeline tests: chunking, ingest -> Qdrant, hybrid retrieval, citations, eval.

Runs against the real Qdrant container (infra/docker-compose.yml) with the offline
fake embeddings provider -- no network calls, no API keys. Ingestion is idempotent
(deterministic point IDs), so this reuses -- and leaves in place -- the same
``legal_docs`` collection the demo/UI reads from.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.citations import format_context_with_citations
from rag.ingest import chunk_text, ingest_directory
from rag.retriever import HybridRetriever

SAMPLE_DOCS = Path(__file__).resolve().parent.parent / "data" / "sample_docs"


def test_chunk_text_respects_size_and_overlap():
    text = "word " * 500
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)


def test_chunk_text_empty_input():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


@pytest.fixture(scope="module", autouse=True)
async def ingested_corpus():
    count = await ingest_directory(SAMPLE_DOCS)
    assert count > 0


@pytest.mark.asyncio
async def test_hybrid_retrieval_finds_relevant_chunk():
    retriever = HybridRetriever()
    try:
        chunks = await retriever.retrieve("force majeure notice period", top_k=3)
    finally:
        await retriever.close()

    assert len(chunks) > 0
    assert any("force_majeure" in c.source for c in chunks)


@pytest.mark.asyncio
async def test_citations_are_numbered_and_reference_source():
    retriever = HybridRetriever()
    try:
        chunks = await retriever.retrieve("indemnification liability cap", top_k=3)
    finally:
        await retriever.close()

    context, citations = format_context_with_citations(chunks)
    assert "[1]" in context
    assert len(citations) == len(chunks)
    assert citations[0]["id"] == 1
    assert all("source" in c for c in citations)


@pytest.mark.asyncio
async def test_search_cases_tool():
    from agents.tools.legal_tools import search_cases

    results = await search_cases("termination for cause cure period", top_k=3)
    assert len(results) > 0
    assert all({"text", "source", "chunk_index", "score"} <= r.keys() for r in results)


@pytest.mark.asyncio
async def test_extract_citations_tool():
    from agents.tools.legal_tools import extract_citations

    results = await extract_citations(["indemnification_clause_memo.md"])
    assert len(results) > 0
    assert all(r["source"] == "indemnification_clause_memo.md" for r in results)


@pytest.mark.asyncio
async def test_eval_dataset_hits_expected_sources():
    from rag.evals.evaluate import evaluate

    summary = await evaluate(top_k=3)
    assert summary["retrieval_hit_rate"] >= 0.8

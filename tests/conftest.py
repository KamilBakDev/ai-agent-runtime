"""Shared pytest fixtures.

Forces the offline ``fake`` LLM/embeddings providers before anything imports
``agents.config``, so the whole suite runs without network access or API keys.
Integration tests still talk to the real Postgres/Qdrant containers started by
``infra/docker-compose.yml`` -- only the LLM is faked.
"""

from __future__ import annotations

import os

os.environ.setdefault("LLM_PROVIDER", "fake")
os.environ.setdefault("EMBEDDINGS_PROVIDER", "fake")

from pathlib import Path

import httpx
import pytest_asyncio

SAMPLE_DOCS = Path(__file__).resolve().parent.parent / "data" / "sample_docs"


@pytest_asyncio.fixture(scope="session", autouse=True)
async def ingested_corpus():
    """Ingest the sample corpus into Qdrant once before any test runs.

    Session-scoped and autouse so it's not an implicit ordering dependency on
    whichever test module happens to run first (ingestion is idempotent, so
    re-running it locally against an already-populated collection is a no-op).
    """
    from rag.ingest import ingest_directory

    await ingest_directory(SAMPLE_DOCS)


@pytest_asyncio.fixture
async def fake_llm():
    from agents.llm import get_chat_model

    return get_chat_model(provider="fake")


@pytest_asyncio.fixture
async def memory_graph(fake_llm):
    from agents.memory.checkpoint import memory_checkpointer
    from agents.orchestrator import compile_graph

    return compile_graph(memory_checkpointer(), llm=fake_llm)


@pytest_asyncio.fixture
async def api_client():
    from api.app import app

    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=transport, base_url="http://test", timeout=30
    ) as client:
        yield client

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

import httpx
import pytest_asyncio


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
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test", timeout=30
        ) as client:
            yield client

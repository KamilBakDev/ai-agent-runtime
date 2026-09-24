"""A2A tests: A2AClient making a real HTTP round trip to /a2a/invoke over ASGI
transport (in-process, no real socket, but a genuine HTTP request/response --
not a bare function call renamed 'A2A')."""

from __future__ import annotations

import json

import httpx
import pytest

from agents.config import get_settings
from agents.tools.a2a_client import A2AClient


def _client_for_app() -> A2AClient:
    from api.app import app

    settings = get_settings().model_copy(
        update={
            "a2a_registry_raw": json.dumps(
                {"researcher": "http://test", "coder": "http://test", "reviewer": "http://test"}
            )
        }
    )
    return A2AClient(settings=settings, transport=httpx.ASGITransport(app=app))


@pytest.mark.asyncio
async def test_a2a_invoke_researcher():
    client = _client_for_app()
    result = await client.invoke_agent("researcher", {"query": "force majeure notice period"})
    assert result["agent"] == "researcher"
    assert result["output"]["research_notes"]


@pytest.mark.asyncio
async def test_a2a_invoke_coder():
    client = _client_for_app()
    result = await client.invoke_agent(
        "coder", {"research_notes": "Notice must be given within 10 business days."}
    )
    assert result["agent"] == "coder"
    assert result["output"]["code_draft"]


@pytest.mark.asyncio
async def test_a2a_invoke_reviewer_approved():
    client = _client_for_app()
    result = await client.invoke_agent(
        "reviewer", {"review_status": "approved", "code_draft": "SELECT 1;"}
    )
    assert result["output"]["final_output"] == "SELECT 1;"


@pytest.mark.asyncio
async def test_a2a_endpoint_404s_for_unhandled_agent_name():
    from api.app import app

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as raw_client:
        r = await raw_client.post("/a2a/invoke", json={"agent": "nonexistent", "task": {}})
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_a2a_invoke_agent_not_in_registry_raises_locally():
    client = A2AClient(settings=get_settings().model_copy(update={"a2a_registry_raw": "{}"}))
    with pytest.raises(ValueError):
        await client.invoke_agent("researcher", {"query": "x"})

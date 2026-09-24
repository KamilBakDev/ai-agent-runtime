"""Integration tests for the FastAPI layer.

Exercises the app over ASGI transport (in-process, no network) with the offline
fake LLM, against the real Postgres started by ``infra/docker-compose.yml``.
"""

from __future__ import annotations

import json

import pytest


@pytest.mark.asyncio
async def test_health(api_client):
    r = await api_client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_create_and_list_sessions(api_client):
    r = await api_client.post("/sessions", json={"title": "pytest session"})
    assert r.status_code == 200
    session = r.json()
    assert session["title"] == "pytest session"
    assert session["status"] == "active"

    r = await api_client.get("/sessions")
    assert r.status_code == 200
    ids = [s["id"] for s in r.json()]
    assert session["id"] in ids


@pytest.mark.asyncio
async def test_get_missing_session_404s(api_client):
    r = await api_client.get("/sessions/00000000-0000-0000-0000-000000000000")
    assert r.status_code == 404


async def _run_chat_to_pending(api_client, message: str) -> str:
    session_id = None
    async with api_client.stream("POST", "/chat", json={"message": message}) as r:
        assert r.status_code == 200
        async for line in r.aiter_lines():
            if line.startswith("data:"):
                data = json.loads(line[len("data:") :].strip())
                if "session_id" in data and session_id is None:
                    session_id = data["session_id"]
    assert session_id is not None
    return session_id


@pytest.mark.asyncio
async def test_chat_reaches_pending_review(api_client):
    session_id = await _run_chat_to_pending(api_client, "Draft SQL for force majeure contracts.")

    r = await api_client.get(f"/sessions/{session_id}/pending_actions")
    assert r.status_code == 200
    pending = r.json()
    assert len(pending) == 1
    assert pending[0]["node_name"] == "reviewer"
    assert pending[0]["status"] == "pending"

    r = await api_client.get(f"/sessions/{session_id}")
    assert r.status_code == 200
    detail = r.json()
    assert detail["graph_next"] == ["reviewer"]
    assert len(detail["messages"]) >= 3


@pytest.mark.asyncio
async def test_resume_approve_completes_turn(api_client):
    session_id = await _run_chat_to_pending(api_client, "Draft SQL for contracts.")

    r = await api_client.post(f"/sessions/{session_id}/resume", json={"review_status": "approved"})
    assert r.status_code == 200
    result = r.json()
    assert result["status"] == "completed"
    assert result["final_output"]

    r = await api_client.get(f"/sessions/{session_id}/pending_actions")
    assert r.json() == []


@pytest.mark.asyncio
async def test_resume_reject_reopens_pending_action(api_client):
    session_id = await _run_chat_to_pending(api_client, "Draft SQL for contracts.")

    r = await api_client.post(
        f"/sessions/{session_id}/resume",
        json={"review_status": "rejected", "review_feedback": "Add a WHERE clause"},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "awaiting_review"

    r = await api_client.get(f"/sessions/{session_id}/pending_actions")
    assert len(r.json()) == 1


@pytest.mark.asyncio
async def test_checkpoints_list_and_restore(api_client):
    session_id = await _run_chat_to_pending(api_client, "Draft SQL for contracts.")
    await api_client.post(f"/sessions/{session_id}/resume", json={"review_status": "approved"})

    r = await api_client.get(f"/checkpoints/{session_id}")
    assert r.status_code == 200
    checkpoints = r.json()
    assert len(checkpoints) >= 4

    mid_checkpoint = next(c for c in checkpoints if c["current_step"] == "coder")
    r = await api_client.post(
        f"/checkpoints/{session_id}/restore", json={"checkpoint_id": mid_checkpoint["checkpoint_id"]}
    )
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_branch_creates_new_session(api_client):
    session_id = await _run_chat_to_pending(api_client, "Draft SQL for contracts.")
    await api_client.post(f"/sessions/{session_id}/resume", json={"review_status": "approved"})

    checkpoints = (await api_client.get(f"/checkpoints/{session_id}")).json()
    mid_checkpoint = next(c for c in checkpoints if c["current_step"] == "coder")

    r = await api_client.post(
        f"/sessions/{session_id}/branch",
        json={"checkpoint_id": mid_checkpoint["checkpoint_id"], "title": "Branch"},
    )
    assert r.status_code == 200
    branch = r.json()
    assert branch["parent_session_id"] == session_id

    r = await api_client.get(f"/sessions/{branch['id']}")
    assert r.status_code == 200
    assert r.json()["current_step"] == "coder"


@pytest.mark.asyncio
async def test_query_sql_tool_allows_select(api_client):
    r = await api_client.post("/tools/query_sql", json={"statement": "SELECT 1 AS one"})
    assert r.status_code == 200
    assert r.json() == [{"one": 1}]


@pytest.mark.asyncio
async def test_query_sql_tool_rejects_write_statement(api_client):
    r = await api_client.post(
        "/tools/query_sql", json={"statement": "DELETE FROM sessions WHERE 1=1"}
    )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_list_tools(api_client):
    r = await api_client.get("/tools")
    assert r.status_code == 200
    assert "query_sql" in r.json()

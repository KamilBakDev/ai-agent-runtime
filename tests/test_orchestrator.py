"""Unit tests for the LangGraph orchestrator: HITL interrupt, resume, and branching.

All of these run against ``MemorySaver`` + the offline fake LLM -- no Postgres, no
network calls.
"""

from __future__ import annotations

import uuid

import pytest

from agents.orchestrator import initial_state


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


@pytest.mark.asyncio
async def test_graph_pauses_before_reviewer(memory_graph):
    thread_id = str(uuid.uuid4())
    config = _config(thread_id)
    state = initial_state(thread_id, "Research and draft SQL for contracts.")

    await memory_graph.ainvoke(state, config)
    snapshot = await memory_graph.aget_state(config)

    assert snapshot.next == ("reviewer",)
    assert snapshot.values["code_draft"]
    assert snapshot.values["research_notes"]


@pytest.mark.asyncio
async def test_resume_approved_produces_final_output(memory_graph):
    thread_id = str(uuid.uuid4())
    config = _config(thread_id)
    state = initial_state(thread_id, "Draft SQL for contracts.")
    await memory_graph.ainvoke(state, config)

    await memory_graph.aupdate_state(config, {"review_status": "approved"})
    await memory_graph.ainvoke(None, config)

    snapshot = await memory_graph.aget_state(config)
    assert snapshot.next == ()
    assert snapshot.values["final_output"] == snapshot.values["code_draft"]


@pytest.mark.asyncio
async def test_reject_loops_back_to_coder(memory_graph):
    thread_id = str(uuid.uuid4())
    config = _config(thread_id)
    state = initial_state(thread_id, "Draft SQL for contracts.")
    await memory_graph.ainvoke(state, config)

    await memory_graph.aupdate_state(
        config, {"review_status": "rejected", "review_feedback": "Add a WHERE clause."}
    )
    await memory_graph.ainvoke(None, config)

    snapshot = await memory_graph.aget_state(config)
    assert snapshot.next == ("reviewer",)
    assert snapshot.values["revision_count"] == 1


@pytest.mark.asyncio
async def test_rejection_is_bounded_by_max_revisions(memory_graph):
    thread_id = str(uuid.uuid4())
    config = _config(thread_id)
    state = initial_state(thread_id, "Draft SQL for contracts.")
    await memory_graph.ainvoke(state, config)

    for _ in range(5):
        snapshot = await memory_graph.aget_state(config)
        if snapshot.next == ():
            break
        await memory_graph.aupdate_state(
            config, {"review_status": "rejected", "review_feedback": "Try again."}
        )
        await memory_graph.ainvoke(None, config)

    final = await memory_graph.aget_state(config)
    assert final.values["revision_count"] <= 3


@pytest.mark.asyncio
async def test_edited_output_is_used_as_final(memory_graph):
    thread_id = str(uuid.uuid4())
    config = _config(thread_id)
    state = initial_state(thread_id, "Draft SQL for contracts.")
    await memory_graph.ainvoke(state, config)

    await memory_graph.aupdate_state(
        config, {"review_status": "edited", "edited_output": "SELECT 1;"}
    )
    await memory_graph.ainvoke(None, config)

    snapshot = await memory_graph.aget_state(config)
    assert snapshot.values["final_output"] == "SELECT 1;"


@pytest.mark.asyncio
async def test_checkpoint_history_is_recorded(memory_graph):
    thread_id = str(uuid.uuid4())
    config = _config(thread_id)
    state = initial_state(thread_id, "Draft SQL for contracts.")
    await memory_graph.ainvoke(state, config)
    await memory_graph.aupdate_state(config, {"review_status": "approved"})
    await memory_graph.ainvoke(None, config)

    history = [s async for s in memory_graph.aget_state_history(config)]
    assert len(history) >= 4
    steps = [s.values.get("current_step") for s in history]
    assert "coder" in steps
    assert "reviewer" in steps

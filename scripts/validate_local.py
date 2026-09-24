#!/usr/bin/env python3
"""Validate that every local component of the stack is reachable and working.

Checks (in order): Postgres reachable, Qdrant reachable, and an end-to-end agent
run (fake LLM + in-memory checkpointer -- no external API needed) that reaches
the human-in-the-loop pause and resumes to completion. Exits non-zero if any
check fails, so it's usable as a CI/CD smoke gate too.

Usage: python scripts/validate_local.py
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from agents.config import get_settings


async def check_postgres() -> tuple[bool, str]:
    settings = get_settings()
    try:
        engine = create_async_engine(settings.database_url)
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        await engine.dispose()
        return True, f"reachable at {settings.database_url.split('@')[-1]}"
    except Exception as exc:  # noqa: BLE001 -- this is a diagnostic script
        return False, str(exc)


async def check_qdrant() -> tuple[bool, str]:
    settings = get_settings()
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{settings.qdrant_url}/collections")
            resp.raise_for_status()
        return True, f"reachable at {settings.qdrant_url}"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


async def check_agent_run() -> tuple[bool, str]:
    try:
        from agents.llm import get_chat_model
        from agents.memory.checkpoint import memory_checkpointer
        from agents.orchestrator import compile_graph, initial_state

        llm = get_chat_model(provider="fake")
        checkpointer = memory_checkpointer()
        graph = compile_graph(checkpointer, llm=llm)

        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        state = initial_state(thread_id, "Draft SQL for a validation smoke test.")
        await graph.ainvoke(state, config)

        snapshot = await graph.aget_state(config)
        if snapshot.next != ("reviewer",):
            return False, f"expected to pause before reviewer, got next={snapshot.next}"

        await graph.aupdate_state(config, {"review_status": "approved"})
        await graph.ainvoke(None, config)
        final = await graph.aget_state(config)
        if not final.values.get("final_output"):
            return False, "resume did not produce a final_output"

        return True, "researcher -> coder -> reviewer (HITL pause/resume) all worked"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


async def main() -> int:
    checks = [
        ("Postgres", check_postgres()),
        ("Qdrant", check_qdrant()),
        ("Agent run (fake LLM, HITL pause/resume)", check_agent_run()),
    ]

    all_ok = True
    for name, coro in checks:
        ok, detail = await coro
        icon = "✅" if ok else "❌"
        print(f"{icon} {name}: {detail}")
        all_ok = all_ok and ok

    print()
    if all_ok:
        print("✅ All components working locally.")
        return 0
    print("❌ One or more components failed. See above.")
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

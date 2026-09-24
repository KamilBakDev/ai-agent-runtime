"""Checkpointer factories: Postgres-backed for real runs, in-memory for fast tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from agents.config import get_settings


@asynccontextmanager
async def postgres_checkpointer(conn_string: str | None = None) -> AsyncIterator[AsyncPostgresSaver]:
    """Yield a Postgres-backed checkpointer with its tables ensured to exist.

    Must be used as ``async with``; the underlying connection pool is closed on exit.
    """
    settings = get_settings()
    conn_string = conn_string or settings.checkpoint_database_url
    async with AsyncPostgresSaver.from_conn_string(conn_string) as saver:
        await saver.setup()
        yield saver


def memory_checkpointer() -> MemorySaver:
    """An in-process checkpointer for unit tests / offline runs (no Postgres needed)."""
    return MemorySaver()

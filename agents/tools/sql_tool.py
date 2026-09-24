"""Read-only SQL query tool, exposed to agents (directly and via MCP)."""

from __future__ import annotations

from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

_FORBIDDEN_KEYWORDS = (
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "truncate",
    "grant",
    "revoke",
    "create",
)


class UnsafeQueryError(ValueError):
    pass


def _assert_read_only(statement: str) -> None:
    lowered = statement.strip().lower()
    if not lowered.startswith("select") and not lowered.startswith("with"):
        raise UnsafeQueryError("Only SELECT (or read-only WITH) statements are allowed.")
    if any(f" {kw} " in f" {lowered} " for kw in _FORBIDDEN_KEYWORDS):
        raise UnsafeQueryError("Statement contains a forbidden write/DDL keyword.")


async def query_sql(db: AsyncSession, statement: str, limit: int = 200) -> list[dict[str, Any]]:
    """Run a read-only SQL query and return rows as dicts, capped at ``limit`` rows."""
    _assert_read_only(statement)
    result = await db.execute(text(statement))
    rows = result.mappings().fetchmany(limit)
    return [dict(row) for row in rows]

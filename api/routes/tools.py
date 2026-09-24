"""Tool registry + direct-invoke endpoints (the same tools the MCP server exposes)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from agents.tools.sql_tool import UnsafeQueryError, query_sql
from db.database import get_db

router = APIRouter(prefix="/tools", tags=["tools"])

TOOL_REGISTRY: dict[str, dict[str, Any]] = {
    "query_sql": {
        "description": "Run a read-only SQL SELECT against the application database.",
        "input_schema": {"statement": "str", "limit": "int (optional, default 200)"},
    },
    "search_cases": {
        "description": "Hybrid vector+BM25 search over the ingested legal document corpus.",
        "input_schema": {"query": "str", "top_k": "int (optional, default 5)"},
    },
    "extract_citations": {
        "description": "Extract source citations for a set of retrieved document chunks.",
        "input_schema": {"doc_ids": "list[str]"},
    },
}


@router.get("")
async def list_tools() -> dict[str, dict[str, Any]]:
    return TOOL_REGISTRY


class SqlQueryRequest(BaseModel):
    statement: str
    limit: int = 200


@router.post("/query_sql")
async def invoke_query_sql(
    payload: SqlQueryRequest, db: AsyncSession = Depends(get_db)
) -> list[dict[str, Any]]:
    try:
        return await query_sql(db, payload.statement, payload.limit)
    except UnsafeQueryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

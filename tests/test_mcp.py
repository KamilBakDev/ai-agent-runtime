"""MCP server/client round-trip tests: spawns the real stdio server subprocess."""

from __future__ import annotations

import pytest

from agents.tools.mcp_client import MCPToolClient


@pytest.mark.asyncio
async def test_mcp_lists_expected_tools():
    async with MCPToolClient() as client:
        tools = await client.list_tools()
    assert set(tools) == {"search_cases", "extract_citations", "query_sql"}


@pytest.mark.asyncio
async def test_mcp_search_cases_round_trip():
    async with MCPToolClient() as client:
        result = await client.call_tool(
            "search_cases", {"query": "force majeure notice period", "top_k": 2}
        )
    assert len(result) == 2
    assert all({"text", "source", "chunk_index", "score"} <= r.keys() for r in result)


@pytest.mark.asyncio
async def test_mcp_query_sql_round_trip():
    async with MCPToolClient() as client:
        result = await client.call_tool("query_sql", {"statement": "SELECT 1 AS one"})
    assert result == [{"one": 1}]


@pytest.mark.asyncio
async def test_mcp_query_sql_rejects_write_statement():
    async with MCPToolClient() as client:
        with pytest.raises(RuntimeError):
            await client.call_tool("query_sql", {"statement": "DELETE FROM sessions"})


@pytest.mark.asyncio
async def test_mcp_extract_citations_round_trip():
    async with MCPToolClient() as client:
        result = await client.call_tool(
            "extract_citations", {"doc_ids": ["indemnification_clause_memo.md"]}
        )
    assert len(result) > 0
    assert all(r["source"] == "indemnification_clause_memo.md" for r in result)

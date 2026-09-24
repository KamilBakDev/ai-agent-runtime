"""MCP (Model Context Protocol) server exposing the same three tools as the /tools
API: search_cases, extract_citations, query_sql. Run standalone over stdio:

    python -m agents.tools.mcp_server

Any MCP-compatible client (Claude Desktop, an MCP Inspector, or our own
``agents.tools.mcp_client``) can then discover and call these tools.
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from agents.tools import legal_tools, sql_tool
from db.database import AsyncSessionLocal

server = MCPServer(
    name="ai-agent-runtime-tools",
    instructions=(
        "Legal-research and data tools for the AI Agent Runtime demo: hybrid search over "
        "an ingested case-law/contract corpus, citation lookup, and read-only SQL access."
    ),
)


@server.tool()
async def search_cases(query: str, top_k: int = 5) -> list[dict]:
    """Hybrid vector+BM25 search over the ingested legal document corpus."""
    return await legal_tools.search_cases(query, top_k)


@server.tool()
async def extract_citations(doc_ids: list[str]) -> list[dict]:
    """Fetch all indexed chunks belonging to the given source document filenames."""
    return await legal_tools.extract_citations(doc_ids)


@server.tool()
async def query_sql(statement: str, limit: int = 200) -> list[dict]:
    """Run a read-only SQL SELECT statement against the application database."""
    async with AsyncSessionLocal() as db:
        return await sql_tool.query_sql(db, statement, limit)


if __name__ == "__main__":
    server.run(transport="stdio")

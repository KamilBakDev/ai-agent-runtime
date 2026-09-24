"""A thin async MCP client that launches ``agents.tools.mcp_server`` as a stdio
subprocess and calls its tools. This is how an agent (or the demo notebook) is
meant to reach the MCP tools -- as opposed to the /tools HTTP routes, which call
the same underlying functions directly for convenience from the API/UI."""

from __future__ import annotations

import json
import sys
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPToolClient:
    """Usage: ``async with MCPToolClient() as client: await client.call_tool(...)``"""

    def __init__(self) -> None:
        self._stack = AsyncExitStack()
        self._session: ClientSession | None = None

    async def __aenter__(self) -> MCPToolClient:
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "agents.tools.mcp_server"]
        )
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self._stack.aclose()

    async def list_tools(self) -> list[str]:
        assert self._session is not None
        result = await self._session.list_tools()
        return [t.name for t in result.tools]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        assert self._session is not None
        result = await self._session.call_tool(name, arguments)
        if result.is_error:
            raise RuntimeError(f"MCP tool {name!r} failed: {result.content}")
        # All three tools here return list[dict]; this SDK serializes a list return
        # as one text block *per item*, so re-collecting every block always gives
        # back the original list (including the empty- and single-item cases).
        return [json.loads(c.text) for c in result.content if c.type == "text"]

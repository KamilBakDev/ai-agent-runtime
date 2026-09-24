"""Agent-to-Agent (A2A) client: invokes another agent's HTTP endpoint by name,
looked up in the registry (``A2A_REGISTRY`` env var: ``{name: base_url}``)."""

from __future__ import annotations

from typing import Any

import httpx

from agents.config import Settings, get_settings


class A2AClient:
    def __init__(
        self,
        settings: Settings | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._transport = transport  # override for tests (e.g. httpx.ASGITransport)

    async def invoke_agent(self, agent_name: str, task: dict[str, Any]) -> dict[str, Any]:
        registry = self.settings.a2a_registry
        if agent_name not in registry:
            raise ValueError(
                f"Unknown agent {agent_name!r} in A2A registry (known: {sorted(registry)})"
            )
        base_url = registry[agent_name].rstrip("/")

        async with httpx.AsyncClient(timeout=30, transport=self._transport) as client:
            resp = await client.post(
                f"{base_url}/a2a/invoke", json={"agent": agent_name, "task": task}
            )
            resp.raise_for_status()
            return resp.json()

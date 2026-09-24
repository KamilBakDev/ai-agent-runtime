"""Shared FastAPI dependencies."""

from __future__ import annotations

from fastapi import Request

from agents.service import AgentService


def get_agent_service(request: Request) -> AgentService:
    return request.app.state.agent_service

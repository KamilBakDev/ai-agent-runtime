"""FastAPI application entrypoint."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agents.config import get_settings
from agents.memory.checkpoint import postgres_checkpointer
from agents.observability import metrics
from agents.service import AgentService
from agents.tools.legal_tools import search_cases
from api.middleware.tracing import setup_observability
from api.routes import a2a, chat, checkpoints, sessions, tools

logger = structlog.get_logger("api.app")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    async with postgres_checkpointer(settings.checkpoint_database_url) as checkpointer:
        app.state.checkpointer = checkpointer
        app.state.agent_service = AgentService(checkpointer, retrieve=search_cases)
        logger.info("startup.complete", llm_provider=settings.llm_provider)
        yield
    logger.info("shutdown.complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Agent Runtime",
        description=(
            "Production-grade multi-agent runtime: LangGraph orchestration, Postgres "
            "checkpoints with time-travel, human-in-the-loop review, MCP tools, A2A, RAG."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    setup_observability(app)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(chat.router)
    app.include_router(sessions.router)
    app.include_router(checkpoints.router)
    app.include_router(tools.router)
    app.include_router(a2a.router)

    @app.get("/health", tags=["health"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics", tags=["health"])
    async def metrics_snapshot() -> dict:
        return metrics.snapshot()

    return app


app = create_app()

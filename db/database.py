"""Async SQLAlchemy engine + session factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from agents.config import get_settings

_settings = get_settings()

# NullPool: no connection is held open across requests/event loops. This app is
# low-throughput enough that per-request connections are fine, and it sidesteps
# asyncpg connections getting bound to whatever event loop happened to open them
# (a real problem across pytest-asyncio's per-test loops).
engine = create_async_engine(_settings.database_url, poolclass=NullPool, future=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


@asynccontextmanager
async def get_session_ctx() -> AsyncIterator[AsyncSession]:
    async with AsyncSessionLocal() as session:
        yield session


async def get_db() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency."""
    async with AsyncSessionLocal() as session:
        yield session

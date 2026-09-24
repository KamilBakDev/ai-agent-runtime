"""POST /chat -- run (or continue) an agent turn, streamed over SSE."""

from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from agents.service import AgentService
from api.deps import get_agent_service
from api.schemas import ChatRequest
from db.database import get_db
from db.models import Session as SessionRow

router = APIRouter(tags=["chat"])


@router.post("/chat")
async def chat(
    payload: ChatRequest,
    db: AsyncSession = Depends(get_db),
    service: AgentService = Depends(get_agent_service),
) -> EventSourceResponse:
    session_id = payload.session_id
    if not session_id:
        session = SessionRow(id=uuid.uuid4(), title=payload.message[:60])
        db.add(session)
        await db.commit()
        session_id = str(session.id)

    async def event_generator():
        yield {"event": "session", "data": json.dumps({"session_id": session_id})}
        async for event in service.run_turn_stream(db, session_id, payload.message):
            yield {"event": event["type"], "data": json.dumps(event, default=str)}

    return EventSourceResponse(event_generator())

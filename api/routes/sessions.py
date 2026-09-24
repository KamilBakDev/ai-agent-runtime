"""Session CRUD, pending actions (HITL), resume, and branching."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agents.service import AgentService
from api.deps import get_agent_service
from api.schemas import (
    BranchOut,
    BranchRequest,
    MessageOut,
    PendingActionOut,
    ResumeRequest,
    SessionCreate,
    SessionDetail,
    SessionSummary,
    TurnResultOut,
)
from db.database import get_db
from db.models import Message
from db.models import Session as SessionRow

router = APIRouter(prefix="/sessions", tags=["sessions"])


def _to_summary(session: SessionRow) -> SessionSummary:
    return SessionSummary(
        id=str(session.id),
        title=session.title,
        status=session.status,
        created_at=session.created_at,
        updated_at=session.updated_at,
        parent_session_id=str(session.parent_session_id) if session.parent_session_id else None,
    )


@router.post("", response_model=SessionSummary)
async def create_session(
    payload: SessionCreate, db: AsyncSession = Depends(get_db)
) -> SessionSummary:
    session = SessionRow(id=uuid.uuid4(), title=payload.title)
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return _to_summary(session)


@router.get("", response_model=list[SessionSummary])
async def list_sessions(db: AsyncSession = Depends(get_db)) -> list[SessionSummary]:
    result = await db.execute(select(SessionRow).order_by(SessionRow.created_at.desc()))
    return [_to_summary(s) for s in result.scalars().all()]


@router.get("/{session_id}", response_model=SessionDetail)
async def get_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    service: AgentService = Depends(get_agent_service),
) -> SessionDetail:
    session = await db.get(SessionRow, uuid.UUID(session_id))
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    msg_result = await db.execute(
        select(Message).where(Message.session_id == session.id).order_by(Message.created_at)
    )
    messages = [MessageOut.model_validate(m, from_attributes=True) for m in msg_result.scalars()]

    try:
        graph_state = await service.get_state(session_id)
    except Exception:
        graph_state = {"next": [], "values": {}}

    return SessionDetail(
        id=str(session.id),
        title=session.title,
        status=session.status,
        created_at=session.created_at,
        updated_at=session.updated_at,
        parent_session_id=str(session.parent_session_id) if session.parent_session_id else None,
        messages=messages,
        graph_next=graph_state.get("next", []),
        current_step=graph_state.get("values", {}).get("current_step"),
    )


@router.get("/{session_id}/pending_actions", response_model=list[PendingActionOut])
async def get_pending_actions(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    service: AgentService = Depends(get_agent_service),
) -> list[PendingActionOut]:
    pending = await service.pending_action(db, session_id)
    if pending is None:
        return []
    return [
        PendingActionOut(
            id=str(pending.id),
            node_name=pending.node_name,
            payload=pending.payload,
            status=pending.status,
            created_at=pending.created_at,
        )
    ]


@router.post("/{session_id}/resume", response_model=TurnResultOut)
async def resume_session(
    session_id: str,
    payload: ResumeRequest,
    db: AsyncSession = Depends(get_db),
    service: AgentService = Depends(get_agent_service),
) -> TurnResultOut:
    result = await service.resume_turn(db, session_id, payload.model_dump(exclude_none=True))
    return TurnResultOut(
        session_id=result.session_id,
        status=result.status,
        new_messages=[MessageOut(**m) for m in result.new_messages],
        final_output=result.final_output,
        pending_action_id=result.pending_action_id,
    )


@router.post("/{session_id}/branch", response_model=BranchOut)
async def branch_session(
    session_id: str,
    payload: BranchRequest,
    db: AsyncSession = Depends(get_db),
    service: AgentService = Depends(get_agent_service),
) -> BranchOut:
    new_session = await service.branch_from_checkpoint(
        db, session_id, payload.checkpoint_id, payload.title
    )
    return BranchOut(
        id=str(new_session.id),
        title=new_session.title,
        parent_session_id=str(new_session.parent_session_id),
        branch_checkpoint_id=new_session.branch_checkpoint_id,
    )

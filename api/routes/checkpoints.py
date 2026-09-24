"""Checkpoint history and time-travel restore."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from agents.service import AgentService
from api.deps import get_agent_service
from api.schemas import CheckpointOut, RestoreRequest

router = APIRouter(prefix="/checkpoints", tags=["checkpoints"])


@router.get("/{session_id}", response_model=list[CheckpointOut])
async def list_checkpoints(
    session_id: str, service: AgentService = Depends(get_agent_service)
) -> list[CheckpointOut]:
    history = await service.list_checkpoints(session_id)
    return [CheckpointOut(**h) for h in history]


@router.post("/{session_id}/restore")
async def restore_checkpoint(
    session_id: str,
    payload: RestoreRequest,
    service: AgentService = Depends(get_agent_service),
) -> dict:
    return await service.restore_checkpoint(session_id, payload.checkpoint_id)

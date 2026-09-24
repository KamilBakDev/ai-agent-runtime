"""Pydantic request/response models for the API layer."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str = Field(min_length=1)


class SessionCreate(BaseModel):
    title: str = "New session"


class MessageOut(BaseModel):
    role: str
    node: str | None = None
    content: str
    created_at: datetime | None = None


class SessionSummary(BaseModel):
    id: str
    title: str
    status: str
    created_at: datetime
    updated_at: datetime
    parent_session_id: str | None = None

    model_config = {"from_attributes": True}


class SessionDetail(SessionSummary):
    messages: list[MessageOut]
    graph_next: list[str] = Field(default_factory=list)
    current_step: str | None = None


class PendingActionOut(BaseModel):
    id: str
    node_name: str
    payload: dict
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}


class ResumeRequest(BaseModel):
    review_status: str = Field(pattern="^(approved|rejected|edited)$")
    review_feedback: str | None = None
    edited_output: str | None = None


class TurnResultOut(BaseModel):
    session_id: str
    status: str
    new_messages: list[MessageOut]
    final_output: str | None = None
    pending_action_id: str | None = None


class CheckpointOut(BaseModel):
    checkpoint_id: str | None
    next: list[str]
    current_step: str | None = None
    created_at: str | None = None


class RestoreRequest(BaseModel):
    checkpoint_id: str


class BranchRequest(BaseModel):
    checkpoint_id: str
    title: str = "Branch"


class BranchOut(BaseModel):
    id: str
    title: str
    parent_session_id: str | None
    branch_checkpoint_id: str | None

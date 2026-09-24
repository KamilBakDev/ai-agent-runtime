"""Ties the LangGraph orchestrator to session/message/pending-action persistence.

This is the single place that knows how to run a turn, pause for human review, resume,
time-travel to an earlier checkpoint, and branch a new session off one. API routes stay
thin wrappers around this service.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agents.orchestrator import compile_graph, initial_state
from agents.subagents.researcher import RetrieveFn
from db.models import Message, PendingAction
from db.models import Session as SessionRow


@dataclass
class TurnResult:
    session_id: str
    status: str  # "awaiting_review" | "completed"
    new_messages: list[dict[str, Any]] = field(default_factory=list)
    final_output: str | None = None
    pending_action_id: str | None = None


def _thread_config(session_id: str, checkpoint_id: str | None = None) -> dict[str, Any]:
    configurable: dict[str, Any] = {"thread_id": session_id}
    if checkpoint_id:
        configurable["checkpoint_id"] = checkpoint_id
    return {"configurable": configurable}


def _serialize_message(m: BaseMessage) -> dict[str, Any]:
    return {
        "role": "assistant" if isinstance(m, AIMessage) else "user",
        "node": getattr(m, "name", None),
        "content": m.content,
    }


class AgentService:
    def __init__(
        self,
        checkpointer: BaseCheckpointSaver,
        llm: BaseChatModel | None = None,
        retrieve: RetrieveFn | None = None,
    ) -> None:
        self.checkpointer = checkpointer
        self.graph = compile_graph(checkpointer, llm=llm, retrieve=retrieve)

    async def _persist_new_messages(
        self, db: AsyncSession, session_id: str, before_count: int, messages: list[BaseMessage]
    ) -> list[dict[str, Any]]:
        new = messages[before_count:]
        out = []
        for m in new:
            row = Message(
                session_id=uuid.UUID(session_id),
                role="assistant" if isinstance(m, AIMessage) else "user",
                node=getattr(m, "name", None),
                content=str(m.content),
            )
            db.add(row)
            out.append(_serialize_message(m))
        return out

    async def _after_step(
        self, db: AsyncSession, session_id: str, before_count: int
    ) -> TurnResult:
        config = _thread_config(session_id)
        snapshot = await self.graph.aget_state(config)
        new_messages = await self._persist_new_messages(
            db, session_id, before_count, snapshot.values.get("messages", [])
        )

        if snapshot.next and "reviewer" in snapshot.next:
            payload = {
                "code_draft": snapshot.values.get("code_draft"),
                "research_notes": snapshot.values.get("research_notes"),
                "revision_count": snapshot.values.get("revision_count", 0),
            }
            pending = PendingAction(
                session_id=uuid.UUID(session_id),
                node_name="reviewer",
                checkpoint_id=snapshot.config["configurable"].get("checkpoint_id"),
                payload=payload,
                status="pending",
            )
            db.add(pending)
            await db.flush()
            await db.commit()
            return TurnResult(
                session_id=session_id,
                status="awaiting_review",
                new_messages=new_messages,
                pending_action_id=str(pending.id),
            )

        await db.commit()
        return TurnResult(
            session_id=session_id,
            status="completed",
            new_messages=new_messages,
            final_output=snapshot.values.get("final_output"),
        )

    async def run_turn(self, db: AsyncSession, session_id: str, user_message: str) -> TurnResult:
        config = _thread_config(session_id)
        state = initial_state(session_id, user_message)
        await self.graph.ainvoke(state, config)
        return await self._after_step(db, session_id, before_count=0)

    async def resume_turn(
        self, db: AsyncSession, session_id: str, decision: dict[str, Any]
    ) -> TurnResult:
        config = _thread_config(session_id)
        snapshot_before = await self.graph.aget_state(config)
        before_count = len(snapshot_before.values.get("messages", []))

        await self.graph.aupdate_state(config, decision)

        result = await db.execute(
            select(PendingAction).where(
                PendingAction.session_id == uuid.UUID(session_id),
                PendingAction.status == "pending",
            )
        )
        pending = result.scalar_one_or_none()
        if pending is not None:
            pending.status = decision.get("review_status", "resolved")
            pending.resolution = decision

        await self.graph.ainvoke(None, config)
        return await self._after_step(db, session_id, before_count=before_count)

    async def get_state(self, session_id: str) -> dict[str, Any]:
        config = _thread_config(session_id)
        snapshot = await self.graph.aget_state(config)
        return {
            "values": {k: v for k, v in snapshot.values.items() if k != "messages"},
            "messages": [_serialize_message(m) for m in snapshot.values.get("messages", [])],
            "next": list(snapshot.next),
            "checkpoint_id": snapshot.config.get("configurable", {}).get("checkpoint_id"),
        }

    async def list_checkpoints(self, session_id: str) -> list[dict[str, Any]]:
        config = _thread_config(session_id)
        history = []
        async for snapshot in self.graph.aget_state_history(config):
            history.append(
                {
                    "checkpoint_id": snapshot.config["configurable"].get("checkpoint_id"),
                    "next": list(snapshot.next),
                    "current_step": snapshot.values.get("current_step"),
                    "created_at": (
                        snapshot.created_at if isinstance(snapshot.created_at, str) else None
                    ),
                }
            )
        return history

    async def restore_checkpoint(self, session_id: str, checkpoint_id: str) -> dict[str, Any]:
        """Time-travel: make ``checkpoint_id`` the thread's latest state (non-destructively)."""
        config = _thread_config(session_id, checkpoint_id)
        await self.graph.aupdate_state(config, {})
        return await self.get_state(session_id)

    async def branch_from_checkpoint(
        self, db: AsyncSession, source_session_id: str, checkpoint_id: str, title: str
    ) -> SessionRow:
        """Fork a checkpoint into a brand-new session/thread."""
        source_config = _thread_config(source_session_id, checkpoint_id)
        snapshot = await self.graph.aget_state(source_config)

        new_session = SessionRow(
            id=uuid.uuid4(),
            title=title,
            parent_session_id=uuid.UUID(source_session_id),
            branch_checkpoint_id=checkpoint_id,
        )
        db.add(new_session)
        await db.flush()

        new_config = _thread_config(str(new_session.id))
        await self.graph.aupdate_state(new_config, dict(snapshot.values))
        await db.commit()
        return new_session

    async def pending_action(self, db: AsyncSession, session_id: str) -> PendingAction | None:
        result = await db.execute(
            select(PendingAction).where(
                PendingAction.session_id == uuid.UUID(session_id),
                PendingAction.status == "pending",
            )
        )
        return result.scalar_one_or_none()

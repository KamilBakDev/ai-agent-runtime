"""POST /a2a/invoke -- lets another agent (or process) address researcher/coder/
reviewer as standalone, stateless HTTP services, independent of the LangGraph
orchestrator's in-process nodes. This is what agents.tools.a2a_client.A2AClient calls."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from agents.llm import get_chat_model
from agents.subagents.coder import CODER_SYSTEM_PROMPT
from agents.subagents.researcher import RESEARCHER_SYSTEM_PROMPT
from agents.tools.legal_tools import search_cases

router = APIRouter(prefix="/a2a", tags=["a2a"])


class A2AInvokeRequest(BaseModel):
    agent: str
    task: dict[str, Any]


class A2AInvokeResponse(BaseModel):
    agent: str
    output: dict[str, Any]


@router.post("/invoke", response_model=A2AInvokeResponse)
async def invoke(payload: A2AInvokeRequest) -> A2AInvokeResponse:
    llm = get_chat_model()

    if payload.agent == "researcher":
        query = payload.task.get("query", "")
        citations = await search_cases(query) if query else []
        context = "\n\n".join(f"[{i + 1}] {c['text']}" for i, c in enumerate(citations))
        response = await llm.ainvoke(
            [
                SystemMessage(content=RESEARCHER_SYSTEM_PROMPT),
                HumanMessage(
                    content=f"User request:\n{query}\n\nRetrieved context:\n{context or '(none)'}"
                ),
            ]
        )
        return A2AInvokeResponse(
            agent="researcher",
            output={"research_notes": response.content, "citations": citations},
        )

    if payload.agent == "coder":
        notes = payload.task.get("research_notes", "")
        feedback = payload.task.get("feedback")
        parts = [f"Research notes:\n{notes}"]
        if feedback:
            parts.append(f"Reviewer feedback on previous draft:\n{feedback}")
        response = await llm.ainvoke(
            [SystemMessage(content=CODER_SYSTEM_PROMPT), HumanMessage(content="\n\n".join(parts))]
        )
        return A2AInvokeResponse(agent="coder", output={"code_draft": response.content})

    if payload.agent == "reviewer":
        status = payload.task.get("review_status")
        draft = payload.task.get("code_draft")
        edited = payload.task.get("edited_output")
        if status == "approved":
            final = draft
        elif status == "edited" and edited:
            final = edited
        else:
            final = None
        return A2AInvokeResponse(
            agent="reviewer", output={"final_output": final, "review_status": status}
        )

    raise HTTPException(status_code=404, detail=f"Unknown agent: {payload.agent!r}")

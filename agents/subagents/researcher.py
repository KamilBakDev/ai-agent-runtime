"""Researcher subagent: gathers context via RAG + optional web search, cites sources."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

RESEARCHER_SYSTEM_PROMPT = (
    "You are the Researcher agent in a multi-agent legal/technical workflow. "
    "Read the user's request, use any retrieved context provided, and produce a concise, "
    "well-cited set of research notes. Cite sources as [1], [2], etc. when context is given."
)

RetrieveFn = Callable[[str], Awaitable[list[dict[str, Any]]]]


def make_researcher_node(
    llm: BaseChatModel, retrieve: RetrieveFn | None = None
) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
    """Build the researcher graph node, bound to a given chat model and retriever."""

    async def researcher_node(state: dict[str, Any]) -> dict[str, Any]:
        last_user = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), ""
        )

        citations: list[dict[str, Any]] = []
        context_block = ""
        if retrieve is not None:
            citations = await retrieve(last_user)
            if citations:
                context_block = "\n\n".join(
                    f"[{i + 1}] {c.get('text', '')}" for i, c in enumerate(citations)
                )

        prompt = [
            SystemMessage(content=RESEARCHER_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"User request:\n{last_user}\n\n"
                    f"Retrieved context:\n{context_block or '(none available)'}"
                )
            ),
        ]
        response = await llm.ainvoke(prompt)
        notes = response.content if isinstance(response, AIMessage) else str(response)

        return {
            "messages": [AIMessage(content=notes, name="researcher")],
            "research_notes": notes,
            "citations": citations,
            "current_step": "researcher",
        }

    return researcher_node

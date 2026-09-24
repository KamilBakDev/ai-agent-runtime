"""Coder subagent: generates Python/SQL from the researcher's notes, revises on rejection."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.observability import extract_usage, metrics

CODER_SYSTEM_PROMPT = (
    "You are the Coder agent in a multi-agent workflow. Given research notes, write the "
    "minimal, correct Python or SQL needed to satisfy the request. If given reviewer "
    "feedback on a previous draft, revise the draft to address it. Return only the code "
    "and a one-line explanation."
)


def make_coder_node(llm: BaseChatModel) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
    """Build the coder graph node, bound to a given chat model."""

    async def coder_node(state: dict[str, Any]) -> dict[str, Any]:
        research_notes = state.get("research_notes", "")
        feedback = state.get("review_feedback")
        revision_count = state.get("revision_count", 0)

        human_parts = [f"Research notes:\n{research_notes}"]
        if feedback:
            human_parts.append(f"Reviewer feedback on previous draft (please address it):\n{feedback}")

        prompt = [
            SystemMessage(content=CODER_SYSTEM_PROMPT),
            HumanMessage(content="\n\n".join(human_parts)),
        ]
        response = await llm.ainvoke(prompt)
        draft = response.content if isinstance(response, AIMessage) else str(response)
        prompt_tokens, completion_tokens = extract_usage(response)
        metrics.record_llm_usage("coder", prompt_tokens, completion_tokens)

        return {
            "messages": [AIMessage(content=draft, name="coder")],
            "code_draft": draft,
            "revision_count": revision_count + (1 if feedback else 0),
            "current_step": "coder",
        }

    return coder_node

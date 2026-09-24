"""LangGraph stateful orchestrator: researcher -> coder -> reviewer (HITL) -> END.

The reviewer node is a static interrupt point (``interrupt_before=["reviewer"]``), so a
run pauses there until a human decision is written into state and the graph is resumed.
On rejection, ``route_after_review`` sends the flow back to the coder for a revision
(bounded by ``MAX_REVISIONS`` in :mod:`agents.subagents.reviewer`), which is the graph's
branching behaviour.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agents.llm import get_chat_model
from agents.subagents.coder import make_coder_node
from agents.subagents.reviewer import reviewer_node, route_after_review
from agents.subagents.researcher import RetrieveFn, make_researcher_node


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    current_step: str
    research_notes: str
    citations: list[dict[str, Any]]
    code_draft: str
    review_status: str | None
    review_feedback: str | None
    edited_output: str | None
    revision_count: int
    final_output: str | None


def build_graph(
    llm: BaseChatModel | None = None, retrieve: RetrieveFn | None = None
) -> StateGraph:
    """Construct the (uncompiled) orchestrator graph."""
    llm = llm or get_chat_model()

    graph: StateGraph = StateGraph(AgentState)
    graph.add_node("researcher", make_researcher_node(llm, retrieve))
    graph.add_node("coder", make_coder_node(llm))
    graph.add_node("reviewer", reviewer_node)

    graph.add_edge(START, "researcher")
    graph.add_edge("researcher", "coder")
    graph.add_edge("coder", "reviewer")
    graph.add_conditional_edges(
        "reviewer", route_after_review, {"coder": "coder", "END": END}
    )

    return graph


def compile_graph(
    checkpointer: BaseCheckpointSaver,
    llm: BaseChatModel | None = None,
    retrieve: RetrieveFn | None = None,
):
    """Build and compile the graph with a checkpointer and the reviewer HITL interrupt."""
    graph = build_graph(llm=llm, retrieve=retrieve)
    return graph.compile(checkpointer=checkpointer, interrupt_before=["reviewer"])


def initial_state(session_id: str, user_message: str) -> AgentState:
    from langchain_core.messages import HumanMessage

    return AgentState(
        messages=[HumanMessage(content=user_message)],
        session_id=session_id,
        current_step="start",
        research_notes="",
        citations=[],
        code_draft="",
        review_status=None,
        review_feedback=None,
        edited_output=None,
        revision_count=0,
        final_output=None,
    )

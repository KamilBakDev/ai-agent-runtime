"""Reviewer subagent: the human-in-the-loop gate.

The graph is compiled with ``interrupt_before=["reviewer"]``, so execution pauses
right before this node runs. A human (via the API/UI) inspects ``state['code_draft']``
and updates the state with a decision (``review_status`` + optional ``review_feedback``
/ ``edited_output``) before the graph is resumed. This node then just packages the
outcome; branching back to the coder for revisions happens on the *next* edge.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

MAX_REVISIONS = 3


async def reviewer_node(state: dict[str, Any]) -> dict[str, Any]:
    review_status = state.get("review_status", "pending")
    edited_output = state.get("edited_output")

    if review_status == "approved":
        final_output = state.get("code_draft", "")
        summary = "Reviewer approved the draft."
    elif review_status == "edited" and edited_output:
        final_output = edited_output
        summary = "Reviewer edited the draft before approval."
    elif review_status == "rejected":
        final_output = None
        summary = f"Reviewer rejected the draft: {state.get('review_feedback', '(no feedback given)')}"
    else:
        # Should not normally be reached: reviewer ran without a decision recorded.
        final_output = None
        summary = "Reviewer node reached without a recorded decision."

    return {
        "messages": [AIMessage(content=summary, name="reviewer")],
        "final_output": final_output,
        "current_step": "reviewer",
    }


def route_after_review(state: dict[str, Any]) -> str:
    """Conditional edge: loop back to the coder on rejection, otherwise finish."""
    if state.get("review_status") == "rejected" and state.get("revision_count", 0) < MAX_REVISIONS:
        return "coder"
    return "END"

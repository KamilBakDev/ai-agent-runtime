"""Human-in-the-loop panel: accept / reject / edit a paused agent turn."""

from __future__ import annotations

import requests
import streamlit as st


def render_pending_actions(api_base_url: str, session_id: str) -> None:
    resp = requests.get(f"{api_base_url}/sessions/{session_id}/pending_actions", timeout=10)
    if resp.status_code != 200 or not resp.json():
        return

    action = resp.json()[0]
    st.warning("⏸️ Awaiting human review")
    st.json(action["payload"])

    col_approve, col_reject, col_edit = st.columns(3)

    if col_approve.button("✅ Approve", key=f"approve-{action['id']}"):
        requests.post(
            f"{api_base_url}/sessions/{session_id}/resume",
            json={"review_status": "approved"},
            timeout=30,
        )
        st.rerun()

    feedback = st.text_input("Rejection feedback", key=f"feedback-{action['id']}")
    if col_reject.button("❌ Reject", key=f"reject-{action['id']}"):
        requests.post(
            f"{api_base_url}/sessions/{session_id}/resume",
            json={"review_status": "rejected", "review_feedback": feedback or "Needs revision."},
            timeout=30,
        )
        st.rerun()

    edited = st.text_area(
        "Edited output",
        value=action["payload"].get("code_draft", ""),
        key=f"edit-{action['id']}",
    )
    if col_edit.button("✏️ Approve edited", key=f"edited-{action['id']}"):
        requests.post(
            f"{api_base_url}/sessions/{session_id}/resume",
            json={"review_status": "edited", "edited_output": edited},
            timeout=30,
        )
        st.rerun()

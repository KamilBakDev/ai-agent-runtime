"""Chat panel: message history + streaming input."""

from __future__ import annotations

import json

import requests
import streamlit as st


def render_chat(api_base_url: str, session_id: str | None) -> str | None:
    """Render chat history and the input box. Returns the (possibly new) session_id."""
    if session_id:
        resp = requests.get(f"{api_base_url}/sessions/{session_id}", timeout=10)
        if resp.status_code == 200:
            detail = resp.json()
            if detail.get("parent_session_id"):
                st.caption(f"🌿 Branched from session `{detail['parent_session_id'][:8]}`")
            for m in detail["messages"]:
                role = "user" if m["role"] == "user" else "assistant"
                with st.chat_message(role):
                    label = f"**{m['node']}** — " if m.get("node") else ""
                    st.markdown(label + m["content"])

    prompt = st.chat_input("Ask the agent...")
    if not prompt:
        return session_id

    with st.chat_message("user"):
        st.markdown(prompt)

    status_box = st.empty()
    payload: dict = {"message": prompt}
    if session_id:
        payload["session_id"] = session_id

    with requests.post(f"{api_base_url}/chat", json=payload, stream=True, timeout=60) as r:
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = json.loads(line[len("data:") :].strip())
            if "session_id" in data and session_id is None:
                session_id = data["session_id"]
                st.session_state["session_id"] = session_id
            if data.get("type") == "node_update":
                status_box.info(f"Running **{data['node']}** ({data.get('current_step')})")
            elif data.get("type") == "turn_complete":
                if data.get("status") == "awaiting_review":
                    status_box.warning("Paused: awaiting human review.")
                else:
                    status_box.success("Turn complete.")

    st.rerun()
    return session_id

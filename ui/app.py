"""Streamlit UI: chat, checkpoint viewer, human-in-the-loop review."""

from __future__ import annotations

import os

import requests
import streamlit as st

from ui.components.chat import render_chat
from ui.components.checkpoint_viewer import render_checkpoints
from ui.components.pending import render_pending_actions

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="AI Agent Runtime", layout="wide")
st.title("AI Agent Runtime")
st.caption("LangGraph multi-agent demo: researcher -> coder -> reviewer (human-in-the-loop)")

if "session_id" not in st.session_state:
    st.session_state["session_id"] = None

with st.sidebar:
    st.header("Sessions")
    if st.button("+ New session"):
        st.session_state["session_id"] = None
        st.rerun()

    try:
        resp = requests.get(f"{API_BASE_URL}/sessions", timeout=5)
        sessions = resp.json() if resp.status_code == 200 else []
    except requests.RequestException:
        sessions = []
        st.error(f"Cannot reach API at {API_BASE_URL}")

    for s in sessions:
        label = f"{s['title'][:28]}  ({s['id'][:8]})"
        if st.button(label, key=f"session-{s['id']}", use_container_width=True):
            st.session_state["session_id"] = s["id"]
            st.rerun()

session_id = st.session_state["session_id"]

col_chat, col_side = st.columns([2, 1])

with col_chat:
    st.subheader("Chat")
    new_session_id = render_chat(API_BASE_URL, session_id)
    if new_session_id != session_id:
        st.session_state["session_id"] = new_session_id

with col_side:
    if st.session_state["session_id"]:
        render_pending_actions(API_BASE_URL, st.session_state["session_id"])
        render_checkpoints(API_BASE_URL, st.session_state["session_id"])
    else:
        st.info("Start a conversation to see pending actions and checkpoints.")

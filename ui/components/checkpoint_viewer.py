"""Checkpoint history viewer: time-travel restore + branch-from-checkpoint."""

from __future__ import annotations

import requests
import streamlit as st


def render_checkpoints(api_base_url: str, session_id: str) -> None:
    st.subheader("Checkpoints (time-travel)")
    resp = requests.get(f"{api_base_url}/checkpoints/{session_id}", timeout=10)
    if resp.status_code != 200 or not resp.json():
        st.caption("No checkpoints yet.")
        return

    for cp in resp.json():
        short_id = (cp["checkpoint_id"] or "")[:12]
        st.markdown(f"`{short_id}` — step **{cp['current_step']}**, next `{cp['next']}`")
        col_restore, col_branch = st.columns(2)

        if col_restore.button("⏪ Restore", key=f"restore-{cp['checkpoint_id']}"):
            r = requests.post(
                f"{api_base_url}/checkpoints/{session_id}/restore",
                json={"checkpoint_id": cp["checkpoint_id"]},
                timeout=10,
            )
            if r.status_code == 200:
                st.success("Restored to this checkpoint.")
                st.rerun()
            else:
                st.error(r.text)

        if col_branch.button("🌿 Branch", key=f"branch-{cp['checkpoint_id']}"):
            r = requests.post(
                f"{api_base_url}/sessions/{session_id}/branch",
                json={
                    "checkpoint_id": cp["checkpoint_id"],
                    "title": f"Branch @ {cp['current_step']}",
                },
                timeout=10,
            )
            if r.status_code == 200:
                st.session_state["session_id"] = r.json()["id"]
                st.success("Branch created.")
                st.rerun()
            else:
                st.error(r.text)

"""Streamlit chat UI for the AI Personal Assistant.

Talks to the FastAPI backend over HTTP when it's reachable (BACKEND_URL),
and transparently falls back to calling the assistant module in-process
otherwise - so `streamlit run streamlit_app.py` works standalone without
also having to run `uvicorn` first.

Run with:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import os
import uuid

import requests
import streamlit as st

from app.assistant import Assistant

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="AI Personal Assistant", page_icon="🗓️", layout="wide")


@st.cache_resource
def _get_local_assistant() -> Assistant:
    return Assistant()


def _backend_reachable() -> bool:
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=1.5)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def _send_message(session_id: str, message: str) -> dict:
    if _backend_reachable():
        resp = requests.post(
            f"{BACKEND_URL}/chat",
            json={"session_id": session_id, "message": message},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    # Fallback: call the assistant module directly, in-process.
    assistant = _get_local_assistant()
    result = assistant.chat(session_id, message)
    return {
        "reply": result.reply,
        "tool_calls": result.tool_calls,
        "compaction_triggered": result.compaction_triggered,
        "turn_count": result.turn_count,
    }


def _fetch_tasks(session_id: str) -> list[dict]:
    if _backend_reachable():
        resp = requests.get(f"{BACKEND_URL}/sessions/{session_id}/tasks", timeout=10)
        resp.raise_for_status()
        return resp.json()["tasks"]
    return _get_local_assistant().get_tasks(session_id)


def _fetch_reminders(session_id: str) -> list[dict]:
    if _backend_reachable():
        resp = requests.get(f"{BACKEND_URL}/sessions/{session_id}/reminders", timeout=10)
        resp.raise_for_status()
        return resp.json()["reminders"]
    return _get_local_assistant().get_reminders(session_id)


if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

st.title("AI Personal Assistant (Agentic)")
st.caption(
    "LLM agent with conversational memory, tool calling (tasks, reminders, "
    "knowledge lookup), and LangGraph orchestration."
)

with st.sidebar:
    st.subheader("Session")
    st.text_input("Session ID", key="session_id")
    backend_status = "connected" if _backend_reachable() else "not running (using in-process fallback)"
    st.caption(f"Backend: {BACKEND_URL} - {backend_status}")

    st.subheader("Tasks")
    for task in _fetch_tasks(st.session_state.session_id):
        status = "✅" if task["completed"] else "⬜"
        st.write(f"{status} #{task['id']} {task['title']}")

    st.subheader("Reminders")
    for reminder in _fetch_reminders(st.session_state.session_id):
        st.write(f"⏰ {reminder['message']} — {reminder['due_at']}")

    if st.button("Refresh"):
        st.rerun()

for entry in st.session_state.chat_log:
    with st.chat_message(entry["role"]):
        st.write(entry["content"])
        if entry.get("tool_calls"):
            with st.expander("Tool calls made"):
                for call in entry["tool_calls"]:
                    st.json(call)

user_input = st.chat_input(
    "e.g. 'add a task to call the dentist and remind me tomorrow at 9am'"
)
if user_input:
    st.session_state.chat_log.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = _send_message(st.session_state.session_id, user_input)
        st.write(response["reply"])
        if response.get("tool_calls"):
            with st.expander("Tool calls made"):
                for call in response["tool_calls"]:
                    st.json(call)
        if response.get("compaction_triggered"):
            st.caption("(Conversation history was summarized to stay within context limits.)")

    st.session_state.chat_log.append(
        {
            "role": "assistant",
            "content": response["reply"],
            "tool_calls": response.get("tool_calls", []),
        }
    )

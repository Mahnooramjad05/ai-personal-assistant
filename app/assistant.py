"""Core assistant module: ties conversational context together with the
LangGraph tool-orchestration loop. This is the single entry point used by
both the FastAPI backend and (optionally, directly) the Streamlit UI."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app import store
from app.config import settings
from app.conversation import ConversationManager
from app.llm_client import LLMClient, get_llm_client
from app.orchestrator import run_orchestration


@dataclass
class ChatResult:
    reply: str
    tool_calls: list[dict[str, Any]]
    compaction_triggered: bool
    turn_count: int


class Assistant:
    """Stateless-per-call facade: every method takes a session_id and reads
    /writes durable state via app.store, so the assistant itself holds no
    session state between calls (safe to use from a multi-worker backend)."""

    def __init__(self, llm: LLMClient | None = None, db_path: str | None = None) -> None:
        self.llm = llm or get_llm_client()
        self.db_path = db_path or settings.database_path
        store.init_db(self.db_path)

    def chat(self, session_id: str, message: str) -> ChatResult:
        convo = ConversationManager(
            session_id=session_id,
            llm=self.llm,
            max_turns_before_summary=settings.max_turns_before_summary,
            db_path=self.db_path,
        )
        convo.add_user_message(message)

        llm_messages = convo.build_llm_messages()
        reply, tool_calls = run_orchestration(self.llm, session_id, llm_messages)

        convo.add_assistant_message(reply)

        return ChatResult(
            reply=reply,
            tool_calls=tool_calls,
            compaction_triggered=convo.last_compaction_happened,
            turn_count=convo.turn_count(),
        )

    def get_tasks(self, session_id: str) -> list[dict[str, Any]]:
        return [t.to_dict() for t in store.list_tasks(session_id, db_path=self.db_path)]

    def get_reminders(self, session_id: str) -> list[dict[str, Any]]:
        return [r.to_dict() for r in store.list_reminders(session_id, db_path=self.db_path)]

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        return store.get_messages(session_id, db_path=self.db_path)

    def get_summary(self, session_id: str) -> str | None:
        return store.get_latest_summary(session_id, db_path=self.db_path)

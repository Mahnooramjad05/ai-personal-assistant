"""Session-scoped conversational context with summarization/compaction.

Long conversations are kept bounded: once the number of turn-pairs (one user
message + one assistant reply = one turn) stored verbatim exceeds
`settings.max_turns_before_summary`, the oldest half of the history is
collapsed into a running summary via a real LLM call (see
`llm_client.LLMClient.summarize`), and only the summary plus the most recent
turns are kept. This prevents the context window from growing unboundedly
across a long session while still preserving salient facts (names, dates,
task/reminder mentions, decisions) via the summary.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app import store
from app.llm_client import LLMClient


@dataclass
class ConversationManager:
    session_id: str
    llm: LLMClient
    max_turns_before_summary: int
    db_path: str | None = None
    _messages: list[dict[str, str]] = field(default_factory=list, init=False)
    _summary: str | None = field(default=None, init=False)
    last_compaction_happened: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._messages = store.get_messages(self.session_id, db_path=self.db_path)
        self._summary = store.get_latest_summary(self.session_id, db_path=self.db_path)

    # -- public API -------------------------------------------------------
    @property
    def summary(self) -> str | None:
        return self._summary

    @property
    def raw_messages(self) -> list[dict[str, str]]:
        return list(self._messages)

    def turn_count(self) -> int:
        """Number of complete user/assistant turn-pairs currently held verbatim."""
        return len(self._messages) // 2

    def add_user_message(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})
        store.append_message(self.session_id, "user", content, db_path=self.db_path)

    def add_assistant_message(self, content: str) -> None:
        self._messages.append({"role": "assistant", "content": content})
        store.append_message(self.session_id, "assistant", content, db_path=self.db_path)
        # Compaction is checked after each full turn (user + assistant reply).
        self._maybe_compact()

    def build_llm_messages(self) -> list[dict[str, str]]:
        """Return the message list to send to the LLM: optional summary
        preamble followed by the verbatim recent turns."""
        messages: list[dict[str, str]] = []
        if self._summary:
            messages.append(
                {
                    "role": "user",
                    "content": f"[Earlier conversation summary]\n{self._summary}",
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": "Understood, I have the earlier context.",
                }
            )
        messages.extend(self._messages)
        return messages

    # -- compaction ---------------------------------------------------
    def _maybe_compact(self) -> None:
        self.last_compaction_happened = False
        if self.turn_count() <= self.max_turns_before_summary:
            return

        # Collapse the oldest half of the stored turns into the summary,
        # keeping the newer half verbatim. This keeps context bounded while
        # still leaving recent exchanges available for immediate reference.
        half = (len(self._messages) // 2) if len(self._messages) % 2 == 0 else len(
            self._messages
        ) - 1
        cutoff = max(2, (half // 2) * 2)  # keep it an even number of messages
        to_summarize = self._messages[:cutoff]
        remaining = self._messages[cutoff:]

        conversation_text = self._render_for_summary(to_summarize)
        new_summary_piece = self.llm.summarize(conversation_text)

        if self._summary:
            combined = f"{self._summary}\n{new_summary_piece}"
        else:
            combined = new_summary_piece

        self._summary = combined
        self._messages = remaining

        store.save_summary(self.session_id, combined, db_path=self.db_path)
        # Persisted messages must mirror in-memory state: drop the summarized
        # prefix from the DB too, keyed off how many rows we just removed.
        store.delete_messages(
            self.session_id,
            up_to_id=self._resolve_cutoff_row_id(cutoff),
            db_path=self.db_path,
        )
        self.last_compaction_happened = True

    def _resolve_cutoff_row_id(self, cutoff_count: int) -> int | None:
        rows = store.get_messages(self.session_id, db_path=self.db_path)
        if cutoff_count <= 0 or cutoff_count > len(rows):
            return None
        # get_messages doesn't return ids, so re-query directly for the id
        # boundary using a dedicated connection.
        with store.get_connection(self.db_path) as conn:
            all_rows = conn.execute(
                "SELECT id FROM messages WHERE session_id = ? ORDER BY id ASC",
                (self.session_id,),
            ).fetchall()
        if not all_rows or cutoff_count > len(all_rows):
            return None
        return all_rows[cutoff_count - 1]["id"]

    @staticmethod
    def _render_for_summary(messages: list[dict[str, str]]) -> str:
        lines = []
        for m in messages:
            speaker = "User" if m["role"] == "user" else "Assistant"
            lines.append(f"{speaker}: {m['content']}")
        return "\n".join(lines)

"""SQLite-backed storage for tasks, reminders, and conversation history.

A single small module owns the schema and all CRUD operations so that both
the tool layer and the FastAPI verification endpoints share one source of
truth. Each session_id partitions its own tasks/reminders/messages, so
multiple users/conversations can share one database file safely.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from app.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    title TEXT NOT NULL,
    completed INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS reminders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    message TEXT NOT NULL,
    due_at TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conversation_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    summary TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def get_db_path() -> str:
    return settings.database_path


def _connect(db_path: str | None = None) -> sqlite3.Connection:
    path = db_path or get_db_path()
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str | None = None) -> None:
    """Create tables if they don't exist yet. Safe to call repeatedly."""
    path = db_path or get_db_path()
    parent = Path(path).parent
    if str(parent) not in ("", "."):
        parent.mkdir(parents=True, exist_ok=True)
    with _connect(path) as conn:
        conn.executescript(SCHEMA)
        conn.commit()


@contextmanager
def get_connection(db_path: str | None = None):
    conn = _connect(db_path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

@dataclass
class Task:
    id: int
    session_id: str
    title: str
    completed: bool
    created_at: str
    completed_at: str | None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "completed": self.completed,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


def create_task(session_id: str, title: str, db_path: str | None = None) -> Task:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO tasks (session_id, title, completed, created_at) "
            "VALUES (?, ?, 0, ?)",
            (session_id, title, _now()),
        )
        task_id = cur.lastrowid
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return Task(
            id=row["id"],
            session_id=row["session_id"],
            title=row["title"],
            completed=bool(row["completed"]),
            created_at=row["created_at"],
            completed_at=row["completed_at"],
        )


def list_tasks(
    session_id: str, include_completed: bool = True, db_path: str | None = None
) -> list[Task]:
    query = "SELECT * FROM tasks WHERE session_id = ?"
    params: list = [session_id]
    if not include_completed:
        query += " AND completed = 0"
    query += " ORDER BY id ASC"
    with get_connection(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
        return [
            Task(
                id=r["id"],
                session_id=r["session_id"],
                title=r["title"],
                completed=bool(r["completed"]),
                created_at=r["created_at"],
                completed_at=r["completed_at"],
            )
            for r in rows
        ]


def find_task_by_title(
    session_id: str, title: str, db_path: str | None = None
) -> Task | None:
    """Case-insensitive fuzzy match: exact match first, then substring."""
    tasks = list_tasks(session_id, db_path=db_path)
    title_lower = title.strip().lower()
    for t in tasks:
        if t.title.strip().lower() == title_lower:
            return t
    for t in tasks:
        if title_lower in t.title.strip().lower() or t.title.strip().lower() in title_lower:
            return t
    return None


def complete_task(
    session_id: str, title: str, db_path: str | None = None
) -> Task | None:
    task = find_task_by_title(session_id, title, db_path=db_path)
    if task is None:
        return None
    with get_connection(db_path) as conn:
        conn.execute(
            "UPDATE tasks SET completed = 1, completed_at = ? WHERE id = ?",
            (_now(), task.id),
        )
    task.completed = True
    return task


def delete_task(
    session_id: str, title: str, db_path: str | None = None
) -> Task | None:
    task = find_task_by_title(session_id, title, db_path=db_path)
    if task is None:
        return None
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM tasks WHERE id = ?", (task.id,))
    return task


# ---------------------------------------------------------------------------
# Reminders
# ---------------------------------------------------------------------------

@dataclass
class Reminder:
    id: int
    session_id: str
    message: str
    due_at: str
    created_at: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "message": self.message,
            "due_at": self.due_at,
            "created_at": self.created_at,
        }


def create_reminder(
    session_id: str, message: str, due_at: datetime, db_path: str | None = None
) -> Reminder:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO reminders (session_id, message, due_at, created_at) "
            "VALUES (?, ?, ?, ?)",
            (session_id, message, due_at.isoformat(timespec="minutes"), _now()),
        )
        reminder_id = cur.lastrowid
        row = conn.execute(
            "SELECT * FROM reminders WHERE id = ?", (reminder_id,)
        ).fetchone()
        return Reminder(
            id=row["id"],
            session_id=row["session_id"],
            message=row["message"],
            due_at=row["due_at"],
            created_at=row["created_at"],
        )


def list_reminders(session_id: str, db_path: str | None = None) -> list[Reminder]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM reminders WHERE session_id = ? ORDER BY due_at ASC",
            (session_id,),
        ).fetchall()
        return [
            Reminder(
                id=r["id"],
                session_id=r["session_id"],
                message=r["message"],
                due_at=r["due_at"],
                created_at=r["created_at"],
            )
            for r in rows
        ]


def list_reminders_in_range(
    session_id: str, start: datetime, end: datetime, db_path: str | None = None
) -> list[Reminder]:
    """Return reminders whose due_at falls within [start, end)."""
    all_reminders = list_reminders(session_id, db_path=db_path)
    result = []
    for r in all_reminders:
        due = datetime.fromisoformat(r.due_at)
        if start <= due < end:
            result.append(r)
    return result


# ---------------------------------------------------------------------------
# Conversation messages + summaries (used by core.ConversationManager)
# ---------------------------------------------------------------------------

def append_message(
    session_id: str, role: str, content: str, db_path: str | None = None
) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) "
            "VALUES (?, ?, ?, ?)",
            (session_id, role, content, _now()),
        )


def get_messages(session_id: str, db_path: str | None = None) -> list[dict]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT role, content, created_at FROM messages "
            "WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in rows]


def delete_messages(
    session_id: str, up_to_id: int | None = None, db_path: str | None = None
) -> None:
    with get_connection(db_path) as conn:
        if up_to_id is None:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        else:
            conn.execute(
                "DELETE FROM messages WHERE session_id = ? AND id <= ?",
                (session_id, up_to_id),
            )


def save_summary(session_id: str, summary: str, db_path: str | None = None) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO conversation_summaries (session_id, summary, created_at) "
            "VALUES (?, ?, ?)",
            (session_id, summary, _now()),
        )


def get_latest_summary(session_id: str, db_path: str | None = None) -> str | None:
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT summary FROM conversation_summaries WHERE session_id = ? "
            "ORDER BY id DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        return row["summary"] if row else None


def clear_session(session_id: str, db_path: str | None = None) -> None:
    """Wipe all state for a session - used by tests to isolate cases."""
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM tasks WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM reminders WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.execute(
            "DELETE FROM conversation_summaries WHERE session_id = ?", (session_id,)
        )

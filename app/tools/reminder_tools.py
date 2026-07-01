"""Reminder tool: create time-based reminders and query them by due date,
backed by the same SQLite store as tasks."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from app import store
from app.tools.datetime_parse import day_range, parse_due_datetime


def create_reminder(session_id: str, message: str, when_text: str) -> dict[str, Any]:
    message = message.strip() or "Reminder"
    due_at = parse_due_datetime(when_text)
    reminder = store.create_reminder(session_id, message, due_at)
    return {"id": reminder.id, "message": reminder.message, "due_at": reminder.due_at}


def list_reminders(session_id: str, when: str = "today") -> dict[str, Any]:
    start, end = day_range(when)
    reminders = store.list_reminders_in_range(session_id, start, end)
    return {
        "when": when,
        "reminders": [r.to_dict() for r in reminders],
        "range": {"start": start.isoformat(), "end": end.isoformat()},
    }


def list_all_reminders(session_id: str) -> dict[str, Any]:
    reminders = store.list_reminders(session_id)
    return {"reminders": [r.to_dict() for r in reminders]}

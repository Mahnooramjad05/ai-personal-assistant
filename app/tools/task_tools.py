"""Task-management tool: create / list / complete / delete, backed by SQLite."""
from __future__ import annotations

from typing import Any

from app import store


def create_task(session_id: str, title: str) -> dict[str, Any]:
    title = title.strip() or "Untitled task"
    task = store.create_task(session_id, title)
    return {"id": task.id, "title": task.title, "created": True}


def list_tasks(session_id: str, include_completed: bool = True) -> dict[str, Any]:
    tasks = store.list_tasks(session_id, include_completed=include_completed)
    return {"tasks": [t.to_dict() for t in tasks]}


def complete_task(session_id: str, title: str) -> dict[str, Any]:
    task = store.complete_task(session_id, title)
    if task is None:
        return {"found": False, "title": title}
    return {"found": True, "title": task.title, "id": task.id}


def delete_task(session_id: str, title: str) -> dict[str, Any]:
    task = store.delete_task(session_id, title)
    if task is None:
        return {"found": False, "title": title}
    return {"found": True, "title": task.title, "id": task.id}

"""Task creation/listing/completion round-trips correctly against SQLite."""
from __future__ import annotations

from app import store


def test_create_and_list_task(db_path, session_id):
    store.create_task(session_id, "Buy groceries", db_path=db_path)
    tasks = store.list_tasks(session_id, db_path=db_path)
    assert len(tasks) == 1
    assert tasks[0].title == "Buy groceries"
    assert tasks[0].completed is False


def test_multiple_tasks_are_isolated_per_session(db_path):
    store.create_task("session-a", "Task A", db_path=db_path)
    store.create_task("session-b", "Task B", db_path=db_path)

    tasks_a = store.list_tasks("session-a", db_path=db_path)
    tasks_b = store.list_tasks("session-b", db_path=db_path)

    assert [t.title for t in tasks_a] == ["Task A"]
    assert [t.title for t in tasks_b] == ["Task B"]


def test_complete_task_round_trip(db_path, session_id):
    store.create_task(session_id, "Call the dentist", db_path=db_path)

    completed = store.complete_task(session_id, "Call the dentist", db_path=db_path)
    assert completed is not None
    assert completed.completed is True

    tasks = store.list_tasks(session_id, db_path=db_path)
    assert tasks[0].completed is True
    assert tasks[0].completed_at is not None


def test_complete_task_fuzzy_match(db_path, session_id):
    store.create_task(session_id, "Call the dentist about braces", db_path=db_path)

    completed = store.complete_task(session_id, "dentist", db_path=db_path)
    assert completed is not None
    assert "dentist" in completed.title.lower()


def test_complete_nonexistent_task_returns_none(db_path, session_id):
    result = store.complete_task(session_id, "does not exist", db_path=db_path)
    assert result is None


def test_delete_task_round_trip(db_path, session_id):
    store.create_task(session_id, "Temporary task", db_path=db_path)
    assert len(store.list_tasks(session_id, db_path=db_path)) == 1

    deleted = store.delete_task(session_id, "Temporary task", db_path=db_path)
    assert deleted is not None
    assert len(store.list_tasks(session_id, db_path=db_path)) == 0


def test_list_tasks_excluding_completed(db_path, session_id):
    store.create_task(session_id, "Task 1", db_path=db_path)
    store.create_task(session_id, "Task 2", db_path=db_path)
    store.complete_task(session_id, "Task 1", db_path=db_path)

    pending = store.list_tasks(session_id, include_completed=False, db_path=db_path)
    assert [t.title for t in pending] == ["Task 2"]

"""FastAPI backend endpoints: /chat plus the direct verification endpoints."""
from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch, db_path):
    from app import config

    monkeypatch.setattr(config.settings, "database_path", db_path)

    # Import main after patching settings and after LLM_PROVIDER=mock is set
    # by conftest, and force a fresh Assistant bound to the temp db.
    from app import main as main_module
    from app.assistant import Assistant

    main_module._assistant = Assistant(db_path=db_path)
    return TestClient(main_module.app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_chat_creates_task_and_verification_endpoint_sees_it(client):
    session_id = f"api-test-{uuid.uuid4().hex[:8]}"
    resp = client.post(
        "/chat", json={"session_id": session_id, "message": "add a task to buy milk"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["tool_calls"][0]["tool"] == "create_task"

    tasks_resp = client.get(f"/sessions/{session_id}/tasks")
    assert tasks_resp.status_code == 200
    tasks = tasks_resp.json()["tasks"]
    assert any("milk" in t["title"].lower() for t in tasks)


def test_chat_multi_step_creates_task_and_reminder(client):
    session_id = f"api-test-{uuid.uuid4().hex[:8]}"
    resp = client.post(
        "/chat",
        json={
            "session_id": session_id,
            "message": "add a task to call the dentist and remind me tomorrow at 9am",
        },
    )
    assert resp.status_code == 200
    tools_called = [c["tool"] for c in resp.json()["tool_calls"]]
    assert tools_called == ["create_task", "create_reminder"]

    reminders_resp = client.get(f"/sessions/{session_id}/reminders")
    assert len(reminders_resp.json()["reminders"]) == 1

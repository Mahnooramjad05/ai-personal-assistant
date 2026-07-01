"""A multi-step request correctly triggers two tool calls in the right
order, using the mock LLM path (no real API key / network needed)."""
from __future__ import annotations

from app import config
from app.assistant import Assistant
from app.llm_client import MockLLMClient


def test_multi_step_request_triggers_task_then_reminder(monkeypatch, db_path, session_id):
    monkeypatch.setattr(config.settings, "database_path", db_path)
    assistant = Assistant(llm=MockLLMClient(), db_path=db_path)

    result = assistant.chat(
        session_id,
        "add a task to call the dentist and remind me tomorrow at 9am",
    )

    tools_called = [c["tool"] for c in result.tool_calls]
    assert tools_called == ["create_task", "create_reminder"], tools_called

    # The reply should reference both actions actually taken.
    assert "dentist" in result.reply.lower()

    # Verify against the store directly - independent confirmation the
    # tools genuinely ran, not just that the LLM claimed they did.
    tasks = assistant.get_tasks(session_id)
    reminders = assistant.get_reminders(session_id)
    assert any("dentist" in t["title"].lower() for t in tasks)
    assert len(reminders) == 1


def test_single_step_request_triggers_one_tool_call(monkeypatch, db_path, session_id):
    monkeypatch.setattr(config.settings, "database_path", db_path)
    assistant = Assistant(llm=MockLLMClient(), db_path=db_path)

    result = assistant.chat(session_id, "add a task to buy milk")

    assert [c["tool"] for c in result.tool_calls] == ["create_task"]
    tasks = assistant.get_tasks(session_id)
    assert any("milk" in t["title"].lower() for t in tasks)


def test_knowledge_lookup_triggers_search_tool(monkeypatch, db_path, session_id):
    monkeypatch.setattr(config.settings, "database_path", db_path)
    assistant = Assistant(llm=MockLLMClient(), db_path=db_path)

    result = assistant.chat(session_id, "what is Python?")

    assert [c["tool"] for c in result.tool_calls] == ["search_knowledge"]
    tool_result = result.tool_calls[0]["result"]
    assert tool_result["found"] is True
    assert "python" in tool_result["answer"].lower()


def test_reminder_query_resolves_against_stored_due_dates(monkeypatch, db_path, session_id):
    monkeypatch.setattr(config.settings, "database_path", db_path)
    assistant = Assistant(llm=MockLLMClient(), db_path=db_path)

    assistant.chat(session_id, "remind me to drink water today at 5pm")
    assistant.chat(session_id, "remind me to submit the report tomorrow at 9am")

    result = assistant.chat(session_id, "what are my reminders for today")

    tools_called = [c["tool"] for c in result.tool_calls]
    assert tools_called == ["list_reminders"]
    listed = result.tool_calls[0]["result"]["reminders"]
    assert len(listed) == 1
    assert "water" in listed[0]["message"].lower()

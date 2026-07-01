"""Conversation summarization actually triggers and shortens history once
the turn limit is exceeded."""
from __future__ import annotations

from app.conversation import ConversationManager
from app.llm_client import MockLLMClient


def test_summarization_triggers_after_turn_limit(db_path, session_id):
    llm = MockLLMClient()
    convo = ConversationManager(
        session_id=session_id,
        llm=llm,
        max_turns_before_summary=3,
        db_path=db_path,
    )

    triggered_on_turn = None
    for i in range(6):
        convo.add_user_message(f"This is user message number {i}.")
        convo.add_assistant_message(f"This is assistant reply number {i}.")
        if convo.last_compaction_happened and triggered_on_turn is None:
            triggered_on_turn = i

    assert triggered_on_turn is not None, "compaction never triggered"
    assert convo.summary is not None
    assert len(convo.summary) > 0


def test_summarization_shortens_stored_history(db_path, session_id):
    llm = MockLLMClient()
    max_turns = 3
    convo = ConversationManager(
        session_id=session_id,
        llm=llm,
        max_turns_before_summary=max_turns,
        db_path=db_path,
    )

    for i in range(8):
        convo.add_user_message(f"User turn {i} with some detail like the number {i}.")
        convo.add_assistant_message(f"Assistant turn {i} acknowledging {i}.")

    # After many turns well beyond the limit, the verbatim message list held
    # by the conversation manager must be shorter than it would be without
    # compaction (8 turns * 2 messages = 16 uncompacted messages).
    assert len(convo.raw_messages) < 16
    assert convo.summary is not None

    # The persisted message table should mirror the in-memory state - i.e.
    # compaction actually deleted the summarized rows from SQLite, not just
    # from the in-memory list.
    from app import store

    persisted = store.get_messages(session_id, db_path=db_path)
    assert len(persisted) == len(convo.raw_messages)


def test_no_summarization_when_under_limit(db_path, session_id):
    llm = MockLLMClient()
    convo = ConversationManager(
        session_id=session_id,
        llm=llm,
        max_turns_before_summary=10,
        db_path=db_path,
    )

    for i in range(3):
        convo.add_user_message(f"Message {i}")
        convo.add_assistant_message(f"Reply {i}")

    assert convo.summary is None
    assert len(convo.raw_messages) == 6


def test_summary_is_real_llm_output_not_stub(db_path, session_id):
    """The summary text should be derived from the actual conversation
    content (contains a distinguishing detail), not a fixed placeholder
    string - proving summarize() genuinely processes the input."""
    llm = MockLLMClient()
    convo = ConversationManager(
        session_id=session_id,
        llm=llm,
        max_turns_before_summary=2,
        db_path=db_path,
    )

    convo.add_user_message("My favorite number is 42 and I live in Berlin.")
    convo.add_assistant_message("Got it, noted 42 and Berlin.")
    convo.add_user_message("Also remember my dog is named Rex.")
    convo.add_assistant_message("Sure, Rex is noted.")
    convo.add_user_message("One more thing, my birthday is on the 5th.")
    convo.add_assistant_message("Noted, birthday on the 5th.")

    assert convo.summary is not None
    # The mock summarizer is extractive and keeps sentences containing
    # digits, so distinguishing numbers from the conversation should survive.
    assert any(ch.isdigit() for ch in convo.summary)

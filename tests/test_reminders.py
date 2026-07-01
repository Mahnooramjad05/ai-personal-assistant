"""Reminder due-date querying returns the correct subset."""
from __future__ import annotations

from datetime import datetime, timedelta

from app import store
from app.tools.datetime_parse import day_range, parse_due_datetime


def test_parse_due_datetime_tomorrow_at_9am():
    now = datetime(2026, 7, 1, 12, 0)
    due = parse_due_datetime("tomorrow at 9am", now=now)
    assert due.date() == (now + timedelta(days=1)).date()
    assert due.hour == 9
    assert due.minute == 0


def test_parse_due_datetime_today_default_time():
    now = datetime(2026, 7, 1, 7, 0)
    due = parse_due_datetime("today", now=now)
    assert due.date() == now.date()
    assert due.hour == 9  # default morning time


def test_parse_due_datetime_rolls_to_tomorrow_if_time_passed():
    now = datetime(2026, 7, 1, 15, 0)  # 3pm
    due = parse_due_datetime("at 9am", now=now)  # no explicit day -> rolls forward
    assert due.date() == (now + timedelta(days=1)).date()
    assert due.hour == 9


def test_list_reminders_for_today_excludes_tomorrow(db_path, session_id):
    now = datetime.now()
    today_due = now.replace(hour=18, minute=0, second=0, microsecond=0)
    tomorrow_due = (now + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)

    store.create_reminder(session_id, "Today's reminder", today_due, db_path=db_path)
    store.create_reminder(session_id, "Tomorrow's reminder", tomorrow_due, db_path=db_path)

    start, end = day_range("today", now=now)
    todays = store.list_reminders_in_range(session_id, start, end, db_path=db_path)

    assert len(todays) == 1
    assert todays[0].message == "Today's reminder"


def test_list_reminders_for_tomorrow_returns_correct_subset(db_path, session_id):
    now = datetime.now()
    today_due = now.replace(hour=18, minute=0, second=0, microsecond=0)
    tomorrow_due = (now + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
    day_after_due = (now + timedelta(days=2)).replace(hour=9, minute=0, second=0, microsecond=0)

    store.create_reminder(session_id, "Today's reminder", today_due, db_path=db_path)
    store.create_reminder(session_id, "Tomorrow's reminder", tomorrow_due, db_path=db_path)
    store.create_reminder(session_id, "Day after reminder", day_after_due, db_path=db_path)

    start, end = day_range("tomorrow", now=now)
    tomorrows = store.list_reminders_in_range(session_id, start, end, db_path=db_path)

    assert len(tomorrows) == 1
    assert tomorrows[0].message == "Tomorrow's reminder"


def test_reminder_tools_respect_configured_db(monkeypatch, db_path, session_id):
    """reminder_tools calls app.store without an explicit db_path, so it
    relies on app.config.settings.database_path - verify that redirect
    actually takes effect end to end."""
    from app import config
    from app.tools import reminder_tools

    monkeypatch.setattr(config.settings, "database_path", db_path)

    result = reminder_tools.create_reminder(session_id, "Water the plants", "today at 8pm")
    assert result["message"] == "Water the plants"

    listing = reminder_tools.list_reminders(session_id, "today")
    assert any(r["message"] == "Water the plants" for r in listing["reminders"])

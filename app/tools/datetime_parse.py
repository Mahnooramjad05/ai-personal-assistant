"""Small, dependency-free natural-language date/time parser for reminders.

This intentionally covers a focused set of phrasings ("today", "tomorrow",
"tonight", "next week", "at 9am", "on friday") rather than depending on an
external NLP date library, keeping the project's dependency footprint small
and fully deterministic for tests.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta

_WEEKDAYS = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
]


def _extract_clock_time(text: str) -> tuple[int, int] | None:
    match = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", text, flags=re.I)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    meridian = (match.group(3) or "").lower()
    if meridian == "pm" and hour < 12:
        hour += 12
    if meridian == "am" and hour == 12:
        hour = 0
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return hour, minute


def parse_due_datetime(when_text: str, now: datetime | None = None) -> datetime:
    """Parse a natural-language time expression into a concrete datetime.

    Defaults sensibly when information is missing: no explicit clock time
    defaults to 09:00; no explicit day defaults to today (or tomorrow if
    that time has already passed today).
    """
    now = now or datetime.now()
    text = when_text.strip().lower()

    base_day = now
    if "tomorrow" in text:
        base_day = now + timedelta(days=1)
    elif "tonight" in text:
        base_day = now
    elif "next week" in text:
        base_day = now + timedelta(days=7)
    else:
        for i, day_name in enumerate(_WEEKDAYS):
            if day_name in text:
                days_ahead = (i - now.weekday()) % 7
                days_ahead = days_ahead or 7  # "on monday" means next monday
                base_day = now + timedelta(days=days_ahead)
                break
        # "today" or no day mentioned -> base_day stays as now

    clock = _extract_clock_time(text)
    if clock:
        hour, minute = clock
    elif "tonight" in text:
        hour, minute = 20, 0
    elif "morning" in text:
        hour, minute = 9, 0
    elif "afternoon" in text:
        hour, minute = 14, 0
    elif "evening" in text:
        hour, minute = 18, 0
    else:
        hour, minute = 9, 0

    candidate = base_day.replace(hour=hour, minute=minute, second=0, microsecond=0)

    # If the resulting time is in the past and no explicit day/relative
    # phrase was given, roll forward to tomorrow (e.g. "remind me at 9am"
    # asked at 3pm should mean tomorrow morning, not today in the past).
    explicit_day_given = any(
        kw in text for kw in ("tomorrow", "tonight", "next week")
    ) or any(day in text for day in _WEEKDAYS)
    if not explicit_day_given and candidate < now:
        candidate += timedelta(days=1)

    return candidate


def day_range(when: str, now: datetime | None = None) -> tuple[datetime, datetime]:
    """Return [start, end) datetime bounds for 'today' / 'tomorrow' / 'week'."""
    now = now or datetime.now()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    when = when.strip().lower()
    if when == "tomorrow":
        start = today_start + timedelta(days=1)
        end = start + timedelta(days=1)
    elif when == "week":
        start = today_start
        end = start + timedelta(days=7)
    else:  # "today" and any unrecognized value defaults to today
        start = today_start
        end = start + timedelta(days=1)
    return start, end

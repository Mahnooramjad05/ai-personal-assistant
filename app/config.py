"""Central configuration loaded from environment variables (.env supported)."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load .env once at import time so every module sees the same environment.
load_dotenv()


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class Settings:
    # Which LLM provider to use: "anthropic", "openai", or "mock".
    # Defaults to "mock" so the project runs fully offline with no API key.
    llm_provider: str = os.getenv("LLM_PROVIDER", "mock").strip().lower()

    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-8")

    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Number of user/assistant turn-pairs kept verbatim before the older
    # portion of the conversation is summarized/compacted.
    max_turns_before_summary: int = _get_int("MAX_TURNS_BEFORE_SUMMARY", 8)

    # SQLite database file location (created at runtime, not checked in).
    database_path: str = os.getenv("DATABASE_PATH", "assistant.db")

    # Backend host/port for reference by the Streamlit app / docs.
    backend_host: str = os.getenv("BACKEND_HOST", "127.0.0.1")
    backend_port: int = _get_int("BACKEND_PORT", 8000)


settings = Settings()

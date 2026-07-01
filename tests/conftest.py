"""Shared pytest fixtures.

Every test gets its own throwaway SQLite file so tests never interfere with
each other or with a developer's real assistant.db, and never touch the
network (LLM_PROVIDER is forced to "mock" here regardless of the local
.env, satisfying the "no real API key / no live network call" requirement).
"""
from __future__ import annotations

import os

os.environ["LLM_PROVIDER"] = "mock"

import uuid

import pytest

from app import store
from app.llm_client import MockLLMClient


@pytest.fixture
def db_path(tmp_path) -> str:
    path = str(tmp_path / f"test_{uuid.uuid4().hex}.db")
    store.init_db(path)
    return path


@pytest.fixture
def session_id() -> str:
    return f"test-session-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def mock_llm() -> MockLLMClient:
    return MockLLMClient()

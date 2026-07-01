"""Thin wrapper exposing knowledge-base retrieval as an orchestrator tool."""
from __future__ import annotations

from typing import Any

from app.tools.retrieval import search_knowledge_base


def search_knowledge(query: str) -> dict[str, Any]:
    result = search_knowledge_base(query)
    return {
        "found": result.found,
        "answer": result.answer,
        "source": result.source,
        "score": result.score,
    }

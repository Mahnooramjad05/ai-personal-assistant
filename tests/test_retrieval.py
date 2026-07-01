"""Retrieval tool: genuine keyword/TF-IDF search over the bundled local
knowledge base (no hardcoded answers, no network)."""
from __future__ import annotations

from app.tools.retrieval import search_knowledge_base


def test_retrieval_finds_relevant_snippet_for_python():
    result = search_knowledge_base("Tell me about the Python programming language")
    assert result.found is True
    assert result.source == "python"
    assert "python" in result.answer.lower()


def test_retrieval_finds_relevant_snippet_for_solar_system():
    result = search_knowledge_base("How many planets are in the solar system?")
    assert result.found is True
    assert result.source == "solar_system"


def test_retrieval_returns_not_found_for_unrelated_query():
    result = search_knowledge_base("asdkjhqwe unrelated gibberish query zzz")
    assert result.found is False
    assert result.answer is None


def test_retrieval_is_not_hardcoded_different_queries_different_sources():
    """Verify this is genuine retrieval, not a fixed canned answer, by
    confirming distinct queries surface distinct, topically-correct sources."""
    python_result = search_knowledge_base("python programming language history")
    ww2_result = search_knowledge_base("World War Two Axis and Allied powers")
    body_result = search_knowledge_base("How many bones are in the human body?")

    assert python_result.source == "python"
    assert ww2_result.source == "world_war_two"
    assert body_result.source == "human_body"
    assert python_result.answer != ww2_result.answer != body_result.answer

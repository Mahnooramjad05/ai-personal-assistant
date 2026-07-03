"""Provider selection and tool-call normalization for the Hermes backend.

Hermes (NousResearch) models are served behind an OpenAI-compatible
chat-completions endpoint (Ollama, vLLM, LM Studio, OpenRouter, ...), so
`HermesLLMClient` reuses the `openai` SDK pointed at a custom `base_url`.
These tests stub the underlying `openai` client's
`chat.completions.create` call - no real network access, no live Hermes/
Ollama server required - matching how the rest of the suite forces
LLM_PROVIDER=mock and never talks to a real LLM.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from app import config
from app.llm_client import HermesLLMClient, MockLLMClient, get_llm_client


def _make_response(message: SimpleNamespace) -> SimpleNamespace:
    """Build a minimal stand-in for an OpenAI ChatCompletion response."""
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_llm_provider_hermes_selects_hermes_client_with_configured_values(monkeypatch):
    monkeypatch.setattr(config.settings, "llm_provider", "hermes")
    monkeypatch.setattr(config.settings, "hermes_base_url", "http://localhost:11434/v1")
    monkeypatch.setattr(config.settings, "hermes_model", "hermes3")
    monkeypatch.setattr(config.settings, "hermes_api_key", "test-key-123")

    client = get_llm_client()

    assert isinstance(client, HermesLLMClient)
    # The provider must be built on the official `openai` SDK, just pointed
    # at a custom base_url, rather than a bespoke HTTP client.
    assert str(client._client.base_url) == "http://localhost:11434/v1/"
    assert client._client.api_key == "test-key-123"
    assert client._model == "hermes3"


def test_llm_provider_hermes_defaults_api_key_placeholder_when_unset(monkeypatch):
    # Many local servers (e.g. Ollama) don't require a real API key, but the
    # openai SDK requires a non-empty string - HermesLLMClient must default
    # to a placeholder rather than crash or pass None.
    monkeypatch.setattr(config.settings, "llm_provider", "hermes")
    monkeypatch.setattr(config.settings, "hermes_base_url", "http://localhost:11434/v1")
    monkeypatch.setattr(config.settings, "hermes_model", "hermes3")
    monkeypatch.setattr(config.settings, "hermes_api_key", None)

    client = get_llm_client()

    assert isinstance(client, HermesLLMClient)
    assert client._client.api_key == "not-needed"


def test_llm_provider_hermes_falls_back_to_mock_without_base_url(monkeypatch):
    # get_llm_client() degrades to the offline mock rather than hard-crashing
    # when a real provider is selected but not fully configured - mirrors
    # the existing anthropic/openai fallback behavior in get_llm_client().
    monkeypatch.setattr(config.settings, "llm_provider", "hermes")
    monkeypatch.setattr(config.settings, "hermes_base_url", None)

    client = get_llm_client()

    assert isinstance(client, MockLLMClient)


def _hermes_client(monkeypatch) -> HermesLLMClient:
    monkeypatch.setattr(config.settings, "hermes_base_url", "http://localhost:11434/v1")
    monkeypatch.setattr(config.settings, "hermes_model", "hermes3")
    monkeypatch.setattr(config.settings, "hermes_api_key", "not-needed")
    return HermesLLMClient()


def test_hermes_normalizes_structured_openai_style_tool_calls(monkeypatch):
    client = _hermes_client(monkeypatch)

    tool_call = SimpleNamespace(
        function=SimpleNamespace(
            name="create_task",
            arguments=json.dumps({"title": "call the dentist"}),
        )
    )
    stub_message = SimpleNamespace(content=None, tool_calls=[tool_call])
    stub_response = _make_response(stub_message)

    def fake_create(**kwargs):
        # The tools param must be sent in the OpenAI-compatible shape during
        # tool-selection turns.
        assert kwargs["tools"], "expected tools to be offered to the model"
        assert kwargs["tools"][0]["type"] == "function"
        return stub_response

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)

    raw = client.complete(
        messages=[{"role": "user", "content": "add a task to call the dentist"}],
        system="MODE=TOOL_SELECTION\n...",
    )

    parsed = json.loads(raw)
    assert parsed == {
        "tool_calls": [{"tool": "create_task", "args": {"title": "call the dentist"}}]
    }


def test_hermes_normalizes_inline_tool_call_tag_fallback(monkeypatch):
    client = _hermes_client(monkeypatch)

    tag_content = (
        '<tool_call>{"name": "create_task", '
        '"arguments": {"title": "call the dentist"}}</tool_call>'
    )
    stub_message = SimpleNamespace(content=tag_content, tool_calls=[])
    stub_response = _make_response(stub_message)

    monkeypatch.setattr(
        client._client.chat.completions, "create", lambda **kwargs: stub_response
    )

    raw = client.complete(
        messages=[{"role": "user", "content": "add a task to call the dentist"}],
        system="MODE=TOOL_SELECTION\n...",
    )

    parsed = json.loads(raw)
    # Must normalize to the exact same internal shape as the structured
    # tool_calls path above, regardless of which wire format the server
    # actually used.
    assert parsed == {
        "tool_calls": [{"tool": "create_task", "args": {"title": "call the dentist"}}]
    }


def test_hermes_falls_back_to_plain_text_when_no_tool_call_present(monkeypatch):
    client = _hermes_client(monkeypatch)

    stub_message = SimpleNamespace(content="Sure, how can I help?", tool_calls=[])
    stub_response = _make_response(stub_message)

    monkeypatch.setattr(
        client._client.chat.completions, "create", lambda **kwargs: stub_response
    )

    raw = client.complete(
        messages=[{"role": "user", "content": "hello"}],
        system="MODE=TOOL_SELECTION\n...",
    )

    assert raw == "Sure, how can I help?"


def test_hermes_does_not_offer_tools_outside_tool_selection_mode(monkeypatch):
    client = _hermes_client(monkeypatch)

    stub_message = SimpleNamespace(content="A short summary.", tool_calls=[])
    stub_response = _make_response(stub_message)

    def fake_create(**kwargs):
        assert "tools" not in kwargs
        return stub_response

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)

    reply = client.complete(
        messages=[{"role": "user", "content": "some conversation text"}],
        system="MODE=SUMMARY\n...",
    )

    assert reply == "A short summary."

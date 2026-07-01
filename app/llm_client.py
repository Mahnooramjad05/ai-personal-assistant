"""Pluggable LLM client.

Supports three backends, selected via the LLM_PROVIDER env var:
  - "anthropic": uses the official `anthropic` SDK (claude-opus-4-8 by default)
  - "openai": uses the official `openai` SDK
  - "mock": a deterministic, offline, no-network implementation used for
    local development and for the test suite (default, so the project runs
    out of the box without any API key).

All three implement the same minimal interface: `complete(messages, system)`.
`messages` is a list of {"role": "user"|"assistant", "content": str} dicts
(no tool-calling protocol here - tool selection is handled one level up, in
app/orchestrator.py, via a small JSON-based function-calling convention that
works identically across all three backends).
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any

from app.config import settings


class LLMClient(ABC):
    """Common interface for all LLM backends."""

    @abstractmethod
    def complete(self, messages: list[dict[str, str]], system: str) -> str:
        """Return the assistant's raw text reply for the given conversation."""
        raise NotImplementedError

    def summarize(self, conversation_text: str) -> str:
        """Summarize a chunk of conversation into a short paragraph.

        Default implementation just calls `complete` with a summarization
        instruction, so subclasses get this for free.
        """
        system = (
            "You are a conversation summarizer. Summarize the following "
            "conversation history into a short, dense paragraph that "
            "preserves names, dates, tasks, reminders, decisions and any "
            "facts that might matter later. Do not add commentary, just the "
            "summary."
        )
        return self.complete(
            messages=[{"role": "user", "content": conversation_text}],
            system=system,
        ).strip()


class MockLLMClient(LLMClient):
    """A deterministic, offline LLM stand-in.

    This is not a stub that fakes a single hardcoded answer - it implements
    real (if simple) natural-language-ish logic:
      - recognizes intents (task creation/listing/completion/deletion,
        reminder creation/querying, knowledge lookup) via keyword/regex
        pattern matching and emits the same structured tool-call JSON a real
        LLM would be asked to emit,
      - can chain two intents from one message (e.g. "add a task ... and
        remind me ..."),
      - produces a genuine extractive summary for conversation compaction
        (keeps the first and last sentence of each turn plus any sentence
        containing a date/number, rather than a hardcoded string).

    It exists so the whole system (orchestration, tool calling, multi-step
    requests, summarization) is exercisable with zero network access and no
    API key - required for the test-suite and for anyone cloning the repo
    without credentials.
    """

    def complete(self, messages: list[dict[str, str]], system: str) -> str:
        last_user = ""
        for m in reversed(messages):
            if m["role"] == "user":
                last_user = m["content"]
                break

        # The orchestrator asks us (via `system`) to either (a) choose tool
        # calls, or (b) synthesize a final answer from tool results. We tell
        # these apart by a marker the orchestrator includes in the system
        # prompt, keeping this mock fully decoupled from real prompt text.
        if "MODE=TOOL_SELECTION" in system:
            return self._select_tools(last_user)
        if "MODE=SYNTHESIS" in system:
            return self._synthesize(messages, last_user)
        if "MODE=SUMMARY" in system:
            return self._extractive_summary(last_user)
        # Generic chit-chat fallback.
        return self._chit_chat(last_user)

    # -- tool selection -----------------------------------------------
    def _select_tools(self, text: str) -> str:
        calls: list[dict[str, Any]] = []
        lowered = text.lower()

        # Split multi-step requests on " and " so "add a task X and remind
        # me Y" is treated as two clauses, each independently classified.
        clauses = re.split(r"\band then\b|\band also\b|\band\b", text, flags=re.I)
        if len(clauses) == 1:
            clauses = [text]

        for clause in clauses:
            call = self._classify_clause(clause, lowered_full=lowered)
            if call:
                # If this reminder has no message of its own (e.g. "remind
                # me tomorrow at 9am" following "add a task to call the
                # dentist"), reuse the subject of the immediately preceding
                # create_task call so the reminder is about the same thing.
                if call["tool"] == "create_reminder" and not call["args"].get("message"):
                    if calls and calls[-1]["tool"] == "create_task":
                        call["args"]["message"] = calls[-1]["args"]["title"]
                    else:
                        call["args"]["message"] = clause.strip() or "Reminder"
                calls.append(call)

        if not calls:
            return json.dumps({"tool_calls": [], "direct_reply": self._chit_chat(text)})
        return json.dumps({"tool_calls": calls})

    def _classify_clause(self, clause: str, lowered_full: str) -> dict[str, Any] | None:
        c = clause.strip()
        cl = c.lower()

        # Reminder query: "what are my reminders for today/tomorrow"
        if "reminder" in cl and any(w in cl for w in ("what", "list", "show", "any")):
            when = "today"
            if "tomorrow" in cl:
                when = "tomorrow"
            elif "week" in cl:
                when = "week"
            return {"tool": "list_reminders", "args": {"when": when}}

        # Reminder creation: "remind me to X at/on <time>"
        if "remind me" in cl or cl.startswith("reminder"):
            time_hint = self._extract_time_hint(c)
            # \b after "to" prevents this from also eating the "to" inside
            # "tomorrow" (e.g. "remind me tomorrow at 9am").
            message = re.sub(
                r"remind me(\s+to\b)?", "", c, flags=re.I
            ).strip(" ,.:;-")
            # Strip the time hint back out of the message when the user
            # gave no explicit reminder text (e.g. "remind me tomorrow at
            # 9am" with no task mentioned) so we don't store the due-date
            # phrase as the reminder's message.
            if time_hint and message.strip().lower() == time_hint.strip().lower():
                message = ""
            # Leave "message" empty (rather than falling back to the raw
            # clause `c`, which would include the time phrase) when no
            # reminder text was given - _select_tools fills it in from the
            # preceding clause's subject, or falls back to `c` as a last
            # resort if there is no preceding task to borrow from.
            return {
                "tool": "create_reminder",
                "args": {"message": message, "when_text": time_hint or "today"},
            }

        # Task listing
        if "task" in cl and any(w in cl for w in ("list", "show", "what are", "my tasks")):
            return {"tool": "list_tasks", "args": {}}

        # Task completion
        if "task" in cl and any(w in cl for w in ("complete", "done", "finish", "mark")):
            title = self._extract_quoted_or_after(c, ["complete", "done", "finish", "mark"])
            return {"tool": "complete_task", "args": {"title": title}}

        # Task deletion
        if "task" in cl and any(w in cl for w in ("delete", "remove", "cancel")):
            title = self._extract_quoted_or_after(c, ["delete", "remove", "cancel"])
            return {"tool": "delete_task", "args": {"title": title}}

        # Task creation
        if "task" in cl and any(w in cl for w in ("add", "create", "new")):
            title = re.sub(
                r"(add|create|new)?\s*(a\s+)?task( to)?", "", c, flags=re.I
            ).strip(" ,.:;-")
            return {"tool": "create_task", "args": {"title": title or c}}

        # Knowledge / information retrieval
        if any(
            w in cl
            for w in (
                "what is",
                "who is",
                "explain",
                "tell me about",
                "define",
                "how does",
                "search for",
            )
        ):
            return {"tool": "search_knowledge", "args": {"query": c}}

        return None

    @staticmethod
    def _extract_time_hint(text: str) -> str:
        # Collect every recognizable day-word AND clock-time fragment (not
        # just the first alternation match) so phrases like "today at 5pm"
        # keep both parts instead of losing the clock time.
        day_match = re.search(
            r"(tomorrow|tonight|next week|today|on \w+day)", text, flags=re.I
        )
        clock_match = re.search(
            r"at\s+\d{1,2}(:\d{2})?\s*(am|pm)?", text, flags=re.I
        )
        parts = [m.group(0) for m in (day_match, clock_match) if m]
        return " ".join(parts)

    @staticmethod
    def _extract_quoted_or_after(text: str, keywords: list[str]) -> str:
        quoted = re.search(r"['\"](.+?)['\"]", text)
        if quoted:
            return quoted.group(1)
        for kw in keywords:
            pattern = rf"{kw}\s+(?:the\s+|task\s+)?(.+)"
            m = re.search(pattern, text, flags=re.I)
            if m:
                return m.group(1).strip(" ,.:;-")
        return text.strip()

    # -- synthesis ------------------------------------------------------
    def _synthesize(self, messages: list[dict[str, str]], last_user: str) -> str:
        # The orchestrator puts tool results into the last "user" message as
        # a JSON blob prefixed with TOOL_RESULTS=. Extract and phrase it.
        tool_results: list[dict[str, Any]] = []
        for m in reversed(messages):
            if m["role"] == "user" and "TOOL_RESULTS=" in m["content"]:
                raw = m["content"].split("TOOL_RESULTS=", 1)[1]
                try:
                    tool_results = json.loads(raw)
                except json.JSONDecodeError:
                    tool_results = []
                break

        if not tool_results:
            return self._chit_chat(last_user)

        parts = []
        for r in tool_results:
            tool = r.get("tool")
            result = r.get("result")
            if tool == "create_task":
                parts.append(f"I've added the task \"{result.get('title')}\" (id {result.get('id')}).")
            elif tool == "list_tasks":
                items = result.get("tasks", [])
                if not items:
                    parts.append("You have no tasks right now.")
                else:
                    listing = "; ".join(
                        f"#{t['id']} {t['title']} ({'done' if t['completed'] else 'pending'})"
                        for t in items
                    )
                    parts.append(f"Here are your tasks: {listing}.")
            elif tool == "complete_task":
                if result.get("found"):
                    parts.append(f"Marked \"{result.get('title')}\" as complete.")
                else:
                    parts.append("I couldn't find that task to complete.")
            elif tool == "delete_task":
                if result.get("found"):
                    parts.append(f"Deleted the task \"{result.get('title')}\".")
                else:
                    parts.append("I couldn't find that task to delete.")
            elif tool == "create_reminder":
                parts.append(
                    f"Reminder set: \"{result.get('message')}\" for "
                    f"{result.get('due_at')}."
                )
            elif tool == "list_reminders":
                items = result.get("reminders", [])
                when = result.get("when", "the requested period")
                if not items:
                    parts.append(f"You have no reminders for {when}.")
                else:
                    listing = "; ".join(f"{r2['message']} at {r2['due_at']}" for r2 in items)
                    parts.append(f"Reminders for {when}: {listing}.")
            elif tool == "search_knowledge":
                snippet = result.get("answer") or result.get("snippet")
                if snippet:
                    parts.append(snippet)
                else:
                    parts.append("I couldn't find anything relevant in the knowledge base.")
            else:
                parts.append(str(result))

        return " ".join(parts)

    def _chit_chat(self, text: str) -> str:
        if not text.strip():
            return "I'm here - ask me to manage tasks, reminders, or look something up."
        return (
            "I can help with tasks, reminders, and quick lookups. "
            "Try: 'add a task to call the dentist and remind me tomorrow at 9am'."
        )

    # -- summarization ----------------------------------------------------
    def _extractive_summary(self, conversation_text: str) -> str:
        lines = [l.strip() for l in conversation_text.split("\n") if l.strip()]
        if not lines:
            return "No prior context."
        kept: list[str] = []
        for line in lines:
            sentences = re.split(r"(?<=[.!?])\s+", line)
            if not sentences:
                continue
            interesting = [
                s for s in sentences if re.search(r"\d", s)
            ]
            chosen = interesting[:1] or sentences[:1]
            kept.extend(chosen)
        summary = " ".join(kept)
        # Keep it dense: cap length so repeated summarization never grows
        # without bound.
        if len(summary) > 600:
            summary = summary[:600].rsplit(" ", 1)[0] + "..."
        return f"Summary of earlier conversation: {summary}"


class AnthropicLLMClient(LLMClient):
    """Real Claude backend via the official `anthropic` SDK."""

    def __init__(self) -> None:
        import anthropic  # imported lazily so it's optional at runtime

        if not settings.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set but LLM_PROVIDER=anthropic"
            )
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._model = settings.anthropic_model

    def complete(self, messages: list[dict[str, str]], system: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system,
            messages=[{"role": m["role"], "content": m["content"]} for m in messages],
        )
        text_blocks = [b.text for b in response.content if b.type == "text"]
        return "\n".join(text_blocks).strip()


class OpenAILLMClient(LLMClient):
    """Real GPT backend via the official `openai` SDK."""

    def __init__(self) -> None:
        from openai import OpenAI  # imported lazily so it's optional at runtime

        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set but LLM_PROVIDER=openai")
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_model

    def complete(self, messages: list[dict[str, str]], system: str) -> str:
        chat_messages = [{"role": "system", "content": system}]
        chat_messages.extend({"role": m["role"], "content": m["content"]} for m in messages)
        response = self._client.chat.completions.create(
            model=self._model,
            messages=chat_messages,
        )
        return (response.choices[0].message.content or "").strip()


def get_llm_client() -> LLMClient:
    """Factory that returns the configured LLM backend.

    Falls back to the mock client if a real provider is selected but not
    fully configured (e.g. missing API key), so the app never hard-crashes
    just because a key is missing - it degrades to offline mode instead.
    """
    provider = settings.llm_provider
    if provider == "anthropic":
        try:
            return AnthropicLLMClient()
        except Exception:
            return MockLLMClient()
    if provider == "openai":
        try:
            return OpenAILLMClient()
        except Exception:
            return MockLLMClient()
    return MockLLMClient()

"""LangGraph-based orchestration loop.

Graph shape:

    user message -> [select_tools] -> [execute_tools] -> [synthesize] -> END
                                            ^   |
                                            |___| (loops while more tool
                                                    calls are pending - the
                                                    LLM can request another
                                                    round of tools after
                                                    seeing results)

`select_tools` asks the LLM (mock or real) to decide which tool(s) to call
for the user's message - this is where multi-step requests like "add a task
to call the dentist and remind me tomorrow at 9am" get decomposed into an
ordered list of tool calls (create_task, then create_reminder). `execute_tools`
actually runs each requested tool against the SQLite-backed task/reminder
store or the knowledge-base retrieval tool, in order. `synthesize` asks the
LLM to turn the raw tool results into a natural-language reply that
references what actually happened.

This is real LangGraph usage (StateGraph, not a hand-rolled if/else loop) so
the orchestration is genuinely graph-based and inspectable/extensible.
"""
from __future__ import annotations

import json
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from app.llm_client import LLMClient
from app.tools import knowledge_tools, reminder_tools, task_tools

TOOL_SELECTION_SYSTEM = (
    "MODE=TOOL_SELECTION\n"
    "You are the planning component of an agentic personal assistant. "
    "Given the user's latest message and the conversation so far, decide "
    "which tool(s) to call, in order. Respond ONLY with JSON of the shape "
    '{"tool_calls": [{"tool": "<name>", "args": {...}}, ...]} '
    "or {\"tool_calls\": [], \"direct_reply\": \"...\"} if no tool is needed. "
    "Available tools: create_task(title), list_tasks(), complete_task(title), "
    "delete_task(title), create_reminder(message, when_text), "
    "list_reminders(when), search_knowledge(query). "
    "A single user message may require multiple tool calls in sequence "
    "(e.g. creating a task AND a reminder)."
)

SYNTHESIS_SYSTEM = (
    "MODE=SYNTHESIS\n"
    "You are a helpful personal assistant. You previously decided to call "
    "some tools; their results are provided. Write a single, natural, "
    "concise reply to the user that accurately reflects those results. "
    "Do not invent information not present in the tool results."
)

MAX_TOOL_ROUNDS = 3

TOOL_DISPATCH = {
    "create_task": lambda session_id, args: task_tools.create_task(
        session_id, args.get("title", "")
    ),
    "list_tasks": lambda session_id, args: task_tools.list_tasks(session_id),
    "complete_task": lambda session_id, args: task_tools.complete_task(
        session_id, args.get("title", "")
    ),
    "delete_task": lambda session_id, args: task_tools.delete_task(
        session_id, args.get("title", "")
    ),
    "create_reminder": lambda session_id, args: reminder_tools.create_reminder(
        session_id, args.get("message", ""), args.get("when_text", "today")
    ),
    "list_reminders": lambda session_id, args: reminder_tools.list_reminders(
        session_id, args.get("when", "today")
    ),
    "search_knowledge": lambda session_id, args: knowledge_tools.search_knowledge(
        args.get("query", "")
    ),
}


class GraphState(TypedDict, total=False):
    session_id: str
    llm_messages: list[dict[str, str]]
    pending_tool_calls: list[dict[str, Any]]
    executed_tool_calls: list[dict[str, Any]]
    direct_reply: str | None
    final_reply: str
    rounds: int


def _select_tools_node(llm: LLMClient):
    def node(state: GraphState) -> GraphState:
        raw = llm.complete(state["llm_messages"], system=TOOL_SELECTION_SYSTEM)
        parsed = _safe_json(raw)
        tool_calls = parsed.get("tool_calls", []) if parsed else []
        direct_reply = parsed.get("direct_reply") if parsed else None
        return {
            **state,
            "pending_tool_calls": tool_calls,
            "direct_reply": direct_reply,
            "rounds": state.get("rounds", 0) + 1,
        }

    return node


def _execute_tools_node(state: GraphState) -> GraphState:
    session_id = state["session_id"]
    executed = list(state.get("executed_tool_calls", []))
    for call in state.get("pending_tool_calls", []):
        tool_name = call.get("tool")
        args = call.get("args", {}) or {}
        handler = TOOL_DISPATCH.get(tool_name)
        if handler is None:
            executed.append({"tool": tool_name, "args": args, "result": {"error": "unknown tool"}})
            continue
        result = handler(session_id, args)
        executed.append({"tool": tool_name, "args": args, "result": result})
    return {**state, "executed_tool_calls": executed, "pending_tool_calls": []}


def _synthesize_node(llm: LLMClient):
    def node(state: GraphState) -> GraphState:
        if not state.get("executed_tool_calls"):
            reply = state.get("direct_reply") or "How can I help?"
            return {**state, "final_reply": reply}

        tool_results_json = json.dumps(state["executed_tool_calls"])
        synth_messages = list(state["llm_messages"])
        synth_messages.append(
            {
                "role": "user",
                "content": f"TOOL_RESULTS={tool_results_json}",
            }
        )
        reply = llm.complete(synth_messages, system=SYNTHESIS_SYSTEM)
        return {**state, "final_reply": reply}

    return node


def _should_continue(state: GraphState) -> str:
    # Stop once we've executed something and hit the round cap, or once a
    # selection round produced no tool calls at all.
    if not state.get("pending_tool_calls") and not state.get("executed_tool_calls"):
        return "synthesize"
    if state.get("rounds", 0) >= MAX_TOOL_ROUNDS:
        return "synthesize"
    return "execute"


def _safe_json(raw: str) -> dict[str, Any] | None:
    raw = raw.strip()
    # Tolerate accidental markdown code fences from real LLMs.
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def build_graph(llm: LLMClient):
    graph = StateGraph(GraphState)
    graph.add_node("select_tools", _select_tools_node(llm))
    graph.add_node("execute", _execute_tools_node)
    graph.add_node("synthesize", _synthesize_node(llm))

    graph.set_entry_point("select_tools")
    graph.add_conditional_edges(
        "select_tools",
        _should_continue,
        {"execute": "execute", "synthesize": "synthesize"},
    )
    graph.add_edge("execute", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()


def run_orchestration(
    llm: LLMClient, session_id: str, llm_messages: list[dict[str, str]]
) -> tuple[str, list[dict[str, Any]]]:
    """Run one turn of the orchestration graph and return (reply, tool_calls)."""
    compiled = build_graph(llm)
    initial_state: GraphState = {
        "session_id": session_id,
        "llm_messages": llm_messages,
        "pending_tool_calls": [],
        "executed_tool_calls": [],
        "rounds": 0,
    }
    result = compiled.invoke(initial_state)
    return result.get("final_reply", ""), result.get("executed_tool_calls", [])

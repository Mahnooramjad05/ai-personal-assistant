"""FastAPI backend exposing the assistant over REST.

Endpoints:
    POST /chat                      -> session_id + message -> assistant reply + tool calls
    GET  /sessions/{id}/tasks       -> list tasks for a session (verification, independent of chat)
    GET  /sessions/{id}/reminders   -> list reminders for a session
    GET  /sessions/{id}/history     -> raw stored message history
    GET  /sessions/{id}/summary     -> latest conversation summary, if any
    GET  /health                   -> liveness probe
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from app.assistant import Assistant

app = FastAPI(
    title="AI Personal Assistant (Agentic)",
    description=(
        "LLM-powered agent with conversational context, tool/function "
        "calling, and LangGraph workflow orchestration."
    ),
    version="0.1.0",
)

_assistant = Assistant()


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ToolCallOut(BaseModel):
    tool: str
    args: dict
    result: dict


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    tool_calls: list[ToolCallOut]
    compaction_triggered: bool
    turn_count: int


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    result = _assistant.chat(request.session_id, request.message)
    return ChatResponse(
        session_id=request.session_id,
        reply=result.reply,
        tool_calls=[
            ToolCallOut(tool=c["tool"], args=c.get("args", {}), result=c.get("result", {}))
            for c in result.tool_calls
        ],
        compaction_triggered=result.compaction_triggered,
        turn_count=result.turn_count,
    )


@app.get("/sessions/{session_id}/tasks")
def get_tasks(session_id: str) -> dict:
    return {"session_id": session_id, "tasks": _assistant.get_tasks(session_id)}


@app.get("/sessions/{session_id}/reminders")
def get_reminders(session_id: str) -> dict:
    return {"session_id": session_id, "reminders": _assistant.get_reminders(session_id)}


@app.get("/sessions/{session_id}/history")
def get_history(session_id: str) -> dict:
    return {"session_id": session_id, "history": _assistant.get_history(session_id)}


@app.get("/sessions/{session_id}/summary")
def get_summary(session_id: str) -> dict:
    return {"session_id": session_id, "summary": _assistant.get_summary(session_id)}

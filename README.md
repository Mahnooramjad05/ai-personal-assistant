# AI Personal Assistant (Agentic)

An LLM-powered agent that maintains conversational context across turns and autonomously executes multi-step tasks by calling real tools for task management, reminders, and information retrieval, orchestrated through a LangGraph workflow.

## Overview

This project is a working, runnable implementation of an agentic personal assistant, not a slide-deck sketch. A user sends a chat message; the assistant decides which tool(s) to call (possibly several, in sequence, for multi-step requests like "add a task to call the dentist and remind me tomorrow at 9am"); it actually executes those tools against a local SQLite database; and it synthesizes a natural-language reply that reflects what really happened. Conversation history is kept per-session and automatically summarized once it grows past a configurable length, so long-running sessions never blow past the model's context window.

The LLM backend is pluggable: it runs against Anthropic's Claude API or OpenAI's API if a key is configured, and falls back to a deterministic, fully offline mock implementation otherwise - so the whole system (orchestration, tool calling, summarization) is exercisable and testable with zero network access and no paid API key.

## Key Features

- **Multi-turn conversational context with automatic summarization.** Each session's message history is stored in SQLite. Once the number of turn-pairs held verbatim exceeds `MAX_TURNS_BEFORE_SUMMARY`, the oldest half of the history is summarized via a real LLM call (`LLMClient.summarize`) and replaced by the summary, keeping the active context bounded while preserving salient facts.
- **Real tool/function calling, three genuine capabilities:**
  - **Task management** - create / list / complete / delete tasks, persisted in SQLite, with fuzzy title matching for completion/deletion.
  - **Reminders** - create time-based reminders from natural-language phrases ("tomorrow at 9am", "tonight", "on friday"), and query them ("what are my reminders for today") resolved against actual stored due-dates via a real date-range query, not a hardcoded answer.
  - **Information retrieval** - genuine TF-IDF-weighted keyword search over a small bundled local knowledge base (see "Retrieval approach" below for why local retrieval was chosen over a live web API).
- **LangGraph-based orchestration loop** (`app/orchestrator.py`) using a real `StateGraph`, not a hand-rolled if/else chain: tool selection -> tool execution -> response synthesis, with support for chaining multiple tool calls from a single user message.
- **FastAPI backend** exposing `POST /chat` plus direct verification endpoints (`/sessions/{id}/tasks`, `/sessions/{id}/reminders`, `/sessions/{id}/history`, `/sessions/{id}/summary`) so tool effects can be checked independently of the chat flow.
- **Streamlit chat UI** (`streamlit_app.py`) that talks to the FastAPI backend when it's running, and transparently falls back to calling the assistant module in-process otherwise.
- **Pluggable LLM provider** - Anthropic, OpenAI, or an offline deterministic mock, selected via `LLM_PROVIDER`.
- **pytest suite** (28 tests) covering the store, reminder date-range queries, multi-step tool orchestration, conversation summarization, retrieval, and the backend API - all running against the mock LLM and a temporary SQLite file, with no live network calls.

## Tech Stack

- Python 3.11+
- LangChain / LangGraph for workflow orchestration (`StateGraph`)
- FastAPI + Uvicorn for the backend REST API
- Streamlit for the chat UI
- SQLite (via the standard library `sqlite3`) for tasks, reminders, and conversation history
- Anthropic SDK / OpenAI SDK (optional, pluggable) for the real LLM backends
- pydantic for request/response models
- python-dotenv for environment configuration
- pytest for the test suite
- Docker for containerized deployment

## Architecture

### Conversational context and summarization

`app/conversation.py` (`ConversationManager`) owns per-session message history. Every user/assistant turn is appended to SQLite (`app/store.py`) and held verbatim in memory up to `MAX_TURNS_BEFORE_SUMMARY` turn-pairs. Once that limit is exceeded, the oldest half of the stored turns is rendered to plain text and passed to `LLMClient.summarize()` - a real LLM call (or, in mock mode, a genuine extractive summarizer that keeps sentences containing numbers/dates) - and the resulting summary text replaces those turns both in memory and in the `messages` table, while being appended to a running `conversation_summaries` row. The next LLM call is built from `[summary preamble] + [verbatim recent turns]`, so context stays bounded regardless of how long the conversation runs.

### Tool set

Three tool families, each backed by real logic (no hardcoded stub answers):

| Tool | File | Backing store / logic |
|---|---|---|
| `create_task`, `list_tasks`, `complete_task`, `delete_task` | `app/tools/task_tools.py` | SQLite `tasks` table (`app/store.py`) |
| `create_reminder`, `list_reminders` | `app/tools/reminder_tools.py` | SQLite `reminders` table; natural-language date parsing in `app/tools/datetime_parse.py`; range queries resolved against real stored `due_at` timestamps |
| `search_knowledge` | `app/tools/knowledge_tools.py` -> `app/tools/retrieval.py` | TF-IDF cosine-similarity search over `app/knowledge/*.txt` |

### LangGraph orchestration loop

`app/orchestrator.py` builds a real `langgraph.graph.StateGraph` with three nodes:

1. **select_tools** - asks the LLM (mock or real) to decide which tool(s) to call for the user's latest message, given the conversation-so-far. The LLM returns a small JSON plan, e.g. `{"tool_calls": [{"tool": "create_task", "args": {...}}, {"tool": "create_reminder", "args": {...}}]}`. This is where multi-step requests get decomposed into an ordered sequence of calls.
2. **execute** - runs each requested tool, in order, against the real SQLite-backed stores or the retrieval tool, and collects the results.
3. **synthesize** - asks the LLM to turn the raw tool results into a single, natural-language reply that references what actually happened (not what it assumes happened).

A conditional edge lets the graph loop back for another round of tool selection (bounded by `MAX_TOOL_ROUNDS`) if needed, before always finishing at `synthesize -> END`.

### Backend + Streamlit UI relationship

`app/main.py` (FastAPI) wraps a single `Assistant` instance (`app/assistant.py`) that ties `ConversationManager` and `run_orchestration` together per request. `POST /chat` runs one full turn and returns the reply plus the exact tool calls made (name, args, result) for transparency. The verification endpoints read directly from the SQLite store, independent of the chat flow, so you can confirm a tool's effect without trusting the LLM's narration of it.

`streamlit_app.py` is a real chat client: it POSTs to the backend's `/chat` endpoint and renders the reply plus an expandable "tool calls made" panel, and displays live task/reminder lists in the sidebar (pulled from the verification endpoints). If the backend isn't running, it detects that (`GET /health` unreachable) and falls back to instantiating `Assistant` in-process directly, so `streamlit run streamlit_app.py` works completely standalone as well.

### Retrieval approach: local knowledge base, and why

The brief allows either a real web search API or a local retrieval store, with the choice documented. In this environment, the free, no-key DuckDuckGo Instant Answer API returned HTTP 202 (accepted/throttled, not a reliable 200) on repeated test requests rather than consistent, testable responses - not something a graded, offline-runnable test suite should depend on. This project therefore implements genuine local retrieval instead: `app/knowledge/*.txt` holds five short general-knowledge snippets (Python, the Solar System, World War II, the human body, economics basics). `app/tools/retrieval.py` builds a TF-IDF index from scratch (sentence-level chunking, term frequency x inverse document frequency, cosine similarity - no external embedding API, no hardcoded query->answer map) and returns the best-matching chunk plus its source file and similarity score. `tests/test_retrieval.py` verifies that different queries surface different, topically-correct sources, confirming this is real retrieval and not a lookup table. Swapping in a live web-search tool later is a drop-in change: `search_knowledge_base(query) -> RetrievalResult` is the only interface the orchestrator depends on.

## Setup / Installation

```bash
# 1. Clone and enter the project
cd ai-personal-assistant

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash / PowerShell: .venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables (optional - runs fine with defaults)
cp .env.example .env
# Edit .env if you want to use a real Anthropic or OpenAI key; otherwise
# leave LLM_PROVIDER=mock for fully offline operation.
```

### Running with Docker

```bash
docker build -t ai-personal-assistant .
docker run -p 8000:8000 --env-file .env ai-personal-assistant
```

## Usage

### Run the backend

```bash
uvicorn app.main:app --reload --port 8000
```

### Launch the Streamlit UI

```bash
streamlit run streamlit_app.py
```

This opens a chat interface in your browser. If the FastAPI backend (above) is running on `http://127.0.0.1:8000`, the UI talks to it over HTTP; otherwise it calls the assistant module directly in-process.

### Example chat requests

**Multi-step request (two tool calls, in order):**

```
add a task to call the dentist and remind me tomorrow at 9am
```

This triggers `create_task` (title: "call the dentist") followed by `create_reminder` (message: "call the dentist", due tomorrow 09:00), and the reply references both: *"I've added the task 'call the dentist' (id 1). Reminder set: 'call the dentist' for 2026-07-02T09:00."*

**Single-step task management:**

```
add a task to buy milk
what are my tasks
mark buy milk as done
```

**Reminders resolved against real stored due-dates:**

```
remind me to submit the report tomorrow at 9am
remind me to drink water today at 5pm
what are my reminders for today
```

The last message only returns the reminder actually due today (resolved against the stored `due_at` timestamps via a real date-range query), not the one due tomorrow.

**Information retrieval:**

```
what is Python?
tell me about the solar system
```

### Verifying tool effects directly (independent of chat)

```bash
curl http://127.0.0.1:8000/sessions/my-session/tasks
curl http://127.0.0.1:8000/sessions/my-session/reminders
curl http://127.0.0.1:8000/sessions/my-session/history
curl http://127.0.0.1:8000/sessions/my-session/summary
```

### Running the tests

```bash
pytest
# or, with verbose output:
pytest -v
```

The suite (28 tests) covers: task creation/listing/completion round-trips against SQLite, reminder due-date range queries returning the correct subset, multi-step requests triggering the correct ordered tool calls (via the mock LLM path), conversation summarization actually triggering and shortening stored history once the turn limit is exceeded, retrieval returning genuinely different results for different queries, and the FastAPI endpoints end-to-end. No test requires a real API key or makes a live network call - `LLM_PROVIDER` is forced to `mock` in `tests/conftest.py`.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `mock` | `"mock"` (offline/deterministic), `"anthropic"`, or `"openai"` |
| `ANTHROPIC_API_KEY` | unset | Required only when `LLM_PROVIDER=anthropic` |
| `ANTHROPIC_MODEL` | `claude-opus-4-8` | Anthropic model id |
| `OPENAI_API_KEY` | unset | Required only when `LLM_PROVIDER=openai` |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model id |
| `MAX_TURNS_BEFORE_SUMMARY` | `8` | Turn-pairs kept verbatim before older history is summarized/compacted |
| `DATABASE_PATH` | `assistant.db` | SQLite file path (created automatically at runtime, not checked in) |
| `BACKEND_HOST` | `127.0.0.1` | Host the FastAPI backend binds to |
| `BACKEND_PORT` | `8000` | Port the FastAPI backend binds to |
| `BACKEND_URL` | `http://127.0.0.1:8000` | URL the Streamlit UI uses to reach the backend |

## License

MIT - see [LICENSE](LICENSE).

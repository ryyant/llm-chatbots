# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python chat program supporting multiple LLM providers (Google Gemini, OpenAI, Anthropic) with runtime model switching, streaming responses, and two interfaces:

- **CLI** (default) — terminal REPL.
- **Web** — Streamlit app with provider/model/system-prompt controls, streaming output, and PDF/txt/md uploads with retrieval-augmented question answering.

## Setup & Commands

Requires Python 3.10+. Use the project venv.

```bash
# Create venv (first time only)
python3.10 -m venv venv

# Activate venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the chat program
python main.py                # CLI mode (default)
python main.py --mode web     # Streamlit web UI

# Run tests
python -m pytest

# Run a single test
python -m pytest tests/test_foo.py::test_bar
```

## Architecture

- `main.py` — entry point; loads `.env`, parses `--mode {cli,web}`. CLI path runs the REPL; web path `os.execvp`'s `streamlit run app.py`. Exposes `handle_input(user_input, session, system_prompt, api_keys) -> HandleResult` for testability.
- `app.py` — Streamlit web UI. Sidebar holds provider/model/system-prompt controls, Apply/Clear buttons, file uploader, indexed-file list, and the "Use uploaded documents in answers" checkbox. Main area streams chat replies.
- `chat.py` — `ChatSession` class: thin wrapper over a provider, exposes `send(message) -> str`, `send_stream(message) -> Iterator[str]`, `history`, `current_provider`, `current_model`.
- `config.py` — shared constants used by both `main.py` and `app.py`: `VALID_PROVIDERS`, `DEFAULT_MODELS`, `API_KEY_ENV_VARS`, `DEFAULT_SYSTEM_PROMPT`, RAG knobs (`RAG_CHUNK_CHARS`, `RAG_OVERLAP_CHARS`, `RAG_TOP_K`, `RAG_EMBEDDING_MODEL`), `SUPPORTED_UPLOAD_EXTENSIONS`.
- `rag.py` — document handling, no Streamlit dependency: `extract_text(filename, data)` (pdf via `pypdf`, txt/md via decode), `chunk(text)` (paragraph-then-sentence with overlap), `embed(texts, api_key, embed_fn=None)` (Gemini `gemini-embedding-001`; `embed_fn` is for testing), `retrieve(query, index, api_key, top_k)` (numpy cosine similarity), `build_context_prompt(chunks, user_message)`. Defines the `Chunk` dataclass.
- `providers/base.py` — `BaseProvider` ABC with abstract `send(message) -> str`, abstract `send_stream(message) -> Iterator[str]`, and `history: list[dict]`.
- `providers/gemini.py` — `GeminiProvider`: uses `google-genai` stateful chat object; `send_stream` uses `send_message_stream`.
- `providers/openai.py` — `OpenAIProvider`: uses `openai` SDK, builds full message array per request; `send_stream` passes `stream=True`.
- `providers/anthropic.py` — `AnthropicProvider`: uses `anthropic` SDK, `system` is a top-level kwarg, `max_tokens=8192`; `send_stream` uses `messages.stream` context manager.
- `providers/__init__.py` — `create_provider(provider, api_key, model, system_prompt) -> BaseProvider` factory with deferred imports.
- `requirements.txt` — Python dependencies (`google-genai`, `openai`, `anthropic`, `python-dotenv`, `streamlit`, `pypdf`, `pytest`).

## Key Conventions

- API keys are stored in a `.env` file (see `.env.example`) and loaded via `python-dotenv` at startup; never hardcoded.
- Conversation history is kept as `list[{"role": "user"|"assistant", "content": str}]` on each provider and surfaced via `ChatSession.history`. Every `send` and `send_stream` appends both turns; on exception each provider rolls back the user message it just appended.
- `ChatSession.__init__` takes `provider`, `api_key`, `model`, and `system_prompt` — env-var loading happens in `main.py` / `app.py`, not inside the class.
- `create_provider` uses deferred imports so a missing SDK for one provider doesn't break the others.
- Switching models via `/model <provider>/<model>` in the REPL — or via the sidebar Apply button in the web UI — creates a fresh `ChatSession` (history cleared).
- Streaming: each provider implements `send_stream` as a generator. It must update `self.history` itself (user message appended before the first yield; full assembled assistant reply appended after the last yield).
- Web UI maintains `st.session_state.display_messages` separately from `session.history` because retrieved document chunks are prepended to the user message sent to the provider, but the UI shows the user's original input. The two histories diverge by design.
- Document uploads require `GEMINI_API_KEY` regardless of the chat provider, because embeddings always use Gemini.
- When the user removes a file or unchecks "Use uploaded documents" *after* augmented messages have been sent in the session, the web UI rebuilds the `ChatSession` and clears `display_messages`. This is the only way to actually remove document context from the model's memory (otherwise prior turns still reference the chunks). Tracked via `st.session_state.docs_used_in_session`.
- `rag.embed` accepts an `embed_fn` keyword argument so tests can inject a fake embedder without patching the Gemini SDK.

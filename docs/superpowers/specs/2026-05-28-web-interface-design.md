# Web Interface — Design Spec

**Date:** 2026-05-28
**Status:** Approved

## Overview

Add a Streamlit-based web UI as a second interface to the existing chat program. The CLI REPL remains the default; a `--mode` flag on `main.py` selects between CLI and web. The web UI exposes the same provider/model switching available in the CLI, adds a system-prompt editor and a clear-conversation button, and streams responses token-by-token.

## Architecture

One new file, three providers extended:

- **`app.py`** — NEW. Streamlit entry point. Renders sidebar controls and chat area. Owns `st.session_state["session"]`.
- **`main.py`** — Adds `--mode {cli,web}` argparse. CLI path is unchanged. Web path subprocess-execs `streamlit run app.py`.
- **`providers/base.py`** — Adds abstract `send_stream(message) -> Iterator[str]`.
- **`providers/{gemini,openai,anthropic}.py`** — Add `send_stream` implementations.
- **`chat.py`** — Adds `ChatSession.send_stream(message) -> Iterator[str]` mirroring `send`.

`ChatSession`, the existing `send` method, and the REPL behavior are unchanged.

## Components

### `main.py` — mode dispatch

- `argparse` adds `--mode {cli,web}`, default `cli`.
- `cli` mode: existing `main()` body runs unchanged.
- `web` mode: `os.execvp("streamlit", ["streamlit", "run", "app.py"])`. `execvp` replaces the Python process so the user sees Streamlit's output directly and Ctrl+C kills Streamlit cleanly.
- If `streamlit` is not on PATH, `execvp` raises `FileNotFoundError`; catch and print a hint to `pip install -r requirements.txt`.

### `providers/base.py`

Add one abstract method:

```python
@abstractmethod
def send_stream(self, message: str) -> Iterator[str]:
    """Send a message and yield response chunks as they arrive."""
```

Each implementation is responsible for updating `self.history` (append user message before the first yield, append assembled assistant reply after the last yield). On exception, roll back the user message the same way `send` does.

### Provider streaming implementations

- **Gemini:** `self._chat.send_message_stream(message)` returns an iterator of response chunks; yield `chunk.text` for each.
- **OpenAI:** `self._client.chat.completions.create(..., stream=True)` returns an iterator of `ChatCompletionChunk`; yield `chunk.choices[0].delta.content` when non-None.
- **Anthropic:** `self._client.messages.stream(...)` is a context manager; iterate `stream.text_stream` and yield each text delta.

Each implementation accumulates the full reply locally so it can append to `self.history` after the stream completes.

### `chat.py` — `ChatSession.send_stream`

```python
def send_stream(self, message: str) -> Iterator[str]:
    return self._provider.send_stream(message)
```

### `app.py` — Streamlit UI

Layout:

```
┌─ Sidebar ────────────────┐ ┌─ Main ───────────────────────┐
│ Provider: [dropdown]     │ │ [user]    Hello              │
│ Model:    [text input]   │ │ [assistant] Hi! ...          │
│ System:   [textarea]     │ │ [user]    ...                │
│ [Apply]                  │ │                              │
│ [Clear conversation]     │ │ [st.chat_input ───────────]  │
└──────────────────────────┘ └──────────────────────────────┘
```

Session state:
- `st.session_state.session` — `ChatSession`, created on first render with defaults from `.env` (same env vars as CLI).
- `st.session_state.provider`, `.model`, `.system_prompt` — current sidebar values.

Behavior:
- **Apply** rebuilds `ChatSession` with current sidebar values. History clears (same semantics as `/model` in CLI). Shows a toast: "Switched to {provider}/{model}".
- **Clear conversation** rebuilds `ChatSession` with the same settings. History clears.
- **Provider dropdown change** does not immediately rebuild — only Apply does. Avoids creating a session for an invalid (provider, model) pair while the user is typing.
- **Model field** has a sensible default per provider (same `DEFAULT_MODELS` dict from `main.py`); when the provider dropdown changes, the model field's placeholder updates.
- **Chat input** calls `session.send_stream(user_input)` and renders via `st.write_stream`. Past messages are rendered from `session.history` on every rerun.
- **Errors** from `send_stream` are caught and rendered with `st.error`; the partial reply (if any) is discarded by the provider's rollback path.
- **Missing API key:** sidebar shows a red error and Apply is disabled if the env var for the selected provider is empty.

Shared constants (`VALID_PROVIDERS`, `DEFAULT_MODELS`, `API_KEY_ENV_VARS`) move from `main.py` into a small `config.py` module so both `main.py` and `app.py` import from one source.

## Data Flow (web mode)

1. User runs `python main.py --mode web`.
2. `main.py` execs `streamlit run app.py`.
3. Streamlit loads `app.py`; first render creates `ChatSession` from `.env` defaults.
4. User types in `st.chat_input` → `app.py` calls `session.send_stream(text)`.
5. `ChatSession.send_stream` returns the provider's iterator.
6. `st.write_stream` consumes the iterator, rendering tokens as they arrive.
7. Provider's `send_stream` finishes, appends full reply to `self.history`.
8. Streamlit reruns; past-messages loop renders the new turn from `session.history`.

## Configuration

No new env vars. The web UI reads the same `.env` as the CLI:

```
GEMINI_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
PROVIDER=gemini            # default selection in sidebar
MODEL=                     # optional default model
SYSTEM_PROMPT=...          # optional default
```

## Testing

- `tests/test_providers_stream.py` — unit tests per provider:
  - `send_stream` yields chunks and assembles them in `history`.
  - API errors during streaming roll back the user message.
  - Mocks the SDK streaming method so no real API calls.
- `tests/test_main_mode.py` — unit test that `--mode web` invokes `os.execvp` with `["streamlit", "run", "app.py"]` (patched).
- `app.py` is not unit tested (Streamlit glue). Manual verification: `python main.py --mode web` opens the browser, can chat, can switch model, can clear, streaming visible.

## Dependencies

Add to `requirements.txt`:

- `streamlit>=1.30.0`

No version bumps to existing deps. All three SDKs already in use support streaming on the versions pinned.

## Out of Scope

- Multi-user sessions / authentication
- Persisting conversation history across reloads
- File upload / image input
- Tool use / function calling in the UI

# File Upload + RAG-Lite — Design Spec

**Date:** 2026-05-28
**Status:** Approved

## Overview

Add file uploads to the Streamlit web UI so the user can attach PDFs, plain text, and Markdown files, then ask questions about them. Uses naive retrieval-augmented generation: documents are chunked, embedded (Gemini `gemini-embedding-001`), and stored in memory. For each user message, the top-k most similar chunks are retrieved and prepended as context. A sidebar checkbox lets the user disable retrieval without removing the indexed documents.

CLI mode is unaffected.

## Components

### `rag.py` (NEW)

Self-contained module, no Streamlit dependency. Public functions:

- `extract_text(filename: str, data: bytes) -> str`
  - `.pdf` → `pypdf.PdfReader`, concatenate page text.
  - `.txt` / `.md` → `data.decode("utf-8", errors="replace")`.
  - Unsupported extension → raises `ValueError`.

- `chunk(text: str, chunk_chars: int = 2000, overlap_chars: int = 200) -> list[str]`
  - Splits on paragraph boundaries first; when a single paragraph exceeds `chunk_chars`, splits by sentence; only if still too large, hard-cuts.
  - `~2000 chars ≈ 500 tokens` heuristic (~4 chars/token).
  - Adjacent chunks overlap by `overlap_chars` to preserve cross-boundary context.
  - Drops empty chunks.

- `embed(texts: list[str], api_key: str) -> list[list[float]]`
  - Batch call to `genai.Client(api_key).models.embed_content(model="gemini-embedding-001", contents=texts)`.
  - Returns one vector per input string.

- `retrieve(query: str, index: list[Chunk], api_key: str, top_k: int = 4) -> list[Chunk]`
  - Embeds the query, computes cosine similarity against all stored embeddings, returns the top-k `Chunk`s by score (descending).
  - Returns `[]` if `index` is empty.

- `Chunk` is a dataclass: `filename: str`, `text: str`, `embedding: list[float]`.

- `build_context_prompt(chunks: list[Chunk], user_message: str) -> str`
  - Formats the augmented message:
    ```
    You have access to the following document excerpts. Use them to answer the user's question when relevant.

    [filename — excerpt 1]
    <chunk text>

    [filename — excerpt 2]
    <chunk text>

    ---

    User question: <user_message>
    ```

### `config.py` (MODIFIED)

Add RAG knobs:

```python
RAG_CHUNK_CHARS = 2000
RAG_OVERLAP_CHARS = 200
RAG_TOP_K = 4
RAG_EMBEDDING_MODEL = "gemini-embedding-001"
SUPPORTED_UPLOAD_EXTENSIONS = (".pdf", ".txt", ".md")
```

### `app.py` (MODIFIED)

**Sidebar (additions, below the existing chat controls):**

- `st.file_uploader("Attach documents", type=["pdf","txt","md"], accept_multiple_files=True)`
- Iterate uploaded files: any whose `(name, size)` pair isn't already in `st.session_state.index_keys` triggers extract → chunk → embed; new chunks are appended to `st.session_state.index`.
- Show indexed docs as a list with a remove button per file. Remove deletes that file's chunks from the index.
- Status caption: `f"Indexed {len(chunks)} chunks across {n_files} files"`.
- `st.checkbox("Use uploaded documents in answers", value=True, key="use_docs")` — default checked. When unchecked: skip retrieval and send the user message as-is. Indexed docs are preserved across toggles.
- Disabled state and message when `GEMINI_API_KEY` is not set ("File uploads require GEMINI_API_KEY in .env, regardless of chat provider").

**Chat flow change:**

- Maintain `st.session_state.display_messages` — a list of `{"role", "content"}` dicts containing the user's *original* messages and the assistant's replies. UI renders this, not `session.history`.
- On chat submit:
  1. Append `{"role": "user", "content": user_input}` to `display_messages`.
  2. If `use_docs` is True AND index is non-empty: `chunks = retrieve(user_input, index, api_key, top_k=4)`; `payload = build_context_prompt(chunks, user_input)`. Else: `payload = user_input`.
  3. Stream `session.send_stream(payload)` via `st.write_stream`; capture full reply.
  4. Append `{"role": "assistant", "content": full_reply}` to `display_messages`.
  5. `st.rerun()`.
- "Clear conversation" button also clears `display_messages` (it already rebuilds the `ChatSession`, which clears provider history).

**Display vs provider history:**

- The UI renders from `display_messages` — clean user input, no chunks visible.
- The provider's internal history contains the augmented payloads. This means provider token usage grows faster than the visible conversation. Acceptable for a personal tool; user can Clear when it gets heavy. Documented in README.

## Data flow

```
upload → extract_text → chunk → embed → index (in session_state)
                                          ↑
user message ───────────────────────────── │
       │                                   │
       ├─→ checkbox unchecked? ─→ send as-is
       │
       └─→ checked & index non-empty
              │
              ├─→ embed query → cosine sim → top-k chunks
              │
              └─→ build_context_prompt → session.send_stream → display
```

## Configuration

No new env vars. Existing `GEMINI_API_KEY` doubles as the embeddings key. File uploads are disabled in the UI when it's missing.

## Dependencies

- `pypdf>=4.0.0` — PDF text extraction.

`google-genai` is already pinned; same SDK is used for embeddings.

## Testing

- `tests/test_rag.py`:
  - `extract_text` for `.txt` and `.md` (no I/O, just bytes).
  - `extract_text` for `.pdf` — generate a small PDF in memory or use a tiny fixture.
  - `extract_text` rejects unsupported extensions.
  - `chunk` splits long text into multiple chunks with overlap; short text becomes one chunk; empty text becomes zero chunks.
  - `retrieve` returns top-k by cosine similarity (mock `embed` to return deterministic vectors).
  - `retrieve` returns `[]` for empty index.
  - `build_context_prompt` includes filename and user message verbatim.

- `app.py` not unit tested.

## Out of scope

- `.doc` / `.docx`
- OCR for image-only PDFs
- Persisting the index across restarts
- Cross-document deduplication
- Showing retrieved chunks in the UI (could add later as a debug expander)

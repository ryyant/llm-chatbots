import os

import streamlit as st
from dotenv import load_dotenv

import rag
from chat import ChatSession
from config import (
    API_KEY_ENV_VARS,
    DEFAULT_MODELS,
    DEFAULT_SYSTEM_PROMPT,
    RAG_CHUNK_CHARS,
    RAG_OVERLAP_CHARS,
    RAG_TOP_K,
    SUPPORTED_UPLOAD_EXTENSIONS,
    VALID_PROVIDERS,
)

load_dotenv()

st.set_page_config(page_title="LLM Chat", page_icon=None, layout="centered")


def _api_key(provider: str) -> str:
    return os.environ.get(API_KEY_ENV_VARS[provider], "")


def _gemini_key() -> str:
    return os.environ.get(API_KEY_ENV_VARS["gemini"], "")


def _build_session(provider: str, model: str, system_prompt: str) -> ChatSession:
    return ChatSession(
        api_key=_api_key(provider),
        model=model,
        system_prompt=system_prompt,
        provider=provider,
    )


def _init_state() -> None:
    if "provider" not in st.session_state:
        env_provider = os.environ.get("PROVIDER", "gemini").lower()
        st.session_state.provider = env_provider if env_provider in VALID_PROVIDERS else "gemini"
    if "model" not in st.session_state:
        st.session_state.model = os.environ.get("MODEL") or DEFAULT_MODELS[st.session_state.provider]
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = os.environ.get("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)
    if "session" not in st.session_state:
        st.session_state.session = None
        if _api_key(st.session_state.provider):
            st.session_state.session = _build_session(
                st.session_state.provider,
                st.session_state.model,
                st.session_state.system_prompt,
            )
    if "display_messages" not in st.session_state:
        st.session_state.display_messages = []
    if "index" not in st.session_state:
        st.session_state.index = []
    if "index_keys" not in st.session_state:
        st.session_state.index_keys = set()
    if "use_docs" not in st.session_state:
        st.session_state.use_docs = True


_init_state()


def _ingest_uploads(uploaded_files) -> None:
    """Extract, chunk, and embed any newly uploaded files. Idempotent on (name, size)."""
    gemini_key = _gemini_key()
    if not gemini_key:
        st.error("File uploads require GEMINI_API_KEY in .env (used for embeddings).")
        return

    new_files = []
    for uf in uploaded_files:
        key = (uf.name, uf.size)
        if key in st.session_state.index_keys:
            continue
        new_files.append((uf, key))

    if not new_files:
        return

    with st.spinner(f"Indexing {len(new_files)} file(s)..."):
        for uf, key in new_files:
            try:
                data = uf.getvalue()
                text = rag.extract_text(uf.name, data)
                pieces = rag.chunk(text, chunk_chars=RAG_CHUNK_CHARS, overlap_chars=RAG_OVERLAP_CHARS)
                if not pieces:
                    st.warning(f"{uf.name}: no extractable text, skipping.")
                    st.session_state.index_keys.add(key)
                    continue
                vectors = rag.embed(pieces, api_key=gemini_key)
                for piece, vec in zip(pieces, vectors):
                    st.session_state.index.append(
                        rag.Chunk(filename=uf.name, text=piece, embedding=vec)
                    )
                st.session_state.index_keys.add(key)
            except Exception as e:
                st.error(f"Failed to index {uf.name}: {e}")


def _remove_file_from_index(filename: str) -> None:
    st.session_state.index = [c for c in st.session_state.index if c.filename != filename]
    st.session_state.index_keys = {k for k in st.session_state.index_keys if k[0] != filename}


with st.sidebar:
    st.header("Settings")

    provider = st.selectbox(
        "Provider",
        options=list(VALID_PROVIDERS),
        index=list(VALID_PROVIDERS).index(st.session_state.provider),
        key="provider_select",
    )

    model = st.text_input(
        "Model",
        value=st.session_state.model
        if provider == st.session_state.provider
        else DEFAULT_MODELS[provider],
        key="model_input",
    )

    system_prompt = st.text_area(
        "System prompt",
        value=st.session_state.system_prompt,
        key="system_prompt_input",
        height=120,
    )

    key_present = bool(_api_key(provider))
    if not key_present:
        st.error(f"Missing {API_KEY_ENV_VARS[provider]} in .env")

    col_apply, col_clear = st.columns(2)
    with col_apply:
        apply_clicked = st.button("Apply", disabled=not key_present, use_container_width=True)
    with col_clear:
        clear_clicked = st.button(
            "Clear",
            disabled=st.session_state.session is None,
            use_container_width=True,
        )

    if apply_clicked:
        try:
            st.session_state.session = _build_session(provider, model, system_prompt)
            st.session_state.provider = provider
            st.session_state.model = model
            st.session_state.system_prompt = system_prompt
            st.session_state.display_messages = []
            st.toast(f"Switched to {provider}/{model}")
        except Exception as e:
            st.error(f"Failed to switch: {e}")

    if clear_clicked and st.session_state.session is not None:
        st.session_state.session = _build_session(
            st.session_state.provider,
            st.session_state.model,
            st.session_state.system_prompt,
        )
        st.session_state.display_messages = []
        st.toast("Conversation cleared")

    if st.session_state.session is not None:
        st.caption(
            f"Active: {st.session_state.session.current_provider}/"
            f"{st.session_state.session.current_model}"
        )

    st.divider()
    st.subheader("Documents")

    gemini_available = bool(_gemini_key())
    if not gemini_available:
        st.info("Set GEMINI_API_KEY in .env to enable file uploads.")

    uploaded = st.file_uploader(
        "Attach documents",
        type=[ext.lstrip(".") for ext in SUPPORTED_UPLOAD_EXTENSIONS],
        accept_multiple_files=True,
        disabled=not gemini_available,
        key="uploader",
    )
    if uploaded:
        _ingest_uploads(uploaded)

    if st.session_state.index:
        filenames = sorted({c.filename for c in st.session_state.index})
        for fname in filenames:
            n_chunks = sum(1 for c in st.session_state.index if c.filename == fname)
            col_name, col_remove = st.columns([4, 1])
            with col_name:
                st.caption(f"{fname} ({n_chunks} chunk{'s' if n_chunks != 1 else ''})")
            with col_remove:
                if st.button("✕", key=f"rm_{fname}", help=f"Remove {fname}"):
                    _remove_file_from_index(fname)
                    st.rerun()
        st.caption(
            f"Indexed {len(st.session_state.index)} chunks across {len(filenames)} file(s)"
        )

    st.checkbox(
        "Use uploaded documents in answers",
        value=st.session_state.use_docs,
        key="use_docs",
        disabled=not st.session_state.index,
        help="When checked, relevant excerpts are retrieved and included with each question.",
    )


st.title("LLM Chat")

session = st.session_state.session

if session is None:
    st.info("Set an API key in .env and click Apply to start chatting.")
else:
    for turn in st.session_state.display_messages:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    user_input = st.chat_input("Message")
    if user_input:
        st.session_state.display_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        payload = user_input
        if st.session_state.use_docs and st.session_state.index:
            try:
                top_chunks = rag.retrieve(
                    user_input,
                    st.session_state.index,
                    api_key=_gemini_key(),
                    top_k=RAG_TOP_K,
                )
                payload = rag.build_context_prompt(top_chunks, user_input)
            except Exception as e:
                st.warning(f"Retrieval failed, sending without context: {e}")

        with st.chat_message("assistant"):
            try:
                reply = st.write_stream(session.send_stream(payload))
                st.session_state.display_messages.append(
                    {"role": "assistant", "content": reply}
                )
            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state.display_messages.pop()
        st.rerun()

import os

import streamlit as st
from dotenv import load_dotenv

from chat import ChatSession
from config import (
    API_KEY_ENV_VARS,
    DEFAULT_MODELS,
    DEFAULT_SYSTEM_PROMPT,
    VALID_PROVIDERS,
)

load_dotenv()

st.set_page_config(page_title="LLM Chat", page_icon=None, layout="centered")


def _api_key(provider: str) -> str:
    return os.environ.get(API_KEY_ENV_VARS[provider], "")


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


_init_state()


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
            st.toast(f"Switched to {provider}/{model}")
        except Exception as e:
            st.error(f"Failed to switch: {e}")

    if clear_clicked and st.session_state.session is not None:
        st.session_state.session = _build_session(
            st.session_state.provider,
            st.session_state.model,
            st.session_state.system_prompt,
        )
        st.toast("Conversation cleared")

    if st.session_state.session is not None:
        st.caption(
            f"Active: {st.session_state.session.current_provider}/"
            f"{st.session_state.session.current_model}"
        )


st.title("LLM Chat")

session = st.session_state.session

if session is None:
    st.info("Set an API key in .env and click Apply to start chatting.")
else:
    for turn in session.history:
        role = "assistant" if turn["role"] == "assistant" else "user"
        with st.chat_message(role):
            st.markdown(turn["content"])

    user_input = st.chat_input("Message")
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            try:
                st.write_stream(session.send_stream(user_input))
            except Exception as e:
                st.error(f"Error: {e}")
        st.rerun()

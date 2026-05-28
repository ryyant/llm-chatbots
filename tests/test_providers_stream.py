from unittest.mock import MagicMock, patch

import pytest


@patch("providers.openai.OpenAI")
def test_openai_stream_yields_chunks_and_builds_history(mock_class):
    mock_client = MagicMock()
    chunks = [
        MagicMock(choices=[MagicMock(delta=MagicMock(content="Hel"))]),
        MagicMock(choices=[MagicMock(delta=MagicMock(content="lo"))]),
        MagicMock(choices=[MagicMock(delta=MagicMock(content="!"))]),
    ]
    mock_client.chat.completions.create.return_value = iter(chunks)
    mock_class.return_value = mock_client
    from providers.openai import OpenAIProvider
    p = OpenAIProvider(api_key="k", model="gpt-4o", system_prompt="Help.")
    out = list(p.send_stream("Hi"))
    assert out == ["Hel", "lo", "!"]
    assert p.history == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    assert mock_client.chat.completions.create.call_args.kwargs["stream"] is True


@patch("providers.openai.OpenAI")
def test_openai_stream_rolls_back_history_on_error(mock_class):
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("boom")
    mock_class.return_value = mock_client
    from providers.openai import OpenAIProvider
    p = OpenAIProvider(api_key="k", model="gpt-4o", system_prompt="Help.")
    with pytest.raises(RuntimeError, match="boom"):
        list(p.send_stream("Hi"))
    assert p.history == []


@patch("providers.openai.OpenAI")
def test_openai_stream_skips_none_deltas(mock_class):
    mock_client = MagicMock()
    chunks = [
        MagicMock(choices=[MagicMock(delta=MagicMock(content=None))]),
        MagicMock(choices=[MagicMock(delta=MagicMock(content="ok"))]),
    ]
    mock_client.chat.completions.create.return_value = iter(chunks)
    mock_class.return_value = mock_client
    from providers.openai import OpenAIProvider
    p = OpenAIProvider(api_key="k", model="gpt-4o", system_prompt="Help.")
    out = list(p.send_stream("Hi"))
    assert out == ["ok"]
    assert p.history[-1] == {"role": "assistant", "content": "ok"}


@patch("providers.anthropic.Anthropic")
def test_anthropic_stream_yields_text_and_builds_history(mock_class):
    mock_client = MagicMock()
    stream_cm = MagicMock()
    stream_cm.__enter__.return_value.text_stream = iter(["He", "llo"])
    stream_cm.__exit__.return_value = None
    mock_client.messages.stream.return_value = stream_cm
    mock_class.return_value = mock_client
    from providers.anthropic import AnthropicProvider
    p = AnthropicProvider(api_key="k", model="claude-opus-4-7", system_prompt="Help.")
    out = list(p.send_stream("Hi"))
    assert out == ["He", "llo"]
    assert p.history == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]


@patch("providers.anthropic.Anthropic")
def test_anthropic_stream_rolls_back_history_on_error(mock_class):
    mock_client = MagicMock()
    mock_client.messages.stream.side_effect = Exception("nope")
    mock_class.return_value = mock_client
    from providers.anthropic import AnthropicProvider
    p = AnthropicProvider(api_key="k", model="claude-opus-4-7", system_prompt="Help.")
    with pytest.raises(RuntimeError, match="nope"):
        list(p.send_stream("Hi"))
    assert p.history == []


@patch("providers.gemini.genai")
def test_gemini_stream_yields_chunks_and_builds_history(mock_genai):
    mock_client = MagicMock()
    mock_chat = MagicMock()
    mock_chat.send_message_stream.return_value = iter([
        MagicMock(text="Hi "),
        MagicMock(text="there"),
    ])
    mock_client.chats.create.return_value = mock_chat
    mock_genai.Client.return_value = mock_client
    from providers.gemini import GeminiProvider
    p = GeminiProvider(api_key="k", model="gemini-2.5-flash", system_prompt="Help.")
    out = list(p.send_stream("Hello"))
    assert out == ["Hi ", "there"]
    assert p.history == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]


@patch("providers.gemini.genai")
def test_gemini_stream_rolls_back_history_on_error(mock_genai):
    mock_client = MagicMock()
    mock_chat = MagicMock()
    mock_chat.send_message_stream.side_effect = Exception("quota")
    mock_client.chats.create.return_value = mock_chat
    mock_genai.Client.return_value = mock_client
    from providers.gemini import GeminiProvider
    p = GeminiProvider(api_key="k", model="gemini-2.5-flash", system_prompt="Help.")
    with pytest.raises(RuntimeError, match="quota"):
        list(p.send_stream("Hi"))
    assert p.history == []

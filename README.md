# llm-chatbots

A command-line AI chat program in Python supporting multiple LLM providers with runtime model switching.

## Supported Providers

| Provider | Models | Free tier |
|----------|--------|-----------|
| Google Gemini | `gemini-2.5-flash`, `gemini-2.0-flash`, ... | [aistudio.google.com](https://aistudio.google.com) |
| OpenAI | `gpt-4o`, `gpt-4o-mini`, ... | [platform.openai.com](https://platform.openai.com) |
| Anthropic | `claude-opus-4-7`, `claude-sonnet-4-6`, ... | [console.anthropic.com](https://console.anthropic.com) |

## Requirements

Python 3.10+

## Setup

1. Create and activate a virtual environment:
   ```bash
   python3.10 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and fill in your keys:
   ```bash
   cp .env.example .env
   ```

   ```
   PROVIDER=gemini              # gemini | openai | anthropic
   GEMINI_API_KEY=your_key
   OPENAI_API_KEY=your_key
   ANTHROPIC_API_KEY=your_key
   MODEL=                       # leave blank to use the default for the provider
   SYSTEM_PROMPT=You are a helpful assistant.
   ```

   You only need the API key for the provider(s) you plan to use.

## Usage

```bash
# activate the venv first if not already active
source venv/bin/activate

python main.py                # CLI mode (default)
python main.py --mode web     # web UI in your browser (Streamlit)
```

### CLI mode

Type your message and press Enter. Use `Ctrl+C` or `Ctrl+D` to quit.

Switch models at runtime:

```
/model                              # show current provider and model
/model openai/gpt-4o                # switch to OpenAI GPT-4o
/model anthropic/claude-opus-4-7    # switch to Anthropic Claude
/model gemini/gemini-2.5-flash      # switch back to Gemini
```

Switching starts a fresh conversation (history is cleared).

### Web mode

`python main.py --mode web` launches a Streamlit app in your browser. The sidebar lets you pick a provider, set the model, edit the system prompt, apply changes, or clear the conversation. Responses stream as they are generated.

## Running Tests

```bash
python -m pytest        # run all tests
python -m pytest tests/test_foo.py::test_bar   # run a single test
```

## Project Structure

```
main.py                  # entry point; --mode cli|web dispatch
app.py                   # Streamlit web UI
chat.py                  # ChatSession: thin wrapper over provider
config.py                # shared constants (providers, defaults, env var names)
providers/
  base.py                # BaseProvider ABC (send + send_stream)
  gemini.py              # GeminiProvider
  openai.py              # OpenAIProvider
  anthropic.py           # AnthropicProvider
  __init__.py            # create_provider() factory
tests/
  test_chat.py
  test_main.py
  test_providers_*.py
```

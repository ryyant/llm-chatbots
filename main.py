import argparse
import os
import sys
from typing import NamedTuple, Optional
from dotenv import load_dotenv

from chat import ChatSession
from config import (
    API_KEY_ENV_VARS,
    DEFAULT_MODELS,
    DEFAULT_SYSTEM_PROMPT,
    VALID_PROVIDERS,
)

_VALID_PROVIDERS_SET = set(VALID_PROVIDERS)


class HandleResult(NamedTuple):
    output: str
    new_session: Optional[ChatSession]


def handle_input(
    user_input: str,
    session: ChatSession,
    system_prompt: str,
    api_keys: dict[str, str],
) -> HandleResult:
    if user_input.startswith("/model"):
        args = user_input[len("/model"):].strip()

        if not args:
            return HandleResult(
                output=f"Current model: {session.current_provider}/{session.current_model}",
                new_session=None,
            )

        if "/" not in args:
            return HandleResult(
                output="Usage: /model <provider>/<model>  e.g. /model openai/gpt-4o",
                new_session=None,
            )

        provider, model = args.split("/", 1)
        provider = provider.strip().lower()
        model = model.strip()

        if provider not in _VALID_PROVIDERS_SET:
            return HandleResult(
                output=f"Unknown provider: {provider!r}. Choose from: {', '.join(sorted(_VALID_PROVIDERS_SET))}",
                new_session=None,
            )

        api_key = api_keys.get(provider, "")
        if not api_key:
            return HandleResult(
                output=f"No API key for {provider!r}. Set {API_KEY_ENV_VARS[provider]} in your .env file.",
                new_session=None,
            )

        try:
            new_session = ChatSession(
                api_key=api_key,
                model=model,
                system_prompt=system_prompt,
                provider=provider,
            )
        except Exception as e:
            return HandleResult(
                output=f"Failed to switch to {provider}/{model}: {e}",
                new_session=None,
            )
        return HandleResult(
            output=f"Switched to {provider}/{model}. Conversation history cleared.",
            new_session=new_session,
        )

    try:
        reply = session.send(user_input)
        return HandleResult(output=reply, new_session=None)
    except Exception as e:
        return HandleResult(output=f"Error: {e}", new_session=None)


def run_cli() -> None:
    api_keys = {name: os.environ.get(env, "") for name, env in API_KEY_ENV_VARS.items()}

    provider = os.environ.get("PROVIDER", "gemini").lower()
    if provider not in _VALID_PROVIDERS_SET:
        print(f"Error: PROVIDER={provider!r} is not supported. Choose from: {', '.join(sorted(_VALID_PROVIDERS_SET))}")
        return

    api_key = api_keys.get(provider, "")
    if not api_key:
        print(f"Error: {API_KEY_ENV_VARS[provider]} is not set. Add it to your .env file.")
        return

    model = os.environ.get("MODEL") or DEFAULT_MODELS[provider]
    system_prompt = os.environ.get("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)

    session = ChatSession(
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        provider=provider,
    )

    print(f"Chat started ({provider}/{model}).")
    print(f"  /model                          show current model")
    print(f"  /model <provider>/<model>       switch model (providers: gemini, openai, anthropic)")
    print(f"  Ctrl+C                          quit\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        try:
            result = handle_input(user_input, session, system_prompt, api_keys)
        except KeyboardInterrupt:
            print("\n^C (cancelled)\n")
            continue

        if result.new_session is not None:
            session = result.new_session

        if user_input.startswith("/"):
            print(f"{result.output}\n")
        else:
            print(f"AI: {result.output}\n")


def run_web() -> None:
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    try:
        os.execvp("streamlit", ["streamlit", "run", app_path])
    except FileNotFoundError:
        print("Error: 'streamlit' is not installed. Run: pip install -r requirements.txt")
        sys.exit(1)


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="LLM chat in CLI or web mode.")
    parser.add_argument(
        "--mode",
        choices=("cli", "web"),
        default="cli",
        help="Interface mode: 'cli' (default) for terminal REPL, 'web' for Streamlit UI.",
    )
    args = parser.parse_args()

    if args.mode == "web":
        run_web()
    else:
        run_cli()


if __name__ == "__main__":
    main()

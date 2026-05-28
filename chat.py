from typing import Iterator

from providers import create_provider
from providers.base import BaseProvider


class ChatSession:
    def __init__(
        self,
        api_key: str,
        model: str,
        system_prompt: str,
        provider: str = "gemini",
    ):
        self.current_provider = provider
        self.current_model = model
        self._provider: BaseProvider = create_provider(
            provider=provider,
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
        )

    @property
    def history(self) -> list[dict]:
        return list(self._provider.history)

    def send(self, message: str) -> str:
        return self._provider.send(message)

    def send_stream(self, message: str) -> Iterator[str]:
        return self._provider.send_stream(message)

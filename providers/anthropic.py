from typing import Iterator

from anthropic import Anthropic

from providers.base import BaseProvider


class AnthropicProvider(BaseProvider):
    def __init__(self, api_key: str, model: str, system_prompt: str):
        self._client = Anthropic(api_key=api_key)
        self._model = model
        self._system_prompt = system_prompt
        self.history: list[dict] = []

    def _messages_payload(self) -> list[dict]:
        return [{"role": h["role"], "content": h["content"]} for h in self.history]

    def send(self, message: str) -> str:
        self.history.append({"role": "user", "content": message})
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=8192,
                system=self._system_prompt,
                messages=self._messages_payload(),
            )
        except Exception as e:
            self.history.pop()
            raise RuntimeError(str(e)) from e
        reply = "".join(block.text for block in response.content if block.type == "text")
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def send_stream(self, message: str) -> Iterator[str]:
        self.history.append({"role": "user", "content": message})
        chunks: list[str] = []
        try:
            with self._client.messages.stream(
                model=self._model,
                max_tokens=8192,
                system=self._system_prompt,
                messages=self._messages_payload(),
            ) as stream:
                for text in stream.text_stream:
                    if text:
                        chunks.append(text)
                        yield text
        except Exception as e:
            self.history.pop()
            raise RuntimeError(str(e)) from e
        self.history.append({"role": "assistant", "content": "".join(chunks)})

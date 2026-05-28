from typing import Iterator

from openai import OpenAI

from providers.base import BaseProvider


class OpenAIProvider(BaseProvider):
    def __init__(self, api_key: str, model: str, system_prompt: str):
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._system_prompt = system_prompt
        self.history: list[dict] = []

    def _build_messages(self) -> list[dict]:
        return [{"role": "system", "content": self._system_prompt}] + [
            {"role": h["role"], "content": h["content"]} for h in self.history
        ]

    def send(self, message: str) -> str:
        self.history.append({"role": "user", "content": message})
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=self._build_messages(),
            )
        except Exception as e:
            self.history.pop()
            raise RuntimeError(str(e)) from e
        reply = response.choices[0].message.content or ""
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def send_stream(self, message: str) -> Iterator[str]:
        self.history.append({"role": "user", "content": message})
        chunks: list[str] = []
        try:
            stream = self._client.chat.completions.create(
                model=self._model,
                messages=self._build_messages(),
                stream=True,
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    chunks.append(delta)
                    yield delta
        except Exception as e:
            self.history.pop()
            raise RuntimeError(str(e)) from e
        self.history.append({"role": "assistant", "content": "".join(chunks)})

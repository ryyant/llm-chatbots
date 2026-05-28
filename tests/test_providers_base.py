from typing import Iterator

import pytest
from providers.base import BaseProvider


def test_base_provider_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseProvider()


def test_base_provider_subclass_must_implement_send():
    class Incomplete(BaseProvider):
        pass

    with pytest.raises(TypeError):
        Incomplete()


def test_base_provider_subclass_missing_send_stream_is_invalid():
    class OnlySend(BaseProvider):
        def __init__(self):
            self.history = []

        def send(self, message: str) -> str:
            return "ok"

    with pytest.raises(TypeError):
        OnlySend()


def test_base_provider_subclass_with_send_and_send_stream_is_valid():
    class Complete(BaseProvider):
        def __init__(self):
            self.history = []

        def send(self, message: str) -> str:
            self.history.append({"role": "user", "content": message})
            self.history.append({"role": "assistant", "content": "ok"})
            return "ok"

        def send_stream(self, message: str) -> Iterator[str]:
            self.history.append({"role": "user", "content": message})
            yield "o"
            yield "k"
            self.history.append({"role": "assistant", "content": "ok"})

    obj = Complete()
    assert obj.send("hi") == "ok"
    assert "".join(obj.send_stream("hi")) == "ok"

from abc import ABC, abstractmethod
from typing import Iterator


class BaseProvider(ABC):
    history: list[dict]

    @abstractmethod
    def send(self, message: str) -> str:
        """Send a message and return the assistant's reply."""

    @abstractmethod
    def send_stream(self, message: str) -> Iterator[str]:
        """Send a message and yield response chunks as they arrive."""

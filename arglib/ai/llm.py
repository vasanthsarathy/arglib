"""Pluggable model hooks for mining and reconciliation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class LLMClient(Protocol):
    def complete(self, prompt: str, *, metadata: dict[str, Any] | None = None) -> str:
        """Return a completion string for a prompt."""


@dataclass
class PromptTemplate:
    system: str
    user: str

    def render(self, *, input: str, context: dict[str, Any] | None = None) -> str:
        ctx = context or {}
        return f"{self.system}\n\n{self.user.format(input=input, **ctx)}"


@dataclass
class LLMHook:
    client: LLMClient
    template: PromptTemplate

    def run(self, *, input: str, context: dict[str, Any] | None = None) -> str:
        prompt = self.template.render(input=input, context=context)
        return self.client.complete(prompt, metadata=context)


@dataclass
class NoOpLLMClient:
    """Placeholder client for testing or offline mode."""

    response: str = ""

    def complete(self, prompt: str, *, metadata: dict[str, Any] | None = None) -> str:
        return self.response


__all__ = ["LLMClient", "LLMHook", "NoOpLLMClient", "PromptTemplate"]

"""Pluggable model hooks for mining and reconciliation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class LLMClient(Protocol):
    def complete(self, prompt: str, *, metadata: dict[str, Any] | None = None) -> str:
        """Return a completion string for a prompt."""


class AsyncLLMClient(Protocol):
    async def complete(
        self, prompt: str, *, metadata: dict[str, Any] | None = None
    ) -> str:
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
class AsyncLLMHook:
    client: AsyncLLMClient
    template: PromptTemplate

    async def run(self, *, input: str, context: dict[str, Any] | None = None) -> str:
        prompt = self.template.render(input=input, context=context)
        return await self.client.complete(prompt, metadata=context)


@dataclass
class NoOpLLMClient:
    """Placeholder client for testing or offline mode."""

    response: str = ""

    def complete(self, prompt: str, *, metadata: dict[str, Any] | None = None) -> str:
        return self.response


@dataclass
class AsyncNoOpLLMClient:
    """Async placeholder client for testing or offline mode."""

    response: str = ""

    async def complete(
        self, prompt: str, *, metadata: dict[str, Any] | None = None
    ) -> str:
        return self.response


@dataclass
class OllamaClient:
    model: str
    base_url: str = "http://localhost:11434"
    timeout: float = 30.0
    options: dict[str, Any] | None = None

    def complete(self, prompt: str, *, metadata: dict[str, Any] | None = None) -> str:
        payload = self._build_payload(prompt, metadata=metadata)
        request = Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except (HTTPError, URLError) as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            return body
        return str(parsed.get("response", ""))

    def _build_payload(
        self, prompt: str, *, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if self.options:
            payload["options"] = self.options
        if metadata:
            payload["metadata"] = metadata
        return payload


__all__ = [
    "AsyncLLMClient",
    "AsyncLLMHook",
    "AsyncNoOpLLMClient",
    "LLMClient",
    "LLMHook",
    "NoOpLLMClient",
    "OllamaClient",
    "PromptTemplate",
]

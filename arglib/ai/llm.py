"""Pluggable model hooks for mining and reconciliation."""

from __future__ import annotations

import json
import os
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
        metadata = None
        if context and "llm_metadata" in context:
            metadata = context.get("llm_metadata")
        return self.client.complete(prompt, metadata=metadata)


@dataclass
class AsyncLLMHook:
    client: AsyncLLMClient
    template: PromptTemplate

    async def run(self, *, input: str, context: dict[str, Any] | None = None) -> str:
        prompt = self.template.render(input=input, context=context)
        metadata = None
        if context and "llm_metadata" in context:
            metadata = context.get("llm_metadata")
        return await self.client.complete(prompt, metadata=metadata)


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
class OpenAIClient:
    model: str
    api_key: str | None = None
    base_url: str = "https://api.openai.com/v1"
    timeout: float = 30.0
    options: dict[str, Any] | None = None

    def complete(self, prompt: str, *, metadata: dict[str, Any] | None = None) -> str:
        payload = self._build_payload(prompt, metadata=metadata)
        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except (HTTPError, URLError) as exc:
            raise RuntimeError(f"OpenAI request failed: {exc}") from exc
        return _parse_openai_response(body)

    def _headers(self) -> dict[str, str]:
        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _build_payload(
        self, prompt: str, *, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.options:
            payload.update(self.options)
        if metadata:
            payload["metadata"] = metadata
        return payload


@dataclass
class AnthropicClient:
    model: str
    api_key: str | None = None
    base_url: str = "https://api.anthropic.com/v1"
    timeout: float = 30.0
    max_tokens: int = 2048
    options: dict[str, Any] | None = None

    def complete(self, prompt: str, *, metadata: dict[str, Any] | None = None) -> str:
        payload = self._build_payload(prompt, metadata=metadata)
        request = Request(
            f"{self.base_url}/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except (HTTPError, URLError) as exc:
            raise RuntimeError(f"Anthropic request failed: {exc}") from exc
        return _parse_anthropic_response(body)

    def _headers(self) -> dict[str, str]:
        api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        return {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    def _build_payload(
        self, prompt: str, *, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.options:
            payload.update(self.options)
        if metadata:
            payload["metadata"] = metadata
        return payload


def _parse_openai_response(body: str) -> str:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return body
    choices = payload.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    return str(message.get("content", ""))


def _parse_anthropic_response(body: str) -> str:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return body
    content = payload.get("content", [])
    if not content:
        return ""
    first = content[0]
    if isinstance(first, dict):
        return str(first.get("text", ""))
    return str(first)


@dataclass
class OllamaClient:
    model: str
    base_url: str = "http://localhost:11434"
    timeout: float = 30.0
    options: dict[str, Any] | None = None

    def complete(self, prompt: str, *, metadata: dict[str, Any] | None = None) -> str:
        try:
            return self._complete_with_sdk(prompt, metadata=metadata)
        except Exception:
            return self._complete_with_http(prompt, metadata=metadata)

    def _complete_with_sdk(
        self, prompt: str, *, metadata: dict[str, Any] | None = None
    ) -> str:
        try:
            from ollama import Client
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("ollama-python not available") from exc

        client = Client(host=self.base_url)
        response = client.generate(
            model=self.model,
            prompt=prompt,
            options=self.options,
            stream=False,
        )
        if isinstance(response, dict):
            return str(response.get("response", ""))
        return str(response)

    def _complete_with_http(
        self, prompt: str, *, metadata: dict[str, Any] | None = None
    ) -> str:
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
        payload: dict[str, Any] = {
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
    "AnthropicClient",
    "LLMClient",
    "LLMHook",
    "NoOpLLMClient",
    "OpenAIClient",
    "OllamaClient",
    "PromptTemplate",
]

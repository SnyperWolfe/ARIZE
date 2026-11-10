"""Minimal client for any OpenAI-compatible chat server (Ollama, LM Studio, llama.cpp, ...)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import httpx


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON text, exactly as the model produced it


@dataclass
class Reply:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)

    def as_message(self) -> dict:
        msg: dict = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": c.arguments},
                }
                for c in self.tool_calls
            ]
        return msg


class LLMError(RuntimeError):
    pass


class ChatModel(Protocol):
    def chat(self, messages: list[dict], tools: list[dict]) -> Reply: ...


class OpenAICompatibleModel:
    def __init__(self, base_url: str, model: str, api_key: str = "", timeout: float = 300.0):
        self.model = model
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout)

    def chat(self, messages: list[dict], tools: list[dict]) -> Reply:
        body: dict = {"model": self.model, "messages": messages}
        if tools:
            body["tools"] = tools
        try:
            resp = self._http.post("/chat/completions", json=body)
        except httpx.HTTPError as exc:
            raise LLMError(
                f"Can't reach the model server at {self._http.base_url} ({exc}). "
                "Is Ollama running? Try: ollama serve"
            ) from exc
        if resp.status_code >= 400:
            raise LLMError(f"Model server error {resp.status_code}: {resp.text[:500]}")
        msg = resp.json()["choices"][0]["message"]
        calls = [
            ToolCall(
                id=c.get("id") or f"call_{i}",
                name=c["function"]["name"],
                arguments=_args_text(c["function"].get("arguments")),
            )
            for i, c in enumerate(msg.get("tool_calls") or [])
        ]
        return Reply(content=msg.get("content") or "", tool_calls=calls)


def _args_text(args) -> str:
    if args is None:
        return "{}"
    if isinstance(args, str):
        return args
    import json

    return json.dumps(args)

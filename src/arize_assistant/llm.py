"""Minimal client for any OpenAI-compatible chat server (Ollama, LM Studio, llama.cpp, ...)."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import httpx

# Receives each new piece of the answer as the model writes it.
TokenHook = Callable[[str], None]


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
    def chat(
        self, messages: list[dict], tools: list[dict], on_token: TokenHook | None = None
    ) -> Reply: ...


class OpenAICompatibleModel:
    def __init__(self, base_url: str, model: str, api_key: str = "", timeout: float = 300.0):
        self.model = model
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout)

    def chat(
        self, messages: list[dict], tools: list[dict], on_token: TokenHook | None = None
    ) -> Reply:
        body: dict = {"model": self.model, "messages": messages}
        if tools:
            body["tools"] = tools
        if on_token is None:
            resp = self._post(body)
            return _parse_message(resp.json()["choices"][0]["message"])
        body["stream"] = True
        try:
            with self._http.stream("POST", "/chat/completions", json=body) as resp:
                if resp.status_code >= 400:
                    resp.read()
                    raise LLMError(f"Model server error {resp.status_code}: {resp.text[:500]}")
                return _read_stream(resp.iter_lines(), on_token)
        except httpx.HTTPError as exc:
            raise self._unreachable(exc) from exc

    def _post(self, body: dict) -> httpx.Response:
        try:
            resp = self._http.post("/chat/completions", json=body)
        except httpx.HTTPError as exc:
            raise self._unreachable(exc) from exc
        if resp.status_code >= 400:
            raise LLMError(f"Model server error {resp.status_code}: {resp.text[:500]}")
        return resp

    def _unreachable(self, exc: Exception) -> LLMError:
        return LLMError(
            f"Can't reach the model server at {self._http.base_url} ({exc}). "
            "Is Ollama running? Open the Ollama app, or run: ollama serve"
        )


def _parse_message(msg: dict) -> Reply:
    calls = [
        ToolCall(
            id=c.get("id") or f"call_{i}",
            name=c["function"]["name"],
            arguments=_args_text(c["function"].get("arguments")),
        )
        for i, c in enumerate(msg.get("tool_calls") or [])
    ]
    return Reply(content=msg.get("content") or "", tool_calls=calls)


def _read_stream(lines, on_token: TokenHook) -> Reply:
    """Assemble a streamed (server-sent events) reply, passing text pieces to ``on_token``."""
    content: list[str] = []
    calls: dict[int, dict] = {}
    for line in lines:
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        if not chunk.get("choices"):
            continue
        delta = chunk["choices"][0].get("delta") or {}
        if delta.get("content"):
            content.append(delta["content"])
            on_token(delta["content"])
        for i, tc in enumerate(delta.get("tool_calls") or []):
            slot = calls.setdefault(tc.get("index", i), {"id": "", "name": "", "arguments": ""})
            slot["id"] = tc.get("id") or slot["id"]
            fn = tc.get("function") or {}
            slot["name"] += fn.get("name") or ""
            args = fn.get("arguments")
            slot["arguments"] += args if isinstance(args, str) else _args_text(args)
    tool_calls = [
        ToolCall(id=c["id"] or f"call_{i}", name=c["name"], arguments=c["arguments"] or "{}")
        for i, c in sorted(calls.items())
    ]
    return Reply(content="".join(content), tool_calls=tool_calls)


def _args_text(args) -> str:
    if args is None:
        return "{}"
    if isinstance(args, str):
        return args
    return json.dumps(args)

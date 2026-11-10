"""Minimal client for any OpenAI-compatible chat server (Ollama, LM Studio, llama.cpp, ...)."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import httpx

log = logging.getLogger("arize")

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
    # This API has no "don't think" switch, so the agent adds qwen3's "/no_think" hint.
    soft_think_switch = True

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

    def look(self, png: bytes, question: str, model: str = "") -> str:
        """Ask a vision-capable model a question about an image."""
        import base64

        image = "data:image/png;base64," + base64.b64encode(png).decode()
        body = {
            "model": model or self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": image}},
                    ],
                }
            ],
        }
        msg = self._post(body).json()["choices"][0]["message"]
        return msg.get("content") or ""

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


class OllamaModel:
    """Talks to Ollama's own API, which (unlike its OpenAI-style one) lets us set how much
    the model can remember (``num_ctx``) and switch thinking on or off properly.

    Ollama's default memory is small (4096 tokens). When a conversation outgrows it,
    Ollama silently drops the beginning, including ARIZE's instructions, and the model
    tends to answer with nothing at all.
    """

    soft_think_switch = False

    def __init__(
        self,
        base_url: str,
        model: str,
        context_length: int = 16384,
        think: bool = False,
        timeout: float = 300.0,
    ):
        self.model = model
        self.context_length = context_length
        self.think: bool | None = think
        root = base_url.rstrip("/").removesuffix("/v1")
        self._http = httpx.Client(base_url=root, timeout=timeout)
        self.last_prompt_tokens = 0

    def _body(self, messages: list[dict], tools: list[dict], stream: bool, model: str = "") -> dict:
        body: dict = {
            "model": model or self.model,
            "messages": [_to_ollama(m) for m in messages],
            "stream": stream,
            "options": {"num_ctx": self.context_length},
        }
        if tools:
            body["tools"] = tools
        if self.think is not None:
            body["think"] = self.think
        return body

    def chat(
        self, messages: list[dict], tools: list[dict], on_token: TokenHook | None = None
    ) -> Reply:
        body = self._body(messages, tools, stream=on_token is not None)
        try:
            if on_token is None:
                return self._finish(self._post(body).json())
            with self._http.stream("POST", "/api/chat", json=body) as resp:
                if resp.status_code >= 400:
                    resp.read()
                    if self._drop_think(resp):
                        return self.chat(messages, tools, on_token)
                    raise LLMError(_server_error(resp))
                content: list[str] = []
                calls: list[dict] = []
                last: dict = {}
                for line in resp.iter_lines():
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    if chunk.get("error"):
                        raise LLMError(f"Ollama error: {chunk['error']}")
                    msg = chunk.get("message") or {}
                    if msg.get("content"):
                        content.append(msg["content"])
                        on_token(msg["content"])
                    calls += msg.get("tool_calls") or []
                    if chunk.get("done"):
                        last = chunk
                last["message"] = {"content": "".join(content), "tool_calls": calls}
                return self._finish(last)
        except httpx.HTTPError as exc:
            raise LLMError(
                f"Can't reach Ollama at {self._http.base_url} ({exc}). "
                "Open the Ollama app, or run: ollama serve"
            ) from exc

    def look(self, png: bytes, question: str, model: str = "") -> str:
        import base64

        msg = {"role": "user", "content": question, "images": [base64.b64encode(png).decode()]}
        body = self._body([], [], stream=False, model=model)
        body["messages"] = [msg]
        return (self._post(body).json().get("message") or {}).get("content") or ""

    def _post(self, body: dict) -> httpx.Response:
        try:
            resp = self._http.post("/api/chat", json=body)
        except httpx.HTTPError as exc:
            raise LLMError(
                f"Can't reach Ollama at {self._http.base_url} ({exc}). "
                "Open the Ollama app, or run: ollama serve"
            ) from exc
        if resp.status_code >= 400:
            if self._drop_think(resp):
                body.pop("think", None)
                return self._post(body)
            raise LLMError(_server_error(resp))
        return resp

    def _drop_think(self, resp: httpx.Response) -> bool:
        """Models without a thinking mode reject the think switch: stop sending it."""
        if self.think is not None and "think" in resp.text.lower():
            self.think = None
            return True
        return False

    def _finish(self, data: dict) -> Reply:
        self.last_prompt_tokens = int(data.get("prompt_eval_count") or 0)
        if self.last_prompt_tokens >= self.context_length * 0.9:
            log.warning(
                "Conversation is %s tokens, close to context_length %s; older parts may be "
                "dropped. Start a new chat or raise context_length.",
                self.last_prompt_tokens,
                self.context_length,
            )
        msg = data.get("message") or {}
        calls = [
            ToolCall(
                id=c.get("id") or f"call_{i}",
                name=(c.get("function") or {}).get("name", ""),
                arguments=_args_text((c.get("function") or {}).get("arguments")),
            )
            for i, c in enumerate(msg.get("tool_calls") or [])
        ]
        return Reply(content=msg.get("content") or "", tool_calls=calls)


def _to_ollama(m: dict) -> dict:
    """Convert an OpenAI-style message to Ollama's format."""
    out: dict = {"role": m["role"], "content": m.get("content") or ""}
    if m.get("tool_calls"):
        out["tool_calls"] = [
            {
                "function": {
                    "name": c["function"]["name"],
                    "arguments": _args_obj(c["function"].get("arguments")),
                }
            }
            for c in m["tool_calls"]
        ]
    if m["role"] == "tool" and m.get("name"):
        out["tool_name"] = m["name"]
    return out


def _args_obj(args):
    if isinstance(args, str):
        try:
            value = json.loads(args or "{}")
            return value if isinstance(value, dict) else {}
        except ValueError:
            return {}
    return args or {}


def _server_error(resp: httpx.Response) -> str:
    text = resp.text[:500]
    if resp.status_code == 404 and "not found" in text.lower():
        return f"{text}\nThe model isn't downloaded yet. Run: ollama pull <model name>"
    return f"Model server error {resp.status_code}: {text}"


def is_ollama(base_url: str) -> bool:
    return ":11434" in base_url


def make_model(cfg) -> ChatModel:
    """The right client for the settings: Ollama's own API when talking to Ollama."""
    provider = cfg.provider.lower()
    if provider == "ollama" or (provider == "auto" and is_ollama(cfg.base_url)):
        return OllamaModel(cfg.base_url, cfg.model, cfg.context_length, cfg.think, cfg.timeout)
    return OpenAICompatibleModel(cfg.base_url, cfg.model, cfg.api_key, cfg.timeout)

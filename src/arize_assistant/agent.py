"""The agent loop: ask the model, run the tools it asks for, repeat until it answers."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import datetime

from .llm import ChatModel, TokenHook
from .memory import MemoryStore
from .tools import ToolRegistry

SYSTEM_PROMPT = """\
You are ARIZE, a private AI assistant running on the user's own computer.
Be direct, warm and concise. Today is {today}. The user's workspace folder is {workspace}.

You can use tools to search the web, read web pages, read and write files, run
commands, and manage your long-term memory. Use a tool whenever it gives a better
answer than guessing; never invent search results, file contents or command output.

Memory: when the user tells you something worth knowing later (their name, preferences,
people, projects, plans), save it with `remember` without being asked. If a saved memory
turns out to be wrong or the user asks you to forget it, use `forget`.
{memories}"""

# Qwen3 models write a long hidden "thinking" section unless told not to.
NO_THINK = "\n/no_think"

_THINK = re.compile(r"<think>.*?(</think>\s*|$)", re.S)

# Called with ("tool", name, arguments) before a tool runs and ("result", name, output) after.
EventHook = Callable[[str, str, str], None]


def strip_thinking(text: str) -> str:
    return _THINK.sub("", text).strip()


class ThinkFilter:
    """Passes streamed text through, minus any <think>...</think> section."""

    def __init__(self, out: TokenHook):
        self.out = out
        self.buf = ""
        self.inside = False
        self.started = False

    def __call__(self, piece: str) -> None:
        self.buf += piece
        while self.buf:
            if self.inside:
                end = self.buf.find("</think>")
                if end < 0:
                    self.buf = self.buf[-8:]
                    return
                self.buf = self.buf[end + 8 :]
                self.inside = False
                continue
            start = self.buf.find("<think>")
            if start >= 0:
                self._emit(self.buf[:start])
                self.buf = self.buf[start + 7 :]
                self.inside = True
                continue
            # Keep a possible partial "<think" at the end until more text arrives.
            longest = min(6, len(self.buf))
            keep = next(
                (n for n in range(longest, 0, -1) if "<think>".startswith(self.buf[-n:])), 0
            )
            self._emit(self.buf[: len(self.buf) - keep])
            self.buf = self.buf[len(self.buf) - keep :]
            return

    def _emit(self, text: str) -> None:
        if not self.started:
            text = text.lstrip()
        if text:
            self.started = True
            self.out(text)


class Agent:
    def __init__(
        self,
        model: ChatModel,
        tools: ToolRegistry,
        memory: MemoryStore,
        max_steps: int = 8,
        on_event: EventHook | None = None,
        think: bool = False,
    ):
        self.model = model
        self.tools = tools
        self.memory = memory
        self.max_steps = max_steps
        self.on_event = on_event or (lambda *_: None)
        self.think = think
        self.history: list[dict] = []

    def system_prompt(self, message: str) -> str:
        found = self.memory.context_for(message)
        if found:
            lines = "\n".join(f"- #{m.id}: {m.text}" for m in found)
            memories = f"\nThings you remember about the user:\n{lines}\n"
        else:
            memories = "\nYou don't have any saved memories yet.\n"
        prompt = SYSTEM_PROMPT.format(
            today=datetime.now().strftime("%A, %B %d, %Y"),
            workspace=self.tools.ctx.workspace,
            memories=memories,
        )
        return prompt if self.think else prompt + NO_THINK

    def reset(self) -> None:
        self.history.clear()

    def ask(self, message: str, on_token: TokenHook | None = None) -> str:
        """Answer ``message``. With ``on_token``, the answer is also streamed piece by piece."""
        self.history.append({"role": "user", "content": message})
        system = {"role": "system", "content": self.system_prompt(message)}
        schemas = self.tools.schemas()
        for _ in range(self.max_steps):
            reply = self._chat([system, *self.history], schemas, on_token)
            self.history.append(reply.as_message())
            if not reply.tool_calls:
                return strip_thinking(reply.content)
            for call in reply.tool_calls:
                self.on_event("tool", call.name, call.arguments)
                output = self.tools.call(call.name, call.arguments)
                self.on_event("result", call.name, output)
                self.history.append(
                    {"role": "tool", "tool_call_id": call.id, "name": call.name, "content": output}
                )
        # Out of steps: ask for a final answer without tools.
        self.history.append(
            {
                "role": "user",
                "content": "You've used all your tool steps. Answer now with what you have.",
            }
        )
        reply = self._chat([system, *self.history], [], on_token)
        self.history.append(reply.as_message())
        return strip_thinking(reply.content)

    def _chat(self, messages, tools, on_token):
        if on_token is None:
            return self.model.chat(messages, tools)
        return self.model.chat(messages, tools, ThinkFilter(on_token))

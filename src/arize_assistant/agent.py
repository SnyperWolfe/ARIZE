"""The agent loop: ask the model, run the tools it asks for, repeat until it answers."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import datetime

from .llm import ChatModel
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

_THINK = re.compile(r"<think>.*?</think>\s*", re.S)

# Called with ("tool", name, arguments) before a tool runs and ("result", name, output) after.
EventHook = Callable[[str, str, str], None]


class Agent:
    def __init__(
        self,
        model: ChatModel,
        tools: ToolRegistry,
        memory: MemoryStore,
        max_steps: int = 8,
        on_event: EventHook | None = None,
    ):
        self.model = model
        self.tools = tools
        self.memory = memory
        self.max_steps = max_steps
        self.on_event = on_event or (lambda *_: None)
        self.history: list[dict] = []

    def system_prompt(self, message: str) -> str:
        found = self.memory.context_for(message)
        if found:
            lines = "\n".join(f"- #{m.id}: {m.text}" for m in found)
            memories = f"\nThings you remember about the user:\n{lines}\n"
        else:
            memories = "\nYou don't have any saved memories yet.\n"
        return SYSTEM_PROMPT.format(
            today=datetime.now().strftime("%A, %B %d, %Y"),
            workspace=self.tools.ctx.workspace,
            memories=memories,
        )

    def reset(self) -> None:
        self.history.clear()

    def ask(self, message: str) -> str:
        self.history.append({"role": "user", "content": message})
        system = {"role": "system", "content": self.system_prompt(message)}
        schemas = self.tools.schemas()
        for _ in range(self.max_steps):
            reply = self.model.chat([system, *self.history], schemas)
            self.history.append(reply.as_message())
            if not reply.tool_calls:
                return _THINK.sub("", reply.content).strip()
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
        reply = self.model.chat([system, *self.history], [])
        self.history.append(reply.as_message())
        return _THINK.sub("", reply.content).strip()

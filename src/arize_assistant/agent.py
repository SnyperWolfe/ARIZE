"""The agent loop: ask the model, run the tools it asks for, repeat until it answers."""

from __future__ import annotations

import logging
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

Godot: the user makes games in Godot 4 with GDScript. Always write Godot 4 code, never
Godot 3: `extends CharacterBody2D`, `@export var speed := 200.0`, `@onready var sprite =
$Sprite2D`, set `velocity` then call `move_and_slide()` with no arguments,
`Input.get_vector("ui_left", "ui_right", "ui_up", "ui_down")`, `body_entered.connect(_on_hit)`,
`await get_tree().create_timer(1.0).timeout`, `instantiate()` (not instance()).
How to work: look first with `godot_project`; write scripts and scenes with `write_file`;
then ALWAYS check with `godot_run` and fix every error it reports before telling the user
it works. Keep .tscn files simple, or build nodes in code inside `_ready()`. A minimal scene:
[gd_scene load_steps=2 format=3]

[ext_resource type="Script" path="res://main.gd" id="1"]

[node name="Main" type="Node2D"]
script = ExtResource("1")

Screen control: use `look_at_screen` to see, then act with `click`, `press_keys`,
`hold_keys`, `type_text` using the (x, y) 0-1000 positions it gives you. Use
`focus_window` to bring a program forward first. Work in small steps and look again
after each action to check it worked. If the user takes back control, stop.
{memories}"""

EMPTY_NOTE = (
    "Your last reply was empty. Reply to the user now in plain words: what you did, "
    "what you found, or what you need from them."
)
STEPS_NOTE = (
    "You've used all {n} tool steps for this message, so stop using tools. Tell the user "
    "in plain words what you've done so far and what's left, and that they can say "
    "'continue' to keep going."
)
EMPTY_ANSWER = (
    "I didn't get an answer back from the model. This usually means the conversation got "
    "too long for it to remember. Click **New chat** and ask again, or raise "
    "`context_length` in the settings file."
)

# Qwen3 models write a long hidden "thinking" section unless told not to.
NO_THINK = "\n/no_think"

log = logging.getLogger("arize")

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
        # Ollama's own API switches thinking off directly; other servers need the hint.
        if self.think or not getattr(self.model, "soft_think_switch", True):
            return prompt
        return prompt + NO_THINK

    def reset(self) -> None:
        self.history.clear()

    def ask(self, message: str, on_token: TokenHook | None = None) -> str:
        """Answer ``message``. With ``on_token``, the answer is also streamed piece by piece."""
        self.history.append({"role": "user", "content": message})
        system = {"role": "system", "content": self.system_prompt(message)}
        schemas = self.tools.schemas()
        for _ in range(self.max_steps):
            reply = self._chat([system, *self.history], schemas, on_token)
            if not reply.tool_calls:
                text = strip_thinking(reply.content)
                if text:
                    self.history.append(reply.as_message())
                    return text
                log.warning("Model gave an empty reply; asking it to answer")
                return self._final(system, on_token, EMPTY_NOTE)
            self.history.append(reply.as_message())
            for call in reply.tool_calls:
                self.on_event("tool", call.name, call.arguments)
                output = self.tools.call(call.name, call.arguments)
                self.on_event("result", call.name, output)
                self.history.append(
                    {"role": "tool", "tool_call_id": call.id, "name": call.name, "content": output}
                )
        log.info("Used all %s tool steps", self.max_steps)
        return self._final(system, on_token, STEPS_NOTE.format(n=self.max_steps))

    def _final(self, system: dict, on_token: TokenHook | None, note: str) -> str:
        """Ask for a plain answer without tools. The note itself isn't kept in history."""
        nudge = {"role": "user", "content": note}
        reply = self._chat([system, *self.history, nudge], [], on_token)
        text = strip_thinking(reply.content)
        if not text:
            log.warning("Model gave an empty reply twice")
            return EMPTY_ANSWER
        self.history.append({"role": "assistant", "content": text})
        return text

    def _chat(self, messages, tools, on_token):
        if on_token is None:
            return self.model.chat(messages, tools)
        return self.model.chat(messages, tools, ThinkFilter(on_token))

"""Tool definitions and the registry the agent uses to run them."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..memory import MemoryStore

# Asked before risky actions; returns True if the user allows it.
Confirm = Callable[[str], bool]

MAX_OUTPUT = 12_000


class ToolError(Exception):
    """A problem the model should hear about (bad path, refused action, ...)."""


# Answers a question about an image: (png_bytes, question) -> answer text.
Vision = Callable[[bytes, str], str]


@dataclass
class ToolContext:
    workspace: Path
    memory: MemoryStore
    confirm: Confirm
    confirm_actions: bool = True
    vision: Vision | None = None
    godot_path: str = ""
    control_minutes: float = 10.0
    # Kinds of action the user allowed for a while: name -> time.monotonic() deadline.
    allowed_until: dict[str, float] = field(default_factory=dict)

    def allow(self, description: str) -> None:
        if self.confirm_actions and not self.confirm(description):
            raise ToolError("The user declined this action. Ask them how to proceed.")

    def allow_for(self, key: str, minutes: float, description: str) -> None:
        """Ask once, then allow this kind of action without asking for ``minutes``."""
        if not self.confirm_actions or time.monotonic() < self.allowed_until.get(key, 0.0):
            return
        self.allow(description)
        self.allowed_until[key] = time.monotonic() + minutes * 60

    def allow_control(self) -> None:
        """Ask before controlling the mouse and keyboard, then allow it for a while."""
        m = self.control_minutes
        shown = int(m) if m >= 1 else m
        self.allow_for(
            "control",
            m,
            f"Control your mouse and keyboard for the next {shown} minutes.\n"
            "To stop it at any time, move your mouse into a corner of the screen.",
        )

    def stop_control(self) -> None:
        self.allowed_until.pop("control", None)

    def resolve(self, path: str) -> Path:
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = self.workspace / p
        p = p.resolve()
        if p != self.workspace and self.workspace not in p.parents:
            raise ToolError(f"{path} is outside the allowed folder {self.workspace}")
        return p


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    run: Callable[..., str]
    required: list[str] = field(default_factory=list)

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required,
                },
            },
        }


class ToolRegistry:
    def __init__(self, tools: list[Tool], ctx: ToolContext):
        self.tools = {t.name: t for t in tools}
        self.ctx = ctx

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools.values()]

    def call(self, name: str, arguments: str) -> str:
        tool = self.tools.get(name)
        if tool is None:
            return f"Error: there is no tool named {name!r}. Available: {', '.join(self.tools)}"
        try:
            args = json.loads(arguments or "{}")
            if not isinstance(args, dict):
                raise ValueError("arguments must be a JSON object")
        except ValueError as exc:
            return f"Error: invalid arguments for {name}: {exc}"
        missing = [r for r in tool.required if r not in args]
        if missing:
            return f"Error: {name} needs {', '.join(missing)}"
        try:
            out = tool.run(self.ctx, **args)
        except ToolError as exc:
            return f"Error: {exc}"
        except TypeError as exc:
            return f"Error: bad arguments for {name}: {exc}"
        except Exception as exc:  # noqa: BLE001 - report every failure back to the model
            return f"Error: {name} failed: {type(exc).__name__}: {exc}"
        out = str(out)
        if len(out) > MAX_OUTPUT:
            out = out[:MAX_OUTPUT] + f"\n... [cut off, {len(out) - MAX_OUTPUT} more characters]"
        return out

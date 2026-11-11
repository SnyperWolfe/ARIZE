from __future__ import annotations

from ..config import Config
from ..memory import MemoryStore
from . import files, godot, memory_tools, phone, screen, web
from .base import Confirm, Tool, ToolContext, ToolError, ToolRegistry, Vision

ALL_TOOLS: list[Tool] = [
    *web.TOOLS,
    *files.TOOLS,
    *memory_tools.TOOLS,
    *godot.TOOLS,
    *screen.TOOLS,
]


# The phone edition: no Godot or screen control; phone tools when Termux:API is there.
MOBILE_TOOLS: list[Tool] = [*web.TOOLS, *files.TOOLS, *memory_tools.TOOLS]


def default_registry(ctx: ToolContext) -> ToolRegistry:
    return ToolRegistry(ALL_TOOLS, ctx)


def mobile_registry(ctx: ToolContext) -> ToolRegistry:
    extra = phone.TOOLS if phone.available() else []
    return ToolRegistry([*MOBILE_TOOLS, *extra], ctx)


def make_context(
    cfg: Config,
    memory: MemoryStore,
    confirm: Confirm,
    vision: Vision | None = None,
    confirm_actions: bool | None = None,
) -> ToolContext:
    return ToolContext(
        workspace=cfg.workspace_path,
        memory=memory,
        confirm=confirm,
        confirm_actions=cfg.confirm_actions if confirm_actions is None else confirm_actions,
        vision=vision,
        godot_path=cfg.godot_path,
        control_minutes=cfg.control_minutes,
    )


__all__ = [
    "ALL_TOOLS",
    "MOBILE_TOOLS",
    "Confirm",
    "Tool",
    "ToolContext",
    "ToolError",
    "ToolRegistry",
    "Vision",
    "default_registry",
    "mobile_registry",
    "make_context",
]

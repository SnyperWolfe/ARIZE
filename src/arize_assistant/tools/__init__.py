from __future__ import annotations

from . import files, memory_tools, web
from .base import Confirm, Tool, ToolContext, ToolError, ToolRegistry

ALL_TOOLS: list[Tool] = [*web.TOOLS, *files.TOOLS, *memory_tools.TOOLS]


def default_registry(ctx: ToolContext) -> ToolRegistry:
    return ToolRegistry(ALL_TOOLS, ctx)


__all__ = [
    "ALL_TOOLS",
    "Confirm",
    "Tool",
    "ToolContext",
    "ToolError",
    "ToolRegistry",
    "default_registry",
]

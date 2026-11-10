"""Tools that let the model save, look up and delete long-term memories."""

from __future__ import annotations

from .base import Tool, ToolContext, ToolError


def remember(ctx: ToolContext, fact: str) -> str:
    m = ctx.memory.add(fact)
    return f"Saved memory #{m.id}: {m.text}"


def search_memory(ctx: ToolContext, query: str) -> str:
    found = ctx.memory.search(query, limit=10)
    if not found:
        return "No matching memories."
    return "\n".join(f"#{m.id}: {m.text}" for m in found)


def forget(ctx: ToolContext, memory_id: int) -> str:
    if not ctx.memory.forget(int(memory_id)):
        raise ToolError(f"There is no memory #{memory_id}")
    return f"Forgot memory #{memory_id}"


TOOLS = [
    Tool(
        "remember",
        "Save a lasting fact about the user or their world (preferences, names, plans, "
        "projects) so you know it in future conversations. One short, self-contained "
        "sentence per fact. Do not save passwords or other secrets.",
        {"fact": {"type": "string", "description": "e.g. 'The user's dog is named Pixel.'"}},
        remember,
        ["fact"],
    ),
    Tool(
        "search_memory",
        "Look up saved memories by keywords.",
        {"query": {"type": "string", "description": "Keywords"}},
        search_memory,
        ["query"],
    ),
    Tool(
        "forget",
        "Delete a saved memory that is wrong, outdated, or the user asked you to forget.",
        {"memory_id": {"type": "integer", "description": "The number shown as #N"}},
        forget,
        ["memory_id"],
    ),
]

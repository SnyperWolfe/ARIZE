"""File and shell tools. Writing and running commands ask the user first."""

from __future__ import annotations

import subprocess

from .base import Tool, ToolContext, ToolError

MAX_READ = 200_000


def read_file(ctx: ToolContext, path: str) -> str:
    p = ctx.resolve(path)
    if not p.is_file():
        raise ToolError(f"{path} is not a file")
    data = p.read_bytes()[:MAX_READ]
    if b"\0" in data[:4096]:
        raise ToolError(f"{path} looks like a binary file")
    return data.decode("utf-8", errors="replace")


def list_folder(ctx: ToolContext, path: str = ".") -> str:
    p = ctx.resolve(path)
    if not p.is_dir():
        raise ToolError(f"{path} is not a folder")
    entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
    lines = [e.name + ("/" if e.is_dir() else "") for e in entries[:500]]
    if len(entries) > 500:
        lines.append(f"... and {len(entries) - 500} more")
    return "\n".join(lines) or "(empty folder)"


def write_file(ctx: ToolContext, path: str, content: str, append: bool = False) -> str:
    p = ctx.resolve(path)
    verb = "Append to" if append else ("Overwrite" if p.exists() else "Create")
    ctx.allow(f"{verb} file {p} ({len(content)} characters)")
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a" if append else "w", encoding="utf-8") as fh:
        fh.write(content)
    return f"Wrote {len(content)} characters to {p}"


def run_command(ctx: ToolContext, command: str, timeout: int = 60) -> str:
    ctx.allow(f"Run command in {ctx.workspace}:\n    {command}")
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=ctx.workspace,
            capture_output=True,
            text=True,
            timeout=max(1, min(int(timeout), 600)),
        )
    except subprocess.TimeoutExpired:
        raise ToolError(f"Command timed out after {timeout}s") from None
    out = f"exit code: {proc.returncode}\n"
    if proc.stdout:
        out += f"--- output ---\n{proc.stdout}"
    if proc.stderr:
        out += f"--- errors ---\n{proc.stderr}"
    return out


TOOLS = [
    Tool(
        "read_file",
        "Read a text file. Paths are relative to the user's workspace folder.",
        {"path": {"type": "string", "description": "File path"}},
        read_file,
        ["path"],
    ),
    Tool(
        "list_folder",
        "List the files and folders inside a folder.",
        {"path": {"type": "string", "description": "Folder path, default '.'"}},
        list_folder,
    ),
    Tool(
        "write_file",
        "Create or overwrite a text file (the user is asked to approve).",
        {
            "path": {"type": "string", "description": "File path"},
            "content": {"type": "string", "description": "Full text to write"},
            "append": {"type": "boolean", "description": "Add to the end instead of replacing"},
        },
        write_file,
        ["path", "content"],
    ),
    Tool(
        "run_command",
        "Run a shell command in the workspace folder and return its output "
        "(the user is asked to approve).",
        {
            "command": {"type": "string", "description": "Shell command"},
            "timeout": {"type": "integer", "description": "Seconds before giving up, default 60"},
        },
        run_command,
        ["command"],
    ),
]

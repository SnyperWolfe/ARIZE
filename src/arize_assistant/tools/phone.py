"""Phone tools for ARIZE running on Android inside Termux.

They use the Termux:API add-on (the ``termux-*`` commands from ``pkg install termux-api``
plus the Termux:API app), so they only appear when those commands exist.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from urllib.parse import urlparse

from .base import Tool, ToolContext, ToolError


def available() -> bool:
    return shutil.which("termux-battery-status") is not None


def _run(*cmd: str, stdin: str | None = None, timeout: float = 15) -> str:
    if shutil.which(cmd[0]) is None:
        raise ToolError(
            f"{cmd[0]} isn't installed. In Termux run: pkg install termux-api, and install "
            "the Termux:API app from F-Droid."
        )
    try:
        proc = subprocess.run(
            list(cmd), input=stdin, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        raise ToolError(
            f"{cmd[0]} didn't answer. Open the Termux:API app once and allow its permissions."
        ) from None
    if proc.returncode != 0:
        raise ToolError((proc.stderr or proc.stdout or f"{cmd[0]} failed").strip()[:300])
    return proc.stdout.strip()


def phone_battery(ctx: ToolContext) -> str:
    data = json.loads(_run("termux-battery-status") or "{}")
    plugged = data.get("plugged", "UNPLUGGED")
    charging = "not charging" if plugged == "UNPLUGGED" else f"charging ({plugged.lower()})"
    temp = data.get("temperature")
    return (
        f"Battery {data.get('percentage', '?')}%, {charging}, health "
        f"{str(data.get('health', '?')).lower()}"
        + (f", {temp:.0f} °C" if isinstance(temp, (int, float)) else "")
    )


def clipboard_read(ctx: ToolContext) -> str:
    text = _run("termux-clipboard-get")
    return text if text else "(the clipboard is empty)"


def clipboard_write(ctx: ToolContext, text: str) -> str:
    _run("termux-clipboard-set", stdin=text)
    return f"Copied {len(text)} characters to the clipboard"


def open_link(ctx: ToolContext, url: str) -> str:
    if urlparse(url).scheme not in {"http", "https", "mailto", "tel", "geo"}:
        raise ToolError("Only http(s), mailto:, tel: and geo: links can be opened")
    _run("termux-open-url", url)
    return f"Opened {url}"


def notify(ctx: ToolContext, title: str, text: str = "") -> str:
    _run("termux-notification", "--title", title, "--content", text, "--id", "arize")
    return "Notification shown"


def vibrate(ctx: ToolContext, milliseconds: int = 500) -> str:
    ms = max(50, min(int(milliseconds), 5000))
    _run("termux-vibrate", "-d", str(ms), "-f")
    return f"Vibrated for {ms} ms"


def flashlight(ctx: ToolContext, on: bool) -> str:
    _run("termux-torch", "on" if on else "off")
    return f"Flashlight {'on' if on else 'off'}"


TOOLS = [
    Tool("phone_battery", "Check the phone's battery level and charging.", {}, phone_battery),
    Tool("clipboard_read", "Read the text on the phone's clipboard.", {}, clipboard_read),
    Tool(
        "clipboard_write",
        "Copy text to the phone's clipboard so the user can paste it anywhere.",
        {"text": {"type": "string"}},
        clipboard_write,
        ["text"],
    ),
    Tool(
        "open_link",
        "Open a link on the phone: web pages, maps (geo:lat,lon or a maps URL), "
        "mailto: or tel: links.",
        {"url": {"type": "string"}},
        open_link,
        ["url"],
    ),
    Tool(
        "notify",
        "Show a notification on the phone.",
        {"title": {"type": "string"}, "text": {"type": "string"}},
        notify,
        ["title"],
    ),
    Tool(
        "vibrate",
        "Vibrate the phone.",
        {"milliseconds": {"type": "integer", "description": "50-5000, default 500"}},
        vibrate,
    ),
    Tool(
        "flashlight",
        "Turn the phone's flashlight on or off.",
        {"on": {"type": "boolean"}},
        flashlight,
        ["on"],
    ),
]

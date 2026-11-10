"""Screen control: look at the screen, click, type, press and hold keys.

Positions are given on a 0-1000 grid: x from the left edge (0) to the right edge (1000),
y from the top (0) to the bottom (1000). That works on any screen size or resolution.

Safety: ARIZE asks once before taking control (then allows it for a few minutes), and
moving the mouse into any corner of the screen stops it immediately.
"""

from __future__ import annotations

import io
import sys
import time

from .base import Tool, ToolContext, ToolError

MAX_SHOT_WIDTH = 1280
KEY_ALIASES = {
    "control": "ctrl", "escape": "esc", "return": "enter", "windows": "win", "cmd": "win",
    "command": "win", "super": "win", "spacebar": "space", "del": "delete", "pgup": "pageup",
    "pgdn": "pagedown", "arrowup": "up", "arrowdown": "down", "arrowleft": "left",
    "arrowright": "right", "option": "alt",
}  # fmt: skip

LOOK_PROMPT = """\
This is a screenshot of the user's screen. Positions use a 0-1000 grid: x goes from the \
left edge (0) to the right edge (1000), y from the top (0) to the bottom (1000). The \
rulers along the top and left edges mark every 100.

Answer the question below. Describe what matters for it. Whenever you mention something \
that could be clicked or that matters for positioning (buttons, menus, fields, \
characters, enemies), give its centre as (x, y) on the 0-1000 grid.

Question: {question}"""


class Desktop:
    """Real mouse, keyboard and screen. Replaced by a fake in tests."""

    def __init__(self):
        try:
            import pyautogui
        # e.g. no display on a server; on Linux it can even call sys.exit() without tkinter
        except (Exception, SystemExit) as exc:
            raise ToolError(f"Screen control isn't available here: {exc}") from exc
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.05
        self.gui = pyautogui
        self.keys = pyautogui
        if sys.platform == "win32":
            try:  # sends keys the way DirectX games expect (scan codes)
                import pydirectinput

                pydirectinput.FAILSAFE = True
                pydirectinput.PAUSE = 0.05
                self.keys = pydirectinput
            except ImportError:
                pass

    def size(self) -> tuple[int, int]:
        w, h = self.gui.size()
        return int(w), int(h)

    def screenshot(self):
        try:
            import mss
            from PIL import Image

            with mss.mss() as sct:
                raw = sct.grab(sct.monitors[1])
                return Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
        except ImportError:
            return self.gui.screenshot()

    def click(self, x, y, button, clicks):
        self.keys.click(x=x, y=y, button=button, clicks=clicks, interval=0.08)

    def move(self, x, y):
        self.keys.moveTo(x, y)

    def drag(self, x1, y1, x2, y2):
        self.gui.moveTo(x1, y1)
        self.gui.dragTo(x2, y2, duration=0.4, button="left")

    def write(self, text):
        self.gui.write(text, interval=0.02)

    def _key_backend(self, key):
        if key not in self.gui.KEYBOARD_KEYS:
            raise ToolError(f"Unknown key {key!r}")
        mapping = getattr(self.keys, "KEYBOARD_MAPPING", None)
        if mapping is not None and key not in mapping:
            return self.gui  # pydirectinput ignores keys it doesn't know (e.g. 'win')
        return self.keys

    def key_down(self, key):
        self._key_backend(key).keyDown(key)

    def key_up(self, key):
        self._key_backend(key).keyUp(key)

    def scroll(self, amount, x, y):
        self.gui.scroll(amount, x=x, y=y)

    def windows(self) -> list[str]:
        try:
            import pygetwindow
        except ImportError as exc:
            raise ToolError("Listing windows only works on Windows") from exc
        return [t for t in pygetwindow.getAllTitles() if t.strip()]

    def focus(self, title: str) -> str:
        import pygetwindow

        matches = [w for w in pygetwindow.getAllWindows() if title.lower() in w.title.lower()]
        if not matches:
            raise ToolError(f"No open window has {title!r} in its title")
        win = matches[0]
        try:
            if win.isMinimized:
                win.restore()
            win.activate()
        except Exception:  # noqa: BLE001 - Windows sometimes refuses; this usually works
            win.minimize()
            win.restore()
        return win.title


_desktop: Desktop | None = None


def desktop() -> Desktop:
    global _desktop
    if _desktop is None:
        _desktop = Desktop()
    return _desktop


def _to_pixels(x: float, y: float) -> tuple[int, int]:
    if not (0 <= float(x) <= 1000 and 0 <= float(y) <= 1000):
        raise ToolError("x and y must be between 0 and 1000")
    w, h = desktop().size()
    # Stay a few pixels away from the corners: those stop screen control.
    px = min(max(round(float(x) / 1000 * (w - 1)), 3), w - 4)
    py = min(max(round(float(y) / 1000 * (h - 1)), 3), h - 4)
    return px, py


def _keys(spec: str) -> list[str]:
    keys = [k.strip().lower() for k in spec.replace(" ", "").split("+") if k.strip()]
    if not keys:
        raise ToolError("No key given")
    return [KEY_ALIASES.get(k, k) for k in keys]


def _hold(combo: list[str], seconds: float) -> None:
    """Press keys down, wait, then release them, even if something fails part-way."""
    d = desktop()
    pressed = []
    try:
        for k in combo:
            d.key_down(k)
            pressed.append(k)
        time.sleep(seconds)
    finally:
        for k in reversed(pressed):
            d.key_up(k)


def _control(ctx: ToolContext, action):
    """Run a mouse/keyboard action, turning the corner-stop into a clear message."""
    ctx.allow_control()
    try:
        return action()
    except Exception as exc:
        if type(exc).__name__ == "FailSafeException":
            ctx.stop_control()
            raise ToolError(
                "Stopped: the user moved the mouse into a corner of the screen to take back "
                "control. Don't continue controlling the screen unless they ask again."
            ) from None
        raise


def _with_rulers(img):
    from PIL import ImageDraw

    img = img.convert("RGB")
    if img.width > MAX_SHOT_WIDTH:
        img = img.resize((MAX_SHOT_WIDTH, round(img.height * MAX_SHOT_WIDTH / img.width)))
    draw = ImageDraw.Draw(img)
    for i in range(0, 1001, 100):
        x = round(i / 1000 * (img.width - 1))
        y = round(i / 1000 * (img.height - 1))
        draw.line([(x, 0), (x, 10)], fill=(255, 0, 255), width=2)
        draw.line([(0, y), (10, y)], fill=(255, 0, 255), width=2)
        if 0 < i < 1000:
            draw.text((x + 3, 10), str(i), fill=(255, 0, 255))
            draw.text((12, y - 5), str(i), fill=(255, 0, 255))
    return img


def look_at_screen(ctx: ToolContext, question: str = "What is on the screen?") -> str:
    if ctx.vision is None:
        raise ToolError("No vision model is set up, so ARIZE can't see the screen")
    img = _with_rulers(desktop().screenshot())
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    try:
        answer = ctx.vision(buf.getvalue(), LOOK_PROMPT.format(question=question))
    except Exception as exc:  # noqa: BLE001
        raise ToolError(
            f"Looking at the screen failed ({exc}). The model must be able to see images: "
            "set vision_model (e.g. qwen3-vl:8b) in the settings file."
        ) from exc
    return answer.strip() or "(the vision model gave no answer)"


def click(ctx: ToolContext, x: float, y: float, button: str = "left", clicks: int = 1) -> str:
    if button not in {"left", "right", "middle"}:
        raise ToolError("button must be left, right or middle")
    px, py = _to_pixels(x, y)
    clicks = max(1, min(int(clicks), 3))
    _control(ctx, lambda: desktop().click(px, py, button, clicks))
    return f"{'Double-' if clicks == 2 else ''}{button}-clicked at ({x}, {y})"


def move_mouse(ctx: ToolContext, x: float, y: float) -> str:
    px, py = _to_pixels(x, y)
    _control(ctx, lambda: desktop().move(px, py))
    return f"Moved the mouse to ({x}, {y})"


def drag(ctx: ToolContext, from_x: float, from_y: float, to_x: float, to_y: float) -> str:
    a, b = _to_pixels(from_x, from_y)
    c, d = _to_pixels(to_x, to_y)
    _control(ctx, lambda: desktop().drag(a, b, c, d))
    return f"Dragged from ({from_x}, {from_y}) to ({to_x}, {to_y})"


def type_text(ctx: ToolContext, text: str) -> str:
    _control(ctx, lambda: desktop().write(text))
    return f"Typed {len(text)} characters"


def press_keys(ctx: ToolContext, keys: str, times: int = 1) -> str:
    combo = _keys(keys)
    times = max(1, min(int(times), 50))

    def press():
        for _ in range(times):
            _hold(combo, 0.03)
            time.sleep(0.05)

    _control(ctx, press)
    return f"Pressed {'+'.join(combo)}" + (f" {times} times" if times > 1 else "")


def hold_keys(ctx: ToolContext, keys: str, seconds: float = 1.0) -> str:
    combo = _keys(keys)
    seconds = max(0.05, min(float(seconds), 10.0))

    _control(ctx, lambda: _hold(combo, seconds))
    return f"Held {'+'.join(combo)} for {seconds:g} seconds"


def scroll(ctx: ToolContext, amount: int, x: float = 500, y: float = 500) -> str:
    px, py = _to_pixels(x, y)
    _control(ctx, lambda: desktop().scroll(int(amount), px, py))
    return f"Scrolled {'up' if int(amount) > 0 else 'down'} {abs(int(amount))} at ({x}, {y})"


def wait(ctx: ToolContext, seconds: float = 1.0) -> str:
    seconds = max(0.0, min(float(seconds), 30.0))
    time.sleep(seconds)
    return f"Waited {seconds:g} seconds"


def list_windows(ctx: ToolContext) -> str:
    titles = desktop().windows()
    return "\n".join(titles) or "No windows found"


def focus_window(ctx: ToolContext, title: str) -> str:
    found = _control(ctx, lambda: desktop().focus(title))
    return f"Brought '{found}' to the front"


_XY = {
    "x": {"type": "number", "description": "0 (left edge) to 1000 (right edge)"},
    "y": {"type": "number", "description": "0 (top) to 1000 (bottom)"},
}
_KEYS = {
    "type": "string",
    "description": "A key or combination, e.g. 'w', 'space', 'enter', 'f5', 'ctrl+s', 'shift+a'",
}

TOOLS = [
    Tool(
        "look_at_screen",
        "Take a screenshot and ask a question about it. Returns a description with "
        "(x, y) positions on a 0-1000 grid. Look before clicking, and look again after "
        "acting to check what happened.",
        {"question": {"type": "string", "description": "What you need to know"}},
        look_at_screen,
    ),
    Tool(
        "click",
        "Click at a position on the screen (0-1000 grid from look_at_screen).",
        {
            **_XY,
            "button": {"type": "string", "enum": ["left", "right", "middle"]},
            "clicks": {"type": "integer", "description": "2 for a double-click"},
        },
        click,
        ["x", "y"],
    ),
    Tool("move_mouse", "Move the mouse without clicking.", dict(_XY), move_mouse, ["x", "y"]),
    Tool(
        "drag",
        "Drag with the left mouse button from one position to another.",
        {
            "from_x": _XY["x"],
            "from_y": _XY["y"],
            "to_x": _XY["x"],
            "to_y": _XY["y"],
        },  # fmt: skip
        drag,
        ["from_x", "from_y", "to_x", "to_y"],
    ),
    Tool(
        "type_text",
        "Type text into whatever has keyboard focus (click a text field first).",
        {"text": {"type": "string"}},
        type_text,
        ["text"],
    ),
    Tool(
        "press_keys",
        "Press a key or key combination, optionally several times.",
        {"keys": _KEYS, "times": {"type": "integer", "description": "Default 1"}},
        press_keys,
        ["keys"],
    ),
    Tool(
        "hold_keys",
        "Hold keys down for some seconds, e.g. to walk in a game ('w' for 2 seconds) "
        "or run ('shift+w').",
        {"keys": _KEYS, "seconds": {"type": "number", "description": "Up to 10"}},
        hold_keys,
        ["keys"],
    ),
    Tool(
        "scroll",
        "Scroll the mouse wheel. Positive scrolls up, negative scrolls down.",
        {"amount": {"type": "integer", "description": "e.g. -5"}, **_XY},
        scroll,
        ["amount"],
    ),
    Tool(
        "wait",
        "Wait a few seconds, e.g. for a game or program to load.",
        {"seconds": {"type": "number", "description": "Up to 30"}},
        wait,
    ),
    Tool("list_windows", "List the titles of open windows.", {}, list_windows),
    Tool(
        "focus_window",
        "Bring a window to the front by part of its title, e.g. 'Godot' or 'Minecraft'.",
        {"title": {"type": "string"}},
        focus_window,
        ["title"],
    ),
]

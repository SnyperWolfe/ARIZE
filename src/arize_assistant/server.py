"""The local web server behind the ARIZE desktop window.

It only listens on 127.0.0.1, only answers requests addressed to that host, and every
API call must carry a secret token that is written into the page it serves. Other
websites open in your browser can't read that page, so they can't use the API.
"""

from __future__ import annotations

import json
import queue
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from itertools import count

from . import __version__
from .agent import Agent
from .config import Config
from .llm import ChatModel, LLMError, OpenAICompatibleModel
from .memory import MemoryStore
from .tools import default_registry, make_context

CONFIRM_TIMEOUT = 300  # seconds to wait for Allow / Deny before treating it as Deny
STATIC = {
    "/icon.png": ("arize.png", "image/png"),
    "/favicon.ico": ("arize.ico", "image/x-icon"),
}


class ArizeApp:
    """Shared state: one assistant, one conversation at a time."""

    def __init__(self, cfg: Config, model: ChatModel | None = None):
        self.cfg = cfg
        self.token = secrets.token_urlsafe(24)
        self.memory = MemoryStore(cfg.memory_path)
        model = model or OpenAICompatibleModel(cfg.base_url, cfg.model, cfg.api_key, cfg.timeout)
        look = getattr(model, "look", None)
        vision = (lambda png, q: look(png, q, cfg.vision_model)) if look else None
        ctx = make_context(cfg, self.memory, self._confirm, vision=vision)
        self.agent = Agent(
            model,
            default_registry(ctx),
            self.memory,
            cfg.max_steps,
            on_event=self._on_event,
            think=cfg.think,
        )
        self.busy = threading.Lock()
        self.last_seen = time.monotonic()
        self._events: queue.Queue | None = None
        self._pending: dict[str, list] = {}  # confirm id -> [threading.Event, allowed]
        self._ids = count(1)

    # --- called from the agent's thread -------------------------------------

    def _emit(self, event: dict) -> None:
        if self._events is not None:
            self._events.put(event)

    def _on_event(self, kind: str, name: str, text: str) -> None:
        if kind == "tool":
            self._emit({"type": "tool", "name": name, "args": text})
        else:
            ok = not text.startswith("Error:")
            self._emit({"type": "result", "name": name, "ok": ok, "preview": text[:300]})

    def _confirm(self, description: str) -> bool:
        cid = str(next(self._ids))
        waiter = [threading.Event(), False]
        self._pending[cid] = waiter
        self._emit({"type": "confirm", "id": cid, "text": description})
        waiter[0].wait(CONFIRM_TIMEOUT)
        self._pending.pop(cid, None)
        return waiter[1]

    # --- called from request handlers ---------------------------------------

    def answer_confirm(self, cid: str, allow: bool) -> bool:
        waiter = self._pending.get(cid)
        if waiter is None:
            return False
        waiter[1] = bool(allow)
        waiter[0].set()
        return True

    def chat(self, message: str):
        """Yield events for one message: token, tool, result, confirm, then done or error."""
        events: queue.Queue = queue.Queue()
        self._events = events

        def work():
            try:
                text = self.agent.ask(
                    message, on_token=lambda t: events.put({"type": "token", "text": t})
                )
                events.put({"type": "done", "text": text})
            except LLMError as exc:
                self.agent.history.pop()  # forget the unanswered message so it can be retried
                events.put({"type": "error", "text": str(exc)})
            except Exception as exc:  # noqa: BLE001 - show any failure in the window
                events.put({"type": "error", "text": f"{type(exc).__name__}: {exc}"})

        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        try:
            while True:
                event = events.get()
                yield event
                if event["type"] in {"done", "error"}:
                    break
        finally:
            # If the window went away mid-answer, deny anything still waiting.
            for waiter in list(self._pending.values()):
                waiter[0].set()
            worker.join(timeout=1)
            self._events = None


class Handler(BaseHTTPRequestHandler):
    app: ArizeApp  # set by make_server
    server_version = "ARIZE"

    def log_message(self, *args):  # keep the console quiet
        pass

    # --- helpers ------------------------------------------------------------

    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data, code: int = 200) -> None:
        self._send(code, json.dumps(data).encode(), "application/json")

    def _host_ok(self) -> bool:
        port = self.server.server_address[1]
        return self.headers.get("Host", "") in {f"127.0.0.1:{port}", f"localhost:{port}"}

    def _token_ok(self) -> bool:
        return secrets.compare_digest(self.headers.get("X-Arize-Token", ""), self.app.token)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        data = json.loads(self.rfile.read(min(length, 1_000_000)))
        return data if isinstance(data, dict) else {}

    # --- routes -------------------------------------------------------------

    def do_GET(self):
        if not self._host_ok():
            return self._send(403, b"Forbidden", "text/plain")
        path = self.path.split("?")[0]
        if path == "/":
            page = _static("index.html").decode().replace("__ARIZE_TOKEN__", self.app.token)
            csp = (
                "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'"
            )
            return self._send(
                200, page.encode(), "text/html; charset=utf-8", {"Content-Security-Policy": csp}
            )
        if path in STATIC:
            name, ctype = STATIC[path]
            return self._send(200, _static(name), ctype)
        if path == "/api/health":  # lets a second launch find the running app
            return self._json({"app": "arize"})
        if not self._token_ok():
            return self._send(403, b"Forbidden", "text/plain")
        self.app.last_seen = time.monotonic()
        if path == "/api/ping":
            return self._json({"ok": True})
        if path == "/api/info":
            cfg = self.app.cfg
            return self._json({"version": __version__, "model": cfg.model, "think": cfg.think})
        if path == "/api/memories":
            mems = self.app.memory.all()
            return self._json([{"id": m.id, "text": m.text, "created": m.created} for m in mems])
        self._send(404, b"Not found", "text/plain")

    def do_POST(self):
        if not (self._host_ok() and self._token_ok()):
            return self._send(403, b"Forbidden", "text/plain")
        self.app.last_seen = time.monotonic()
        try:
            body = self._body()
        except ValueError:
            return self._json({"error": "bad JSON"}, 400)
        path = self.path.split("?")[0]
        if path == "/api/chat":
            return self._chat(str(body.get("message", "")).strip())
        if path == "/api/confirm":
            ok = self.app.answer_confirm(str(body.get("id")), bool(body.get("allow")))
            return self._json({"ok": ok})
        if path == "/api/new":
            with self.app.busy:
                self.app.agent.reset()
            return self._json({"ok": True})
        if path == "/api/memories/delete":
            return self._json({"ok": self.app.memory.forget(int(body.get("id", 0)))})
        self._send(404, b"Not found", "text/plain")

    def _chat(self, message: str) -> None:
        if not message:
            return self._json({"error": "empty message"}, 400)
        if not self.app.busy.acquire(blocking=False):
            return self._json({"error": "ARIZE is still answering"}, 409)
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            for event in self.app.chat(message):
                self.app.last_seen = time.monotonic()
                self.wfile.write((json.dumps(event) + "\n").encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.app.busy.release()
        self.close_connection = True


def _static(name: str) -> bytes:
    return resources.files("arize_assistant").joinpath("web", name).read_bytes()


def make_server(app: ArizeApp, port: int = 0) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"app": app})
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    server.daemon_threads = True
    return server


def stop_when_idle(server: ThreadingHTTPServer, app: ArizeApp, idle: float = 45) -> None:
    """Shut the server down once the window has been closed (no pings for ``idle`` seconds)."""

    def watch():
        while True:
            time.sleep(5)
            if time.monotonic() - app.last_seen > idle and not app.busy.locked():
                server.shutdown()
                return

    threading.Thread(target=watch, daemon=True).start()

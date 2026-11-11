"""ARIZE on an Android phone, inside Termux: start Ollama, start ARIZE, open the app."""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
import webbrowser

import httpx

from . import __version__
from .config import Config, write_default_config
from .desktop import _running_version, _stop_old
from .llm import is_ollama

PORT = 8765
log = logging.getLogger("arize")


def _ollama_up(cfg: Config) -> bool:
    root = cfg.base_url.rstrip("/").removesuffix("/v1")
    try:
        return httpx.get(root + "/api/tags", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


def ensure_ollama(cfg: Config) -> bool:
    """Start Ollama in the background if it isn't running yet."""
    if not is_ollama(cfg.base_url) or _ollama_up(cfg):
        return True
    if shutil.which("ollama") is None:
        print("Ollama isn't installed. In Termux run: pkg install ollama")
        return False
    print("Starting Ollama…")
    log_file = open(cfg.home / "ollama.log", "a")  # noqa: SIM115 - kept open for the child
    subprocess.Popen(["ollama", "serve"], stdout=log_file, stderr=log_file, start_new_session=True)
    for _ in range(60):
        if _ollama_up(cfg):
            return True
        time.sleep(0.5)
    print(f"Ollama didn't start. See {cfg.home / 'ollama.log'}")
    return False


def open_app(url: str) -> None:
    if shutil.which("termux-open-url"):
        subprocess.run(["termux-open-url", url], check=False)
    else:
        webbrowser.open(url)


def run_mobile(port: int = PORT, open_ui: bool = True) -> int:
    from .server import ArizeApp, make_server

    write_default_config(mobile=True)
    cfg = Config.load()
    cfg.home.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=cfg.home / "app.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    url = f"http://127.0.0.1:{port}/"
    running = _running_version(port)
    if running == __version__:
        print("ARIZE is already running.")
        if open_ui:
            open_app(url)
        return 0
    if running is not None:
        print(f"Replacing ARIZE {running} with {__version__}…")
        _stop_old(cfg, port)

    if shutil.which("termux-wake-lock"):
        # Keeps Android from pausing Termux (and ARIZE) while the screen is off.
        subprocess.run(["termux-wake-lock"], check=False)
    ensure_ollama(cfg)

    app = ArizeApp(cfg, mobile=True)
    server = make_server(app, port)
    try:
        (cfg.home / "app.token").write_text(app.token, encoding="utf-8")
    except OSError:
        pass
    print(f"ARIZE {__version__} is running at {url} (model {cfg.model})")
    print("Leave this open. To stop ARIZE, press Ctrl+C or close this Termux session.")
    log.info("ARIZE mobile listening on %s (model %s)", url, cfg.model)
    if open_ui:
        open_app(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    print("ARIZE stopped.")
    return 0

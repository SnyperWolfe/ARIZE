"""Open ARIZE in its own window, and create desktop / Start menu shortcuts."""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import sys
import time
import webbrowser
from importlib import resources
from pathlib import Path

import httpx

from . import __version__
from .config import Config, write_default_config

DEFAULT_PORT = 8765
log = logging.getLogger("arize")


# --- finding a browser that can show a chrome-less "app" window ----------------


def _app_browser() -> list[str] | None:
    system = platform.system()
    candidates: list[str] = []
    if system == "Windows":
        for base in filter(
            None, [os.environ.get(v) for v in ("PROGRAMFILES(X86)", "PROGRAMFILES", "LOCALAPPDATA")]
        ):
            candidates += [
                rf"{base}\Microsoft\Edge\Application\msedge.exe",
                rf"{base}\Google\Chrome\Application\chrome.exe",
                rf"{base}\BraveSoftware\Brave-Browser\Application\brave.exe",
            ]
    elif system == "Darwin":
        candidates = [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
            "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
        ]
    else:
        candidates = [
            shutil.which(n) or ""
            for n in ("google-chrome", "chromium", "chromium-browser", "microsoft-edge", "brave")
        ]
    for c in candidates:
        if c and Path(c).exists():
            return [c]
    return None


def open_window(url: str, cfg: Config) -> None:
    browser = _app_browser()
    if browser is None:
        webbrowser.open(url)
        return
    profile = cfg.home / "window"  # separate profile = its own window, remembers mic permission
    subprocess.Popen(
        [
            *browser,
            f"--app={url}",
            f"--user-data-dir={profile}",
            "--window-size=1180,820",
            "--no-first-run",
            "--no-default-browser-check",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# --- running the app ------------------------------------------------------------


def _running_version(port: int) -> str | None:
    """Version of the ARIZE app already running on ``port``, or None."""
    try:
        r = httpx.get(f"http://127.0.0.1:{port}/api/health", timeout=1)
        data = r.json()
    except (httpx.HTTPError, ValueError):
        return None
    if data.get("app") != "arize":
        return None
    return str(data.get("version", "old"))


def _stop_old(cfg: Config, port: int) -> bool:
    """Ask an older ARIZE still running in the background to quit (after an update)."""
    try:
        token = (cfg.home / "app.token").read_text(encoding="utf-8").strip()
        httpx.post(
            f"http://127.0.0.1:{port}/api/shutdown", headers={"X-Arize-Token": token}, timeout=3
        )
    except (OSError, httpx.HTTPError):
        return False
    for _ in range(50):
        if _running_version(port) is None:
            return True
        time.sleep(0.1)
    return False


def run_app(cfg: Config, port: int = DEFAULT_PORT, open_ui: bool = True) -> int:
    from .server import ArizeApp, make_server, stop_when_idle

    write_default_config()
    cfg.home.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=cfg.home / "app.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    running = _running_version(port)
    if running == __version__:
        log.info("ARIZE already running on port %s; opening another window", port)
        if open_ui:
            open_window(f"http://127.0.0.1:{port}/", cfg)
        return 0
    if running is not None:
        log.info("Replacing ARIZE %s still running on port %s", running, port)
        if not _stop_old(cfg, port):
            log.warning("Couldn't stop the old ARIZE; starting on another port")
    app = ArizeApp(cfg)
    try:
        server = make_server(app, port)
    except OSError:
        server = make_server(app, 0)  # port taken by something else: pick any free one
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    try:  # lets the next version (after an update) ask this one to quit
        (cfg.home / "app.token").write_text(app.token, encoding="utf-8")
    except OSError:
        pass
    log.info("ARIZE app listening on %s (model %s)", url, cfg.model)
    if open_ui:
        app.last_seen = time.monotonic() + 60  # give the window time to open
        stop_when_idle(server, app)
        open_window(url, cfg)
    else:
        print(f"ARIZE is running at {url}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    log.info("ARIZE app stopped")
    return 0


# --- shortcuts -----------------------------------------------------------------


def _asset(name: str) -> Path:
    return Path(str(resources.files("arize_assistant").joinpath("web", name)))


def _windowless_python() -> str:
    exe = Path(sys.executable)
    pythonw = exe.with_name("pythonw.exe")
    return str(pythonw if pythonw.exists() else exe)


def _ps_quote(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def create_shortcuts() -> list[str]:
    """Create ARIZE shortcuts for this system. Returns where they were put."""
    system = platform.system()
    if system == "Windows":
        return _windows_shortcuts()
    if system == "Darwin":
        return _mac_shortcut()
    return _linux_shortcuts()


def _windows_shortcuts() -> list[str]:
    icon = _asset("arize.ico")
    target, args = _windowless_python(), "-m arize_assistant app"
    script = f"""
$shell = New-Object -ComObject WScript.Shell
$places = @([Environment]::GetFolderPath('Desktop'),
            [Environment]::GetFolderPath('Programs'))
foreach ($dir in $places) {{
  $path = Join-Path $dir 'ARIZE.lnk'
  $lnk = $shell.CreateShortcut($path)
  $lnk.TargetPath = {_ps_quote(target)}
  $lnk.Arguments = {_ps_quote(args)}
  $lnk.WorkingDirectory = $HOME
  $lnk.IconLocation = {_ps_quote(str(icon))}
  $lnk.Description = 'ARIZE - your private AI assistant'
  $lnk.Save()
  Write-Output $path
}}
"""
    out = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def _linux_shortcuts() -> list[str]:
    entry = (
        "[Desktop Entry]\nType=Application\nName=ARIZE\n"
        "Comment=Your private AI assistant\n"
        f'Exec="{sys.executable}" -m arize_assistant app\n'
        f"Icon={_asset('arize.png')}\nTerminal=false\nCategories=Utility;\n"
    )
    made = []
    apps = Path.home() / ".local/share/applications"
    desktop = Path.home() / "Desktop"
    for folder in (apps, desktop):
        if folder == apps or folder.exists():
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / "arize.desktop"
            path.write_text(entry, encoding="utf-8")
            path.chmod(0o755)
            made.append(str(path))
    return made


def _mac_shortcut() -> list[str]:
    path = Path.home() / "Desktop" / "ARIZE.command"
    path.write_text(
        f'#!/bin/sh\nnohup "{sys.executable}" -m arize_assistant app >/dev/null 2>&1 &\n'
        "osascript -e 'tell application \"Terminal\" to close front window' >/dev/null 2>&1 &\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return [str(path)]

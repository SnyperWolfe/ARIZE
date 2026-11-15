"""Update ARIZE from GitHub: check for a newer version, install it, restart the app.

Installs straight from the repository's zip with pip, so it doesn't matter where (or
whether) the source folder still is.
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import httpx

from . import __version__

REPO = "SnyperWolfe/ARIZE"
BRANCH = "main"
_VERSION = re.compile(r'__version__\s*=\s*"([^"]+)"')


def _version_url(repo: str) -> str:
    return f"https://raw.githubusercontent.com/{repo}/{BRANCH}/src/arize_assistant/__init__.py"


def zip_url(repo: str = REPO) -> str:
    return f"https://github.com/{repo}/archive/refs/heads/{BRANCH}.zip"


def parse_version(text: str) -> tuple[int, ...]:
    return tuple(int(n) for n in re.findall(r"\d+", text)[:3])


def latest_version(repo: str = REPO) -> str:
    """The version on GitHub. Raises httpx.HTTPError or ValueError if it can't tell."""
    resp = httpx.get(_version_url(repo), timeout=15, follow_redirects=True)
    resp.raise_for_status()
    found = _VERSION.search(resp.text)
    if not found:
        raise ValueError("couldn't find the version number on GitHub")
    return found.group(1)


def check(repo: str = REPO) -> dict:
    try:
        latest = latest_version(repo)
    except (httpx.HTTPError, ValueError) as exc:
        return {"current": __version__, "latest": None, "newer": False, "error": str(exc)}
    newer = parse_version(latest) > parse_version(__version__)
    return {"current": __version__, "latest": latest, "newer": newer}


def _console_python() -> str:
    """python.exe rather than pythonw.exe, so pip's output can be read."""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe" and exe.with_name("python.exe").exists():
        return str(exe.with_name("python.exe"))
    return str(exe)


def install(repo: str = REPO) -> Iterator[str]:
    """Install the latest version with pip, yielding its output line by line.

    The last line is "OK" on success or "FAILED (exit code N)".
    """
    cmd = [
        _console_python(), "-m", "pip", "install", "--upgrade",
        "--disable-pip-version-check", "--no-input", zip_url(repo),
    ]  # fmt: skip
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        creationflags=flags,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        if line.strip():
            yield line.rstrip()
    code = proc.wait()
    yield "OK" if code == 0 else f"FAILED (exit code {code})"


def restart_app(port: int) -> None:
    """Start the freshly installed ARIZE; it takes over from this one on the same port."""
    exe = Path(sys.executable)
    pythonw = exe.with_name("pythonw.exe")
    python = str(pythonw if pythonw.exists() else exe)
    kwargs: dict = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen(
        [python, "-m", "arize_assistant", "app", "--port", str(port), "--replace"], **kwargs
    )

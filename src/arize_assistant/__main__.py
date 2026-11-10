import os
import sys
import traceback

# Started from the desktop icon there is no console window: sys.stdout and sys.stderr
# are None, and anything printed would crash. Send it all to a log file instead.
WINDOWLESS = sys.stdout is None or sys.stderr is None
if WINDOWLESS:
    log_dir = os.path.join(os.environ.get("ARIZE_HOME") or os.path.expanduser("~/.arize"))
    os.makedirs(log_dir, exist_ok=True)
    log_file = open(os.path.join(log_dir, "app.log"), "a", encoding="utf-8", buffering=1)  # noqa: SIM115
    sys.stdout = sys.stdout or log_file
    sys.stderr = sys.stderr or log_file


def _show_error(text: str) -> None:
    """Tell the user something went wrong, even without a console window."""
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, text, "ARIZE could not start", 0x10)


try:
    from .cli import main

    sys.exit(main())
except SystemExit:
    raise
except BaseException as exc:
    if not WINDOWLESS:
        raise
    traceback.print_exc()
    _show_error(
        f"{type(exc).__name__}: {exc}\n\nDetails were saved in {os.path.join(log_dir, 'app.log')}"
    )
    sys.exit(1)

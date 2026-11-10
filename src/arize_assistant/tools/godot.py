"""Godot 4 tools: inspect a project, create one, run it to catch errors, open the editor.

Scripts and scenes are written with the normal file tools; these tools add what's
specific to Godot, above all running the game headless and reading back its errors so
ARIZE can fix them.
"""

from __future__ import annotations

import glob
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

from .base import Tool, ToolContext, ToolError

SKIP_DIRS = {".godot", ".import", ".git", ".vscode", "__pycache__"}
SHOW_EXT = {
    ".gd", ".tscn", ".tres", ".gdshader", ".cfg", ".json", ".cs", ".png", ".svg",
    ".wav", ".ogg", ".mp3", ".ttf", ".glb", ".gltf",
}  # fmt: skip
ERROR_LINE = re.compile(r"(SCRIPT ERROR|ERROR|Parse Error|WARNING|USER ERROR|Invalid)", re.I)
RUN_MINUTES = 15


# --- finding Godot ------------------------------------------------------------


def _candidates() -> list[str]:
    home = Path.home()
    system = platform.system()
    found: list[str] = []
    if os.environ.get("GODOT"):
        found.append(os.environ["GODOT"])
    for name in ("godot", "godot4", "Godot", "godot-mono"):
        if path := shutil.which(name):
            found.append(path)
    if system == "Windows":
        roots = [home / "Downloads", home / "Desktop", home / "Documents", home,
                 home / "OneDrive" / "Desktop", Path(r"C:\Godot"), Path(r"C:\Program Files"),
                 Path(r"C:\Program Files (x86)\Steam\steamapps\common\Godot Engine")]  # fmt: skip
        for root in roots:
            for pattern in ("Godot*.exe", "Godot*/Godot*.exe", "*/Godot*.exe"):
                found += sorted(glob.glob(str(root / pattern)), reverse=True)
        found += glob.glob(str(home / r"AppData\Local\Microsoft\WinGet\Packages\*Godot*\*.exe"))
    elif system == "Darwin":
        found += ["/Applications/Godot.app/Contents/MacOS/Godot",
                  str(home / "Applications/Godot.app/Contents/MacOS/Godot")]  # fmt: skip
    else:
        found += sorted(glob.glob(str(home / "*/Godot_v4*")), reverse=True)
    return found


def find_godot(configured: str = "") -> str:
    if configured:
        if Path(configured).expanduser().is_file():
            return str(Path(configured).expanduser())
        raise ToolError(f"godot_path in the settings points to {configured}, which doesn't exist")
    for c in _candidates():
        p = Path(c)
        if not p.is_file() or "console" in p.name.lower():
            continue
        if "_v3" in p.name or "Godot_v3" in str(p):
            continue
        return str(p)
    raise ToolError(
        "Godot wasn't found. Download Godot 4 from https://godotengine.org/download, then put "
        "its full path in the settings file as godot_path."
    )


def _console_variant(exe: str) -> str:
    """On Windows, the *_console.exe build next to Godot prints output reliably."""
    p = Path(exe)
    if platform.system() == "Windows" and "console" not in p.stem.lower():
        console = p.with_name(p.stem + "_console" + p.suffix)
        if console.exists():
            return str(console)
    return exe


# --- projects -----------------------------------------------------------------


def _project_dir(ctx: ToolContext, path: str) -> Path:
    p = ctx.resolve(path)
    if p.is_file() and p.name == "project.godot":
        p = p.parent
    if not (p / "project.godot").exists():
        nearby = _find_projects(p) if p.is_dir() else []
        hint = (
            " Godot projects found inside it: " + ", ".join(str(n) for n in nearby[:10])
            if nearby
            else ""
        )
        raise ToolError(f"{path} isn't a Godot project (no project.godot).{hint}")
    return p


def _find_projects(root: Path, depth: int = 3) -> list[Path]:
    found = []
    try:
        for entry in root.iterdir():
            if not entry.is_dir() or entry.name.startswith(".") or entry.name in SKIP_DIRS:
                continue
            if (entry / "project.godot").exists():
                found.append(entry)
            elif depth > 1:
                found += _find_projects(entry, depth - 1)
    except OSError:
        pass
    return found


def _read_settings(project: Path) -> dict[str, dict[str, str]]:
    sections: dict[str, dict[str, str]] = {}
    current = sections.setdefault("", {})
    for line in (
        (project / "project.godot").read_text(encoding="utf-8", errors="replace").splitlines()
    ):
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = sections.setdefault(line[1:-1], {})
        elif "=" in line:
            key, _, value = line.partition("=")
            current[key.strip()] = value.strip()
    return sections


def _tree(project: Path, limit: int = 300) -> list[str]:
    lines: list[str] = []
    for root, dirs, files in os.walk(project):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("."))
        rel = Path(root).relative_to(project)
        for f in sorted(files):
            if Path(f).suffix.lower() in SHOW_EXT:
                lines.append("res://" + (rel / f).as_posix().lstrip("./"))
                if len(lines) >= limit:
                    return lines + ["..."]
    return lines


def godot_project(ctx: ToolContext, path: str = ".") -> str:
    p = ctx.resolve(path)
    if p.is_dir() and not (p / "project.godot").exists():
        projects = _find_projects(p)
        if not projects:
            return f"No Godot projects found in {p}."
        return "Godot projects found:\n" + "\n".join(str(x) for x in projects)
    project = _project_dir(ctx, path)
    s = _read_settings(project)
    app = s.get("application", {})
    lines = [
        f"Project folder: {project}",
        f"Name: {app.get('config/name', '?')}",
        f"Main scene: {app.get('run/main_scene', '(not set)')}",
        f"Godot features: {app.get('config/features', '?')}",
    ]
    if s.get("autoload"):
        lines.append("Autoloads: " + ", ".join(f"{k}={v}" for k, v in s["autoload"].items()))
    actions = [k for k in s.get("input", {})]
    if actions:
        lines.append("Input actions: " + ", ".join(actions))
    try:
        lines.append(f"Godot program: {find_godot(ctx.godot_path)}")
    except ToolError as exc:
        lines.append(f"Godot program: not found ({exc})")
    lines.append("\nFiles:")
    lines += _tree(project) or ["(no scripts or scenes yet)"]
    return "\n".join(lines)


PROJECT_FILE = """\
; Engine configuration file.
config_version=5

[application]

config/name="{name}"
{main}config/features=PackedStringArray("4.3")

[display]

window/size/viewport_width=1280
window/size/viewport_height=720
window/stretch/mode="canvas_items"

[rendering]

renderer/rendering_method="{renderer}"
"""


def godot_new_project(
    ctx: ToolContext, path: str, name: str, main_scene: str = "", renderer: str = "forward_plus"
) -> str:
    project = ctx.resolve(path)
    if (project / "project.godot").exists():
        raise ToolError(f"There is already a Godot project in {project}")
    if renderer not in {"forward_plus", "mobile", "gl_compatibility"}:
        raise ToolError("renderer must be forward_plus, mobile or gl_compatibility")
    ctx.allow(f"Create a new Godot project '{name}' in {project}")
    project.mkdir(parents=True, exist_ok=True)
    main = f'run/main_scene="{main_scene}"\n' if main_scene else ""
    text = PROJECT_FILE.format(name=name.replace('"', "'"), main=main, renderer=renderer)
    (project / "project.godot").write_text(text, encoding="utf-8")
    return (
        f"Created {project / 'project.godot'}. Now write the scripts (.gd) and scenes (.tscn) "
        "with write_file, then check them with godot_run."
    )


def godot_set_main_scene(ctx: ToolContext, path: str, scene: str) -> str:
    project = _project_dir(ctx, path)
    if not scene.startswith("res://"):
        scene = "res://" + scene.lstrip("/")
    ctx.allow(f"Set the main scene of {project.name} to {scene}")
    file = project / "project.godot"
    text = file.read_text(encoding="utf-8")
    line = f'run/main_scene="{scene}"'
    if "run/main_scene=" in text:
        text = re.sub(r"run/main_scene=.*", line, text)
    elif "[application]" in text:
        text = text.replace("[application]\n", f"[application]\n\n{line}\n", 1)
    else:
        text += f"\n[application]\n\n{line}\n"
    file.write_text(text, encoding="utf-8")
    return f"Main scene is now {scene}"


def _summarize(output: str, code: int | None, label: str) -> str:
    lines = output.splitlines()
    problems = []
    for i, line in enumerate(lines):
        if ERROR_LINE.search(line):
            # Godot prints the file and line on the next line ("at: ..." / "   at: res://...")
            nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
            problems.append(line.strip() + (f"  {nxt}" if nxt.startswith("at:") else ""))
    head = f"{label}: exit code {code}" if code is not None else label
    if problems:
        unique = list(dict.fromkeys(problems))
        head += f"\n{len(unique)} problem(s) found:\n" + "\n".join(unique[:40])
    else:
        head += "\nNo errors or warnings were printed."
    tail = "\n".join(lines[-60:])
    return f"{head}\n\n--- last output ---\n{tail}" if tail else head


def godot_run(ctx: ToolContext, path: str = ".", seconds: float = 5, scene: str = "") -> str:
    project = _project_dir(ctx, path)
    exe = _console_variant(find_godot(ctx.godot_path))
    ctx.allow_for(
        "godot",
        RUN_MINUTES,
        f"Run Godot projects in the background (no window) to test them, for the next "
        f"{RUN_MINUTES} minutes. First up: {project}",
    )
    seconds = max(1.0, min(float(seconds), 60.0))
    notes = ""
    if not (project / ".godot").exists():
        # A fresh project has to import its files once before it can run.
        imp = subprocess.run(
            [exe, "--headless", "--path", str(project), "--editor", "--quit"],
            capture_output=True, text=True, timeout=300, errors="replace",
        )  # fmt: skip
        notes = "(Imported the project first.)\n"
        if imp.returncode not in (0, None):
            notes += _summarize(imp.stdout + imp.stderr, imp.returncode, "Import") + "\n\n"
    cmd = [exe, "--headless", "--path", str(project)]
    if scene:
        cmd.append(scene if scene.startswith("res://") else "res://" + scene.lstrip("/"))
    cmd += ["--quit-after", str(int(seconds * 60))]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=seconds + 120, errors="replace"
        )
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or "") + (exc.stderr or "")
        if isinstance(out, bytes):
            out = out.decode(errors="replace")
        return notes + _summarize(out, None, "The game was still running and was stopped")
    return notes + _summarize(proc.stdout + proc.stderr, proc.returncode, "Test run finished")


def godot_play(ctx: ToolContext, path: str = ".", scene: str = "") -> str:
    project = _project_dir(ctx, path)
    exe = find_godot(ctx.godot_path)
    ctx.allow_for(
        "godot",
        RUN_MINUTES,
        f"Run Godot projects to test them, for the next {RUN_MINUTES} minutes. First up: {project}",
    )
    cmd = [exe, "--path", str(project)]
    if scene:
        cmd.append(scene if scene.startswith("res://") else "res://" + scene.lstrip("/"))
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return "The game window is opening. Use look_at_screen to see it."


def godot_editor(ctx: ToolContext, path: str = ".") -> str:
    project = _project_dir(ctx, path)
    exe = find_godot(ctx.godot_path)
    subprocess.Popen(
        [exe, "--editor", "--path", str(project)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return "The Godot editor is opening (it can take a few seconds)."


_PATH = {"type": "string", "description": "Project folder (the one with project.godot)"}
_SCENE = {"type": "string", "description": "Optional scene, e.g. res://levels/level_1.tscn"}

TOOLS = [
    Tool(
        "godot_project",
        "Show a Godot project's name, main scene, input actions and files. Given a folder "
        "that isn't a project, lists the Godot projects inside it.",
        {"path": _PATH},
        godot_project,
    ),
    Tool(
        "godot_new_project",
        "Create a new, empty Godot 4 project folder.",
        {
            "path": {"type": "string", "description": "New folder for the project"},
            "name": {"type": "string", "description": "Game name"},
            "main_scene": {"type": "string", "description": "e.g. res://main.tscn (optional)"},
            "renderer": {
                "type": "string",
                "enum": ["forward_plus", "mobile", "gl_compatibility"],
                "description": "gl_compatibility suits older computers and web games",
            },
        },
        godot_new_project,
        ["path", "name"],
    ),
    Tool(
        "godot_set_main_scene",
        "Set which scene the game starts with.",
        {"path": _PATH, "scene": {"type": "string", "description": "e.g. res://main.tscn"}},
        godot_set_main_scene,
        ["path", "scene"],
    ),
    Tool(
        "godot_run",
        "Run the game in the background (no window) for a few seconds and return any "
        "errors and printed output. Use it after every change to find and fix mistakes.",
        {
            "path": _PATH,
            "seconds": {"type": "number", "description": "How long to run, default 5"},
            "scene": _SCENE,
        },
        godot_run,
    ),
    Tool(
        "godot_play",
        "Open the game in a normal window so the user (and look_at_screen) can see it.",
        {"path": _PATH, "scene": _SCENE},
        godot_play,
    ),
    Tool("godot_editor", "Open the project in the Godot editor.", {"path": _PATH}, godot_editor),
]

"""Settings, loaded from ``~/.arize/config.toml`` and ``ARIZE_*`` environment variables.

Environment variables win over the file, and the file wins over the defaults.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

DEFAULT_CONFIG = """\
# ARIZE settings. Delete a line to fall back to the default.

# Any server that speaks the OpenAI chat API works. Ollama's is below;
# LM Studio is usually http://localhost:1234/v1
base_url = "http://localhost:11434/v1"

# Pick a model that supports tool calling, e.g. qwen3, llama3.1, mistral-nemo
model = "qwen3:8b"

# Only needed for hosted services; local servers ignore it.
api_key = ""

# How many tool steps ARIZE may take for a single message.
max_steps = 30

# Ask before writing files or running commands. Set to false at your own risk.
confirm_actions = true

# Folder ARIZE may read and write files in ("~" = your home folder).
workspace = "~"

# Let models like qwen3 "think" before answering: smarter but much slower.
think = false

# Model used to look at your screen. Leave empty to use the main model; it must be
# able to see images, e.g. qwen3-vl:8b
vision_model = ""

# Full path to the Godot 4 program. Leave empty to search the usual places.
godot_path = ""

# After you allow screen control, how many minutes before ARIZE has to ask again.
control_minutes = 10
"""


def home_dir() -> Path:
    return Path(os.environ.get("ARIZE_HOME", "~/.arize")).expanduser()


@dataclass
class Config:
    base_url: str = "http://localhost:11434/v1"
    model: str = "qwen3:8b"
    api_key: str = ""
    max_steps: int = 30
    confirm_actions: bool = True
    workspace: str = "~"
    think: bool = False
    vision_model: str = ""
    godot_path: str = ""
    control_minutes: float = 10.0
    timeout: float = 300.0
    home: Path = field(default_factory=home_dir)

    @property
    def memory_path(self) -> Path:
        return self.home / "memory.db"

    @property
    def workspace_path(self) -> Path:
        return Path(self.workspace).expanduser().resolve()

    @classmethod
    def load(cls, path: Path | None = None) -> Config:
        cfg = cls()
        path = path or cfg.home / "config.toml"
        values: dict = {}
        if path.exists():
            values.update(tomllib.loads(path.read_text(encoding="utf-8")))
        for f in fields(cls):
            env = os.environ.get(f"ARIZE_{f.name.upper()}")
            if env is not None:
                values[f.name] = env
        for f in fields(cls):
            if f.name in values and f.name != "home":
                setattr(cfg, f.name, _coerce(values[f.name], type(getattr(cfg, f.name))))
        return cfg


def _coerce(value, kind):
    if kind is bool and isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return kind(value)


def write_default_config(path: Path | None = None, force: bool = False) -> Path:
    path = path or home_dir() / "config.toml"
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(DEFAULT_CONFIG, encoding="utf-8")
    return path

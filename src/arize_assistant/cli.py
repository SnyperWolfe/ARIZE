"""The ``arize`` command."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .agent import Agent
from .config import Config, write_default_config
from .llm import LLMError, OpenAICompatibleModel
from .memory import MemoryStore
from .tools import ALL_TOOLS, ToolContext, default_registry

DIM, BOLD, CYAN, YELLOW, RESET = "\033[2m", "\033[1m", "\033[36m", "\033[33m", "\033[0m"
if not sys.stdout.isatty():
    DIM = BOLD = CYAN = YELLOW = RESET = ""

HELP = """Commands:
  /new        start a fresh conversation (memories are kept)
  /memories   list what ARIZE remembers
  /forget N   delete memory #N
  /help       show this help
  /quit       leave (or press Ctrl+D)"""


def ask_user(description: str) -> bool:
    print(f"\n{YELLOW}ARIZE wants to:{RESET} {description}")
    try:
        return input(f"{YELLOW}Allow? [y/N]{RESET} ").strip().lower() in {"y", "yes"}
    except EOFError:
        return False


def show_event(kind: str, name: str, text: str) -> None:
    if kind == "tool":
        try:
            args = json.loads(text)
            shown = ", ".join(f"{k}={v!r}" for k, v in args.items())
        except ValueError:
            shown = text
        if len(shown) > 120:
            shown = shown[:117] + "..."
        print(f"{DIM}  ↳ {name}({shown}){RESET}")
    elif text.startswith("Error:"):
        print(f"{DIM}    {text.splitlines()[0][:150]}{RESET}")


def build_agent(cfg: Config, yes: bool = False) -> Agent:
    memory = MemoryStore(cfg.memory_path)
    ctx = ToolContext(
        workspace=cfg.workspace_path,
        memory=memory,
        confirm=ask_user,
        confirm_actions=cfg.confirm_actions and not yes,
    )
    model = OpenAICompatibleModel(cfg.base_url, cfg.model, cfg.api_key, cfg.timeout)
    return Agent(model, default_registry(ctx), memory, cfg.max_steps, on_event=show_event)


def print_memories(memory: MemoryStore) -> None:
    items = memory.all()
    if not items:
        print("ARIZE doesn't remember anything yet.")
    for m in items:
        print(f"  #{m.id}  {m.text}  {DIM}({m.created[:10]}){RESET}")


def chat(agent: Agent, cfg: Config) -> int:
    print(f"{BOLD}{CYAN}ARIZE{RESET} {DIM}v{__version__} · {cfg.model} · /help for commands{RESET}")
    while True:
        try:
            line = input(f"\n{BOLD}you ›{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line.startswith("/"):
            cmd, _, arg = line.partition(" ")
            if cmd in {"/quit", "/exit"}:
                return 0
            elif cmd == "/new":
                agent.reset()
                print("Started a new conversation.")
            elif cmd == "/memories":
                print_memories(agent.memory)
            elif cmd == "/forget" and arg.strip().isdigit():
                ok = agent.memory.forget(int(arg))
                print("Forgotten." if ok else f"There is no memory #{arg.strip()}.")
            else:
                print(HELP)
            continue
        try:
            answer = agent.ask(line)
        except LLMError as exc:
            agent.history.pop()  # drop the unanswered message so you can retry
            print(f"{YELLOW}{exc}{RESET}")
            continue
        except KeyboardInterrupt:
            print(f"\n{DIM}(stopped){RESET}")
            continue
        print(f"\n{BOLD}{CYAN}ARIZE ›{RESET} {answer}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="arize", description="ARIZE — your private AI assistant with tools and memory."
    )
    parser.add_argument("--version", action="version", version=f"arize {__version__}")
    parser.add_argument("--model", help="model name to use for this run")
    parser.add_argument(
        "--yes", action="store_true", help="don't ask before writing files or running commands"
    )
    sub = parser.add_subparsers(dest="command")
    p_ask = sub.add_parser("ask", help="ask one question and exit")
    p_ask.add_argument("question", nargs="+")
    sub.add_parser("init", help="create ~/.arize/config.toml")
    p_mem = sub.add_parser("memory", help="see or change what ARIZE remembers")
    p_mem.add_argument(
        "action", choices=["list", "add", "forget", "clear"], nargs="?", default="list"
    )
    p_mem.add_argument("value", nargs="*")
    sub.add_parser("tools", help="list the tools ARIZE can use")
    args = parser.parse_args(argv)

    if args.command == "init":
        path = write_default_config(force=False)
        print(f"Settings file: {path}")
        return 0

    cfg = Config.load()
    if args.model:
        cfg.model = args.model

    if args.command == "tools":
        for t in ALL_TOOLS:
            print(f"  {BOLD}{t.name}{RESET}  {t.description}")
        return 0

    if args.command == "memory":
        memory = MemoryStore(cfg.memory_path)
        text = " ".join(args.value)
        if args.action == "add":
            if not text:
                parser.error("memory add needs some text")
            print(f"Saved #{memory.add(text).id}")
        elif args.action == "forget":
            if not text.isdigit():
                parser.error("memory forget needs a number, e.g. arize memory forget 3")
            print("Forgotten." if memory.forget(int(text)) else f"There is no memory #{text}.")
        elif args.action == "clear":
            for m in memory.all():
                memory.forget(m.id)
            print("All memories deleted.")
        else:
            print_memories(memory)
        return 0

    write_default_config()
    agent = build_agent(cfg, yes=args.yes)
    if args.command == "ask":
        try:
            print(agent.ask(" ".join(args.question)))
        except LLMError as exc:
            print(exc, file=sys.stderr)
            return 1
        return 0
    return chat(agent, cfg)


if __name__ == "__main__":
    sys.exit(main())

import json

import httpx
import pytest

from arize_assistant import cli
from arize_assistant.agent import Agent
from arize_assistant.config import Config
from arize_assistant.llm import OpenAICompatibleModel, Reply, ToolCall
from arize_assistant.memory import MemoryStore
from arize_assistant.tools import ToolContext, default_registry
from arize_assistant.tools.web import html_to_text, parse_search_results


class ScriptedModel:
    """Returns pre-written replies in order and records what it was sent."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages, tools, on_token=None):
        self.calls.append((messages, tools))
        reply = self.replies.pop(0)
        if on_token and reply.content:
            for i in range(0, len(reply.content), 3):  # stream in small pieces
                on_token(reply.content[i : i + 3])
        return reply


def call(name, **args):
    return ToolCall(id=f"c_{name}", name=name, arguments=json.dumps(args))


@pytest.fixture
def memory():
    m = MemoryStore(":memory:")
    yield m
    m.close()


@pytest.fixture
def make_ctx(tmp_path, memory):
    def make(answer=True, confirm_actions=True):
        asked = []

        def confirm(desc):
            asked.append(desc)
            return answer

        ctx = ToolContext(tmp_path.resolve(), memory, confirm, confirm_actions)
        ctx.asked = asked
        return ctx

    return make


# --- memory ---------------------------------------------------------------


def test_memory_add_search_forget(memory):
    a = memory.add("The user's dog is named Pixel.")
    memory.add("The user prefers metric units.")
    assert memory.add("the user's dog is named pixel.").id == a.id  # no duplicates
    assert [m.text for m in memory.search("what is my dog called")] == [a.text]
    assert memory.forget(a.id)
    assert not memory.forget(a.id)
    assert memory.search("dog") == []
    assert len(memory.all()) == 1


def test_memory_context_mixes_relevant_and_recent(memory):
    for i in range(10):
        memory.add(f"Filler fact number {i}.")
    old = memory.add("The user lives in Lisbon.")
    for i in range(10, 20):
        memory.add(f"Filler fact number {i}.")
    picked = memory.context_for("weather in lisbon", limit=4)
    assert old in picked and len(picked) == 4


def test_memory_persists_on_disk(tmp_path):
    path = tmp_path / "sub" / "memory.db"
    MemoryStore(path).add("Persisted fact.")
    assert [m.text for m in MemoryStore(path).all()] == ["Persisted fact."]


# --- tools ----------------------------------------------------------------


def test_file_tools_and_workspace_guard(make_ctx, tmp_path):
    reg = default_registry(make_ctx())
    assert "Wrote" in reg.call("write_file", json.dumps({"path": "a/b.txt", "content": "hi"}))
    assert (tmp_path / "a" / "b.txt").read_text() == "hi"
    assert reg.call("read_file", json.dumps({"path": "a/b.txt"})) == "hi"
    assert "a/" in reg.call("list_folder", "{}")
    assert "outside the allowed folder" in reg.call(
        "read_file", json.dumps({"path": "/etc/passwd"})
    )
    assert "outside" in reg.call("read_file", json.dumps({"path": "../x"}))


def test_risky_actions_need_approval(make_ctx, tmp_path):
    ctx = make_ctx(answer=False)
    reg = default_registry(ctx)
    out = reg.call("run_command", json.dumps({"command": "touch made"}))
    assert "declined" in out and not (tmp_path / "made").exists()
    assert "touch made" in ctx.asked[0]
    out = reg.call("write_file", json.dumps({"path": "f.txt", "content": "x"}))
    assert "declined" in out and not (tmp_path / "f.txt").exists()


def test_run_command_when_allowed(make_ctx):
    reg = default_registry(make_ctx(confirm_actions=False))
    out = reg.call("run_command", json.dumps({"command": "echo hello"}))
    assert "exit code: 0" in out and "hello" in out


def test_registry_reports_bad_calls(make_ctx):
    reg = default_registry(make_ctx())
    assert "no tool named" in reg.call("nope", "{}")
    assert "invalid arguments" in reg.call("read_file", "{not json")
    assert "needs path" in reg.call("read_file", "{}")
    assert "bad arguments" in reg.call("read_file", json.dumps({"path": "x", "bogus": 1}))


def test_memory_tools(make_ctx, memory):
    reg = default_registry(make_ctx())
    assert "Saved memory #1" in reg.call("remember", json.dumps({"fact": "Likes tea."}))
    assert "#1: Likes tea." in reg.call("search_memory", json.dumps({"query": "tea"}))
    assert "Forgot" in reg.call("forget", json.dumps({"memory_id": 1}))
    assert "no memory #1" in reg.call("forget", json.dumps({"memory_id": 1}))


def test_search_result_parsing():
    page = (
        '<a rel="nofollow" class="result__a" '
        'href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage&rut=x">Example '
        "<b>Page</b></a>"
        '<a class="result__snippet" href="#">A &amp; B snippet</a>'
    )
    assert parse_search_results(page, 5) == [
        {"title": "Example Page", "url": "https://example.com/page", "snippet": "A & B snippet"}
    ]


def test_html_to_text():
    title, text = html_to_text(
        "<html><head><title>T</title><script>x=1</script></head>"
        "<body><nav>menu</nav><h1>Head</h1><p>One  two</p></body></html>"
    )
    assert title == "T" and text == "Head\nOne two"


# --- agent ----------------------------------------------------------------


def test_agent_runs_tools_then_answers(make_ctx, memory):
    model = ScriptedModel(
        [
            Reply("", [call("remember", fact="The user's name is Sam.")]),
            Reply("<think>noted</think>Nice to meet you, Sam!"),
        ]
    )
    events = []
    agent = Agent(
        model, default_registry(make_ctx()), memory, on_event=lambda *e: events.append(e[:2])
    )
    assert agent.ask("Hi, I'm Sam") == "Nice to meet you, Sam!"
    assert [m.text for m in memory.all()] == ["The user's name is Sam."]
    assert events == [("tool", "remember"), ("result", "remember")]
    tool_msg = model.calls[1][0][-1]
    assert tool_msg["role"] == "tool" and "Saved memory" in tool_msg["content"]

    # A new conversation sees the memory in its system prompt.
    agent.reset()
    model.replies.append(Reply("Your name is Sam."))
    agent.ask("What's my name?")
    assert "The user's name is Sam." in model.calls[-1][0][0]["content"]


def test_agent_stops_after_max_steps(make_ctx, memory):
    loop = [Reply("", [call("list_folder")]) for _ in range(3)]
    model = ScriptedModel([*loop, Reply("Here's what I found.")])
    agent = Agent(model, default_registry(make_ctx()), memory, max_steps=3)
    assert agent.ask("loop forever") == "Here's what I found."
    assert model.calls[-1][1] == []  # final call is made without tools


# --- model client ---------------------------------------------------------


def test_openai_client_parses_tool_calls(monkeypatch):
    def handler(request):
        body = json.loads(request.content)
        assert body["model"] == "m" and body["tools"]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "x",
                                    "type": "function",
                                    "function": {"name": "web_search", "arguments": {"query": "q"}},
                                }
                            ],
                        }
                    }
                ]
            },
        )

    client = OpenAICompatibleModel("http://test/v1", "m")
    client._http = httpx.Client(base_url="http://test/v1", transport=httpx.MockTransport(handler))
    reply = client.chat([{"role": "user", "content": "hi"}], [{"type": "function"}])
    assert reply.content == ""
    assert reply.tool_calls[0].name == "web_search"
    assert json.loads(reply.tool_calls[0].arguments) == {"query": "q"}


# --- config & cli ---------------------------------------------------------


def test_config_file_and_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ARIZE_HOME", str(tmp_path))
    (tmp_path / "config.toml").write_text('model = "llama3.1"\nmax_steps = 3\n')
    monkeypatch.setenv("ARIZE_CONFIRM_ACTIONS", "false")
    cfg = Config.load()
    assert (cfg.model, cfg.max_steps, cfg.confirm_actions) == ("llama3.1", 3, False)
    assert cfg.memory_path == tmp_path / "memory.db"


def test_cli_memory_commands(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ARIZE_HOME", str(tmp_path))
    assert cli.main(["memory", "add", "Likes", "jazz."]) == 0
    assert cli.main(["memory"]) == 0
    assert "#1  Likes jazz." in capsys.readouterr().out
    cli.main(["memory", "forget", "1"])
    cli.main(["memory", "list"])
    assert "doesn't remember anything" in capsys.readouterr().out


# --- streaming, thinking, and the app server -------------------------------


def test_stream_parsing_assembles_text_and_tool_calls():
    from arize_assistant.llm import _read_stream

    def sse(delta):
        return "data: " + json.dumps({"choices": [{"delta": delta}]})

    lines = [
        sse({"content": "Hel"}),
        sse({"content": "lo"}),
        sse(
            {
                "tool_calls": [
                    {"index": 0, "id": "t1", "function": {"name": "web_", "arguments": ""}}
                ]
            }
        ),
        sse(
            {"tool_calls": [{"index": 0, "function": {"name": "search", "arguments": '{"query":'}}]}
        ),
        sse({"tool_calls": [{"index": 0, "function": {"arguments": ' "cats"}'}}]}),
        "",
        "data: [DONE]",
    ]
    seen = []
    reply = _read_stream(iter(lines), seen.append)
    assert seen == ["Hel", "lo"] and reply.content == "Hello"
    assert reply.tool_calls[0].name == "web_search"
    assert json.loads(reply.tool_calls[0].arguments) == {"query": "cats"}


def test_think_filter_hides_thinking_even_when_split():
    from arize_assistant.agent import ThinkFilter

    out = []
    f = ThinkFilter(out.append)
    for piece in ["<thi", "nk>secret ", "plans</th", "ink>\n\nHi ", "there <", "b>"]:
        f(piece)
    assert "".join(out) == "Hi there <b>"


def test_no_think_switch(make_ctx, memory):
    model = ScriptedModel([Reply("a"), Reply("b")])
    agent = Agent(model, default_registry(make_ctx()), memory)
    agent.ask("hi")
    assert model.calls[-1][0][0]["content"].endswith("/no_think")
    agent.think = True
    agent.ask("hi")
    assert "/no_think" not in model.calls[-1][0][0]["content"]


def test_agent_streams_answer(make_ctx, memory):
    model = ScriptedModel([Reply("<think>hmm</think>Streaming works.")])
    agent = Agent(model, default_registry(make_ctx()), memory)
    pieces = []
    assert agent.ask("go", on_token=pieces.append) == "Streaming works."
    assert "".join(pieces) == "Streaming works."


@pytest.fixture
def app_server(tmp_path, monkeypatch):
    import threading

    from arize_assistant.server import ArizeApp, make_server

    monkeypatch.setenv("ARIZE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ARIZE_WORKSPACE", str(tmp_path))
    model = ScriptedModel([])
    app = ArizeApp(Config.load(), model=model)
    server = make_server(app, 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    client = httpx.Client(base_url=base, headers={"X-Arize-Token": app.token}, timeout=10)
    yield app, model, client, base
    server.shutdown()


def test_server_page_and_security(app_server):
    app, _, client, base = app_server
    page = client.get("/")
    assert page.status_code == 200 and app.token in page.text
    assert client.get("/icon.png").headers["content-type"] == "image/png"
    assert client.get("/api/info").json()["model"]
    # No token, wrong token, or a foreign Host header are all refused.
    assert httpx.get(base + "/api/memories").status_code == 403
    assert (
        httpx.post(
            base + "/api/chat", json={"message": "hi"}, headers={"X-Arize-Token": "nope"}
        ).status_code
        == 403
    )
    assert client.get("/", headers={"Host": "evil.example:80"}).status_code == 403


def test_server_chat_with_permission_prompt(app_server, tmp_path):
    app, model, client, _ = app_server
    model.replies += [
        Reply("", [call("write_file", path="hello.txt", content="hi")]),
        Reply("Saved it."),
    ]
    events = []
    with client.stream("POST", "/api/chat", json={"message": "save a file"}) as resp:
        for line in resp.iter_lines():
            if not line:
                continue
            ev = json.loads(line)
            events.append(ev)
            if ev["type"] == "confirm":
                assert "hello.txt" in ev["text"]
                r = client.post("/api/confirm", json={"id": ev["id"], "allow": True})
                assert r.json()["ok"]
    types = [e["type"] for e in events]
    assert types[:3] == ["tool", "confirm", "result"] and types[-1] == "done"
    assert events[-1]["text"] == "Saved it."
    assert "".join(e["text"] for e in events if e["type"] == "token") == "Saved it."
    assert (tmp_path / "hello.txt").read_text() == "hi"


def test_server_memories_and_new_chat(app_server):
    app, _, client, _ = app_server
    m = app.memory.add("Likes chess.")
    assert client.get("/api/memories").json()[0]["text"] == "Likes chess."
    assert client.post("/api/memories/delete", json={"id": m.id}).json()["ok"]
    assert client.get("/api/memories").json() == []
    app.agent.history.append({"role": "user", "content": "x"})
    assert client.post("/api/new").json()["ok"] and app.agent.history == []


def test_server_reports_model_errors(app_server):
    from arize_assistant.llm import LLMError

    app, model, client, _ = app_server

    def boom(*a, **k):
        raise LLMError("Can't reach the model server")

    model.chat = boom
    lines = [
        json.loads(x) for x in client.post("/api/chat", json={"message": "hi"}).text.splitlines()
    ]
    assert lines[-1] == {"type": "error", "text": "Can't reach the model server"}
    assert app.agent.history == []


def test_linux_shortcut(tmp_path, monkeypatch):
    from arize_assistant import desktop

    monkeypatch.setattr(desktop.Path, "home", lambda: tmp_path)
    (tmp_path / "Desktop").mkdir()
    made = desktop._linux_shortcuts()
    assert len(made) == 2
    text = (tmp_path / "Desktop" / "arize.desktop").read_text()
    assert "-m arize_assistant app" in text and "arize.png" in text


# --- Godot tools (with a stand-in Godot that prints Godot-style output) ------------

FAKE_GODOT = r"""#!/usr/bin/env python3
import sys, pathlib
args = sys.argv[1:]
log = pathlib.Path(__file__).with_name("calls.log")
log.open("a").write(" ".join(args) + "\n")
project = pathlib.Path(args[args.index("--path") + 1])
if "--editor" in args and "--quit" in args:
    (project / ".godot").mkdir(exist_ok=True)
    sys.exit(0)
script = (project / "main.gd").read_text()
if "undefined_thing" in script:
    print('SCRIPT ERROR: Parse Error: Identifier "undefined_thing" not declared'
          ' in the current scope.')
    print("          at: GDScript::reload (res://main.gd:5)")
    print('ERROR: Failed to load script "res://main.gd" with error "Parse error".')
    print("   at: load (modules/gdscript/gdscript.cpp:2936)")
    sys.exit(1)
print("Godot Engine v4.3.stable.official - https://godotengine.org")
print("hello from ARIZE")
"""


@pytest.fixture
def godot_ctx(make_ctx, tmp_path):
    exe = tmp_path / "bin" / "godot"
    exe.parent.mkdir()
    exe.write_text(
        FAKE_GODOT.replace("#!/usr/bin/env python3", "#!" + __import__("sys").executable)
    )
    exe.chmod(0o755)
    ctx = make_ctx()
    ctx.godot_path = str(exe)
    return ctx, default_registry(ctx), exe


def test_godot_build_test_fix_cycle(godot_ctx, tmp_path):
    ctx, reg, exe = godot_ctx

    def run(tool, **a):
        return reg.call(tool, json.dumps(a))

    assert "Created" in run(
        "godot_new_project", path="game", name="My Game", main_scene="res://main.tscn"
    )
    project_file = (tmp_path / "game" / "project.godot").read_text()
    assert 'config/name="My Game"' in project_file
    assert 'run/main_scene="res://main.tscn"' in project_file
    run("write_file", path="game/main.gd", content="extends Node2D\nvar x = undefined_thing\n")

    out = run("godot_run", path="game", seconds=1)
    assert "Imported the project first" in out
    assert "2 problem(s) found" in out
    assert 'Identifier "undefined_thing" not declared' in out and "res://main.gd:5" in out
    calls = (exe.parent / "calls.log").read_text().splitlines()
    assert calls[0].startswith("--headless --path") and calls[0].endswith("--editor --quit")
    assert calls[1].endswith("--quit-after 60")

    run("write_file", path="game/main.gd", content="extends Node2D\n")
    out = run("godot_run", path="game")
    assert "No errors or warnings" in out and "exit code 0" in out
    assert "Imported" not in out  # only the first time
    # Permission to test-run was asked once, then remembered.
    assert sum("Run Godot projects" in a for a in ctx.asked) == 1

    info = run("godot_project", path="game")
    assert 'Name: "My Game"' in info and "res://main.gd" in info and str(exe) in info
    assert "game" in run("godot_project", path=".")
    assert "isn't a Godot project" in run("godot_run", path=".")
    run("godot_set_main_scene", path="game", scene="levels/one.tscn")
    assert 'run/main_scene="res://levels/one.tscn"' in (tmp_path / "game/project.godot").read_text()


def test_godot_missing_program(make_ctx, tmp_path, monkeypatch):
    from arize_assistant.tools import godot

    monkeypatch.setattr(godot, "_candidates", lambda: [])
    ctx = make_ctx()
    reg = default_registry(ctx)
    reg.call("godot_new_project", json.dumps({"path": "g", "name": "G"}))
    out = reg.call("godot_run", json.dumps({"path": "g"}))
    assert "Godot wasn't found" in out and "godot_path" in out


# --- screen control (with a fake mouse, keyboard and screen) --------------------


class FakeDesktop:
    def __init__(self):
        self.actions = []
        self.fail = False

    def size(self):
        return (1920, 1080)

    def screenshot(self):
        from PIL import Image

        return Image.new("RGB", (1920, 1080), (20, 30, 40))

    def _do(self, *a):
        if self.fail:
            raise type("FailSafeException", (Exception,), {})("corner")
        self.actions.append(a)

    def click(self, x, y, button, clicks):
        self._do("click", x, y, button, clicks)

    def move(self, x, y):
        self._do("move", x, y)

    def drag(self, *a):
        self._do("drag", *a)

    def write(self, text):
        self._do("write", text)

    def key_down(self, k):
        self._do("down", k)

    def key_up(self, k):
        self._do("up", k)

    def scroll(self, *a):
        self._do("scroll", *a)

    def windows(self):
        return ["Godot Engine - My Game", "ARIZE"]

    def focus(self, title):
        self._do("focus", title)
        return "Godot Engine - My Game"


@pytest.fixture
def screen(make_ctx, monkeypatch):
    from arize_assistant.tools import screen as screen_mod

    fake = FakeDesktop()
    monkeypatch.setattr(screen_mod, "_desktop", fake)
    seen = {}

    def vision(png, question):
        from PIL import Image

        seen["size"] = Image.open(__import__("io").BytesIO(png)).size
        seen["question"] = question
        return "The Play button is at (500, 620)."

    ctx = make_ctx()
    ctx.vision = vision
    return ctx, default_registry(ctx), fake, seen


def test_look_and_click(screen):
    ctx, reg, fake, seen = screen
    out = reg.call("look_at_screen", json.dumps({"question": "Where is Play?"}))
    assert out == "The Play button is at (500, 620)."
    assert seen["size"] == (1280, 720)  # shrunk for the model
    assert "0-1000 grid" in seen["question"] and "Where is Play?" in seen["question"]
    assert ctx.asked == []  # looking doesn't need permission

    assert "clicked" in reg.call("click", json.dumps({"x": 500, "y": 620}))
    assert fake.actions[-1] == ("click", 960, 669, "left", 1)
    reg.call("click", json.dumps({"x": 0, "y": 1000, "clicks": 2}))
    assert fake.actions[-1] == ("click", 3, 1076, "left", 2)  # kept clear of the corner
    assert "between 0 and 1000" in reg.call("click", json.dumps({"x": 1200, "y": 5}))
    # Permission is asked once for the whole control session.
    assert len(ctx.asked) == 1 and "mouse and keyboard" in ctx.asked[0]


def test_keys_and_hold(screen):
    ctx, reg, fake, _ = screen
    reg.call("press_keys", json.dumps({"keys": "Control + S"}))
    assert fake.actions == [("down", "ctrl"), ("down", "s"), ("up", "s"), ("up", "ctrl")]
    fake.actions.clear()
    out = reg.call("hold_keys", json.dumps({"keys": "shift+w", "seconds": 0.1}))
    assert "Held shift+w" in out
    assert fake.actions == [("down", "shift"), ("down", "w"), ("up", "w"), ("up", "shift")]
    fake.actions.clear()
    reg.call("press_keys", json.dumps({"keys": "space", "times": 3}))
    assert fake.actions.count(("down", "space")) == 3
    reg.call("type_text", json.dumps({"text": "hello"}))
    assert fake.actions[-1] == ("write", "hello")
    assert "Godot Engine" in reg.call("list_windows", "{}")
    assert "Brought 'Godot Engine - My Game'" in reg.call(
        "focus_window", json.dumps({"title": "godot"})
    )


def test_corner_stop_and_denied_control(screen, make_ctx, monkeypatch):
    ctx, reg, fake, _ = screen
    reg.call("click", json.dumps({"x": 10, "y": 10}))
    fake.fail = True
    out = reg.call("press_keys", json.dumps({"keys": "w"}))
    assert "moved the mouse into a corner" in out
    fake.fail = False
    reg.call("click", json.dumps({"x": 10, "y": 10}))
    assert len(ctx.asked) == 2  # after a corner-stop, permission is asked again

    from arize_assistant.tools import screen as screen_mod

    denied = make_ctx(answer=False)
    monkeypatch.setattr(screen_mod, "_desktop", FakeDesktop())
    out = default_registry(denied).call("click", json.dumps({"x": 1, "y": 1}))
    assert "declined" in out and screen_mod._desktop.actions == []


def test_vision_errors_are_explained(screen):
    ctx, reg, _, _ = screen

    def broken(png, q):
        raise RuntimeError("model does not support images")

    ctx.vision = broken
    out = reg.call("look_at_screen", "{}")
    assert "vision_model" in out and "qwen3-vl" in out


# --- Ollama's own API and empty replies -------------------------------------------


def _ollama(handler, **kw):
    from arize_assistant.llm import OllamaModel

    m = OllamaModel("http://localhost:11434/v1", "qwen3-vl:8b", **kw)
    m._http = httpx.Client(
        base_url="http://localhost:11434", transport=httpx.MockTransport(handler)
    )
    return m


def test_ollama_sends_context_size_and_converts_history():
    sent = {}

    def handler(request):
        assert request.url.path == "/api/chat"
        sent.update(json.loads(request.content))
        return httpx.Response(200, json={
            "message": {"role": "assistant", "content": "",
                        "tool_calls": [{"function": {"name": "godot_project",
                                                     "arguments": {"path": "game"}}}]},
            "done": True, "prompt_eval_count": 1234,
        })  # fmt: skip

    m = _ollama(handler, context_length=16384)
    history = [
        {"role": "system", "content": "sys"},
        Reply("", [call("look_at_screen", question="q")]).as_message(),
        {"role": "tool", "tool_call_id": "c", "name": "look_at_screen", "content": "a window"},
    ]
    reply = m.chat(history, [{"type": "function"}])
    assert sent["options"] == {"num_ctx": 16384} and sent["think"] is False
    assert sent["stream"] is False and sent["tools"]
    assert sent["messages"][1]["tool_calls"][0]["function"]["arguments"] == {"question": "q"}
    assert sent["messages"][2]["tool_name"] == "look_at_screen"
    assert reply.tool_calls[0].name == "godot_project"
    assert json.loads(reply.tool_calls[0].arguments) == {"path": "game"}
    assert m.last_prompt_tokens == 1234


def test_ollama_streaming_and_think_fallback():
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if "think" in body:
            return httpx.Response(400, json={"error": '"llama3.2" does not support thinking'})
        lines = [
            {"message": {"content": "Hel"}, "done": False},
            {"message": {"content": "lo"}, "done": False},
            {"message": {"content": "", "tool_calls": [
                {"function": {"name": "wait", "arguments": {"seconds": 1}}}]}, "done": False},
            {"message": {"content": ""}, "done": True, "prompt_eval_count": 50},
        ]  # fmt: skip
        return httpx.Response(200, text="\n".join(json.dumps(x) for x in lines))

    m = _ollama(handler)
    pieces = []
    reply = m.chat([{"role": "user", "content": "hi"}], [], pieces.append)
    assert len(bodies) == 2 and "think" not in bodies[1]  # retried without think
    assert pieces == ["Hel", "lo"] and reply.content == "Hello"
    assert reply.tool_calls[0].name == "wait"
    m.chat([{"role": "user", "content": "again"}], [], pieces.append)
    assert len(bodies) == 3  # remembers not to send think


def test_ollama_look_sends_image_and_missing_model_hint():
    def handler(request):
        body = json.loads(request.content)
        if body["model"] == "missing":
            return httpx.Response(404, json={"error": "model 'missing' not found"})
        assert body["model"] == "qwen3-vl:4b" and body["messages"][0]["images"]
        return httpx.Response(200, json={"message": {"content": "A button at (10, 20)"}})

    m = _ollama(handler)
    assert m.look(b"png", "where?", "qwen3-vl:4b") == "A button at (10, 20)"
    from arize_assistant.llm import LLMError

    with pytest.raises(LLMError, match="ollama pull"):
        m.look(b"png", "where?", "missing")


def test_make_model_picks_ollama_for_ollama_urls():
    from arize_assistant.llm import OllamaModel, OpenAICompatibleModel, make_model

    cfg = Config()
    assert isinstance(make_model(cfg), OllamaModel)
    cfg.base_url = "http://localhost:1234/v1"
    assert isinstance(make_model(cfg), OpenAICompatibleModel)
    cfg.provider = "ollama"
    assert isinstance(make_model(cfg), OllamaModel)


def test_empty_reply_is_retried_then_explained(make_ctx, memory):
    from arize_assistant.agent import EMPTY_ANSWER

    model = ScriptedModel([Reply("", [call("list_folder")]), Reply(""), Reply("I opened it.")])
    agent = Agent(model, default_registry(make_ctx()), memory)
    assert agent.ask("open godot") == "I opened it."
    assert "empty" in model.calls[-1][0][-1]["content"]  # the nudge
    assert model.calls[-1][1] == []  # without tools
    assert agent.history[-1] == {"role": "assistant", "content": "I opened it."}
    assert all("last reply was empty" not in str(m.get("content")) for m in agent.history)

    model.replies += [Reply(""), Reply("<think>...</think>")]
    assert agent.ask("and now?") == EMPTY_ANSWER


def test_no_think_hint_only_when_needed(make_ctx, memory):
    model = ScriptedModel([Reply("ok")])
    model.soft_think_switch = False  # like OllamaModel
    agent = Agent(model, default_registry(make_ctx()), memory)
    agent.ask("hi")
    assert "/no_think" not in model.calls[-1][0][0]["content"]

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

    def chat(self, messages, tools):
        self.calls.append((messages, tools))
        return self.replies.pop(0)


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

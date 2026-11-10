# ARIZE

**A private AI assistant that runs on your own computer.** ARIZE can search the web,
read pages, work with your files, run commands, and remember things about you between
conversations. Your chats and memories stay on your own computer.

```
you › I'm Sam, and I'm planning a trip to Lisbon in May.
  ↳ remember(fact="The user's name is Sam.")
  ↳ remember(fact='Sam is planning a trip to Lisbon in May.')
ARIZE › Nice to meet you, Sam! May is a lovely time for Lisbon...

(next day)
you › what's the weather usually like where I'm going?
  ↳ web_search(query='Lisbon weather in May')
ARIZE › In May Lisbon is usually around 17–23 °C and mostly dry...
```

## Install

1. Install [Ollama](https://ollama.com) and download a model that can use tools:

   ```bash
   ollama pull qwen3:8b
   ```

2. Install ARIZE (needs Python 3.10+). From this folder:

   ```bash
   pip install .
   ```

3. Start chatting:

   ```bash
   arize
   ```

## Using it

| Command | What it does |
|---|---|
| `arize` | Chat. Type `/help` inside for chat commands (`/new`, `/memories`, `/forget N`, `/quit`). |
| `arize ask "question"` | Ask one question and print the answer. |
| `arize memory` | List what ARIZE remembers. |
| `arize memory add "I'm vegetarian"` | Teach it something directly. |
| `arize memory forget 3` | Delete memory #3. `arize memory clear` deletes all. |
| `arize tools` | List the tools ARIZE can use. |
| `arize --model llama3.1` | Use a different model for this run. |
| `arize init` | Create the settings file. |

## Tools

| Tool | What it does | Asks you first? |
|---|---|---|
| `web_search` | Searches the web with DuckDuckGo (no account or key needed) | No |
| `read_webpage` | Opens a link and reads its text | No |
| `read_file`, `list_folder` | Looks at files in your workspace folder | No |
| `write_file` | Creates or changes a file | **Yes** |
| `run_command` | Runs a terminal command | **Yes** |
| `remember`, `search_memory`, `forget` | Manages its long-term memory | No |

ARIZE can only touch files inside your **workspace** folder (your home folder by
default; change it in settings).

## Memory

Memories are short facts saved in `~/.arize/memory.db`, a local SQLite
database. ARIZE saves things you tell it on its own (your name, preferences, plans).
Before every reply it looks up the memories that match your message, plus the most
recent ones, and gives them to the model. To see or change memories, use
`/memories` in chat or the `arize memory` commands.

## Settings

`~/.arize/config.toml` is created the first time you run `arize`:

```toml
base_url = "http://localhost:11434/v1"   # Ollama; LM Studio is http://localhost:1234/v1
model = "qwen3:8b"
api_key = ""                              # only for hosted services
max_steps = 8                             # tool steps allowed per message
confirm_actions = true                    # ask before writing files / running commands
workspace = "~"                           # folder ARIZE may read and write in
```

Every setting can also be set with an environment variable, e.g. `ARIZE_MODEL=llama3.1`.
Set `ARIZE_HOME` to keep settings and memory somewhere other than `~/.arize`.

ARIZE works with any server that speaks the OpenAI chat format, including Ollama,
LM Studio, llama.cpp, vLLM, or a hosted service. The model must support tool calling.

## Development

```bash
pip install -e '.[dev]'
pytest
ruff check . && ruff format --check .
```

The code lives in `src/arize_assistant/`:

- `cli.py`: the `arize` command and chat screen
- `agent.py`: the loop that calls the model, runs tools, and repeats
- `llm.py`: talks to the model server
- `memory.py`: the memory database
- `tools/`: web, file/command, and memory tools. To add a tool, write a function
  and add a `Tool(...)` entry to a module's `TOOLS` list.

## License

MIT, see [LICENSE](LICENSE).

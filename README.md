# investment-scanner-agent

An educational example agent built with LangChain, LangGraph, a local
Ollama model, and one MCP tool. It scans news and price data for a narrow
scope: **equities (stocks) and ETFs only**. It is a news/trend summarizer,
not a financial advisor -- it never recommends buying or selling anything.

## Project structure

```
investment-scanner-agent/
├── README.md
├── NOTES.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── config.ini                   # runtime settings (model, cache path/freshness)
├── run.py                       # entry point / demo script (hardcoded questions)
├── app.py                       # Streamlit UI (question box -> answer box)
├── market_log.md                # generated on first run -- the news cache (gitignored)
└── investment_scanner/          # the package
    ├── __init__.py              # exposes build_agent(), ask()
    ├── agent.py                 # the agent: tools, system prompt, create_agent
    │                            # (LangGraph rebuild commented at the bottom)
    ├── config.py                # loads config.ini, with built-in defaults
    └── mcp_server.py            # standalone MCP server (equity/ETF price tool)
```

- `investment_scanner/agent.py` -- the agent, built with LangChain's
  `create_agent()` on top of a local Ollama model. A commented-out
  LangGraph version of the exact same agent is at the bottom of the file --
  uncomment it once you're ready to move past `create_agent` and control
  the graph yourself.
- `investment_scanner/config.py` -- reads `config.ini` from the repo root
  (`ollama_model`, `log_path`, `freshness_hours`) and exposes it as
  `OLLAMA_MODEL`, `LOG_PATH`, `CACHE_FRESHNESS_HOURS`. Falls back to the
  project's built-in defaults if the file, or a given key in it, is
  missing, so the project still runs out of the box with no `config.ini`
  at all.
- `investment_scanner/mcp_server.py` -- your MCP use case: a standalone
  server exposing one tool, `get_stock_snapshot`, backed by free Yahoo
  Finance data via `yfinance`. The agent talks to this as a separate
  process over stdio, the same way it would talk to any third-party MCP
  server.
- `config.ini` -- edit this to change the Ollama model or cache behavior
  without touching code (see "Configuration" below).
- `run.py` -- the script you actually run; imports the package and asks it
  a couple of demo questions.
- `app.py` -- a one-page Streamlit UI: type a question in a text box, hit
  Ask, see the agent's answer in a text box below it. Uses the same
  `build_agent()`/`ask()` API as `run.py`.

## Why everything here is free

- **Ollama** runs the model locally on your own machine -- no API key, no
  per-token billing. The only cost is your own hardware/electricity.
- **ddgs** (DuckDuckGo search) is a keyless, free news search library.
- **yfinance** pulls from Yahoo Finance's public endpoints for free, no
  account needed.

## Hardware

- **RAM**: the default model, `llama3.1` (8B), needs roughly 8 GB RAM to
  run at all; 16 GB is a more comfortable baseline if you're also running
  a browser, IDE, or Streamlit alongside it. Skip the 70B/405B `llama3.1`
  variants unless you have serious hardware (64GB+ RAM or a strong GPU) --
  this project assumes the 8B default.
- **GPU**: not required. Ollama runs 8B models fine on CPU, just slower
  per response; Apple Silicon Macs get solid performance automatically via
  Metal, and an NVIDIA/CUDA GPU helps on other machines but isn't needed.
- **Disk**: a few GB free for the model pull, plus normal space for the
  Python venv and dependencies.
- **Network**: needed for setup (`pip install`, `ollama pull`) and at
  runtime for the two live-data tools -- `scan_market_news` (DuckDuckGo)
  and `get_stock_snapshot` (Yahoo Finance) -- unless a cached hit avoids
  it. Model inference itself is fully offline once the model is pulled.

## One-time setup

1. Install Python 3.10+ if you don't already have it -- download from
   [python.org](https://www.python.org/downloads/), or on a Mac:
   ```
   brew install python
   ```
   Check what you have with `python3 --version`.
2. Install Ollama if you haven't already -- download from
   [ollama.com](https://ollama.com/download), or on a Mac:
   ```
   brew install ollama
   ```
   Make sure it's actually running afterward (the Ollama app, or
   `ollama serve` in a terminal).
3. Pull a tool-calling-capable model (required -- not every Ollama model
   supports tool calls):
   ```
   ollama pull llama3.1
   ```
   This downloads several GB, so it'll take a few minutes depending on
   your connection. `qwen2.5` and `mistral-nemo` are alternatives that
   also support tool calling, if you want to compare.
4. Open a terminal in the project folder and set up Python:
   ```
   cd ~/projects/investment-scanner-agent
   python3 -m venv .venv
   source .venv/bin/activate   # on Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Run it

```
python run.py
```

This runs two example questions -- a news scan and a price snapshot -- and
prints the agent's answers to the terminal. Edit the `questions` list in
`run.py` to try your own.

### Or run the UI

```
streamlit run app.py
```

This opens a one-page app in your browser with a question box and an
answer box -- type any question and click Ask. The agent is built once
(the first click will be slower while Ollama and the MCP server start up)
and reused for the rest of the session, with each question/answer sharing
the same conversation memory.

## Configuration

`config.ini` (repo root) controls runtime settings without touching code:

```ini
[agent]
ollama_model = llama3.1

[cache]
log_path = market_log.md
freshness_hours = 4
```

- `ollama_model` -- which pulled Ollama model the agent uses. Must be
  tool-calling-capable (see "One-time setup" above).
- `log_path` -- where the news cache file lives, relative to the repo root.
- `freshness_hours` -- how long a cached `scan_market_news` result stays
  fresh before it's treated as stale and re-searched live.

`investment_scanner/config.py` reads this file and falls back to the
defaults above for the file itself, or any individual key, if missing --
so the project runs fine with no `config.ini` at all.

## How the pieces fit together

1. `build_agent()` (in `investment_scanner/agent.py`) starts a `ChatOllama`
   model pointed at your local model.
2. It launches `investment_scanner/mcp_server.py` as a subprocess via
   `MultiServerMCPClient` and pulls its tool(s) in with `get_tools()`.
3. Those MCP tools are combined with the locally-defined `scan_market_news`
   tool into one list and handed to `create_agent()` along with a system
   prompt that scopes the agent to equities/ETFs and bans investment
   advice.
4. `create_agent()` builds a LangGraph graph behind the scenes (a loop of
   model -> tool call -> tool -> model, repeating until a final answer),
   though you never touch that graph directly in this version.
5. `MemorySaver` gives the agent per-`thread_id` memory, so calling `ask()`
   twice with the same `thread_id` lets it recall the earlier turn.

## The news cache: market_log.md

`scan_market_news` checks `market_log.md` (created at the repo root on
first run) before searching. If it finds an entry for the same query
newer than `CACHE_FRESHNESS_HOURS` (4 hours by default, set in
`config.ini` -- see "Configuration" above), it returns that instead of
making a new network call;
otherwise it searches live and appends the result as a new dated entry.
This is deliberately separate from `MemorySaver`: `MemorySaver` remembers
a *conversation* in RAM for as long as the process is running;
`market_log.md` remembers *search results* on disk, across separate runs,
until they go stale. `get_stock_snapshot` is intentionally NOT cached the
same way -- a 4-hour-old price is just wrong, unlike a 4-hour-old
headline. The file is gitignored by default since it's generated runtime
state, not source -- remove that line from `.gitignore` if you'd rather
keep a version-controlled history of what the agent has searched.

## Moving to LangGraph

The bottom of `investment_scanner/agent.py` has a fully commented-out
LangGraph rebuild of the same agent (`build_langgraph_agent`, using
`StateGraph`, `ToolNode`, and `tools_condition` directly). It's commented
out so you have a working `create_agent` copy first. When you're ready to
add things `create_agent` doesn't support out of the box -- custom routing
between specialized sub-agents, a human-approval step before certain tool
calls, deterministic non-LLM steps mixed into the flow -- uncomment that
section (and stop calling `build_agent`) and build from there.

## Known rough edges (this is a teaching example, not production code)

- No retry logic if Ollama, DuckDuckGo, or Yahoo Finance are briefly
  unreachable.
- `yfinance` scrapes public Yahoo Finance endpoints, which occasionally
  change shape or rate-limit -- if `get_stock_snapshot` starts failing,
  that's the most likely cause, not a bug in the MCP wiring.
- No evaluation harness, logging, or alerting.
- Not every Ollama model supports tool calling; if the agent seems to
  ignore your tools entirely, double check `OLLAMA_MODEL` in `agent.py` is
  set to one that does.
- The `market_log.md` cache matches queries by exact normalized text
  (case/whitespace-insensitive only) -- asking the same thing in
  differently-worded ways still counts as separate cache entries and
  triggers separate searches. It also only grows; nothing prunes old
  entries from the file over time.

## License

MIT -- see `LICENSE`. Change the copyright holder there if you'd like.

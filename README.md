# investment-scanner-agent

An educational example agent built with LangChain, LangGraph, a local
Ollama model, and one MCP tool. It scans news and price data for a narrow
scope: **equities (stocks) and ETFs only**. It is a news/trend summarizer,
not a financial advisor -- it never recommends buying or selling anything.

## Project structure

```
investment-scanner-agent/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── run.py                       # entry point / demo script (hardcoded questions)
├── app.py                       # Streamlit UI (question box -> answer box)
└── investment_scanner/          # the package
    ├── __init__.py              # exposes build_agent(), ask()
    ├── agent.py                 # the agent: tools, system prompt, create_agent
    │                            # (LangGraph rebuild commented at the bottom)
    └── mcp_server.py            # standalone MCP server (equity/ETF price tool)
```

- `investment_scanner/agent.py` -- the agent, built with LangChain's
  `create_agent()` on top of a local Ollama model. A commented-out
  LangGraph version of the exact same agent is at the bottom of the file --
  uncomment it once you're ready to move past `create_agent` and control
  the graph yourself.
- `investment_scanner/mcp_server.py` -- your MCP use case: a standalone
  server exposing one tool, `get_stock_snapshot`, backed by free Yahoo
  Finance data via `yfinance`. The agent talks to this as a separate
  process over stdio, the same way it would talk to any third-party MCP
  server.
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

## One-time setup

1. Install Ollama if you haven't already -- download from
   [ollama.com](https://ollama.com/download), or on a Mac:
   ```
   brew install ollama
   ```
   Make sure it's actually running afterward (the Ollama app, or
   `ollama serve` in a terminal).
2. Pull a tool-calling-capable model (required -- not every Ollama model
   supports tool calls):
   ```
   ollama pull llama3.1
   ```
   This downloads several GB, so it'll take a few minutes depending on
   your connection. `qwen2.5` and `mistral-nemo` are alternatives that
   also support tool calling, if you want to compare.
3. Install Python 3.10+ if you don't already have it -- download from
   [python.org](https://www.python.org/downloads/), or on a Mac:
   ```
   brew install python
   ```
   Check what you have with `python3 --version`.
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



## License

MIT -- see `LICENSE`. Change the copyright holder there if you'd like.

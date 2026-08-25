# Build Notes: Investment Trends Scanner

A record of the design decisions and reasoning behind this project, kept
alongside the code so the "why" travels with the "what" on GitHub. Written
as a learning log from building an AI agent with free, local tooling.

## 1. What an agent actually is

An agent is a loop wrapped around a language model: instead of answering
once, the model repeatedly decides what to do next, takes an action,
observes the result, and decides again until the task is done. This
pattern is often called **ReAct** (Reason + Act). Four pieces make it
work:

- **The model** -- needs function/tool-calling ability: the capacity to
  output a structured request like `{"tool": "search", "args": {...}}`
  instead of just free text.
- **Tools** -- functions the model can invoke (search, a calculator, an
  API call, file I/O). Each needs a clear name, description, and input
  schema; the description is the *only* thing the model sees when
  deciding whether/how to call it, so it has to read like real
  documentation.
- **Memory** -- short-term (the running conversation fed back in each
  loop) and optionally long-term (external storage the agent can read/
  write across sessions).
- **Orchestration loop** -- the control logic: send state to the model,
  execute any requested tool call, append the result, repeat until a
  final answer or a stop condition (max steps, timeout).

For complex tasks, agents often add a **planning** step (break the goal
into subtasks first) and **guardrails** (iteration limits, permission
checks, human-approval steps) once they start taking real-world actions.

This is all vendor-agnostic -- any frontier model (OpenAI, Anthropic,
Google, or open-weight models like Llama/Qwen) can fill the "reasoning
engine" role. The framework around it doesn't change much by provider.

## 2. Why LangChain, and how it fits together

**LangChain** is the higher-level toolkit: model integrations for every
major provider, the `@tool` decorator for defining tools, memory/
checkpointer helpers, and the `create_agent()` factory this project uses.
It replaced the older `initialize_agent`/`AgentExecutor` pattern you'll
still see in older tutorials.

**LangGraph** is the lower-level execution engine underneath it -- a
general graph/state-machine runtime for defining workflows with loops,
branching, and persistence. As of LangChain's current architecture,
`create_agent()` is *built directly on top of* LangGraph: calling
`create_agent(...)` assembles a LangGraph graph for you (model node ->
tool node -> model node, looping) without you touching the graph API
directly.

They're not competing choices. `create_agent` covers the common case well
(one model, some tools, memory, a system prompt). You drop down to
LangGraph's `StateGraph` API directly when you need something it doesn't
model cleanly -- a planner routing to specialized sub-agents, a
human-approval interrupt before a risky tool call, deterministic non-LLM
steps mixed into the flow. That's exactly why this repo keeps a
hand-written LangGraph version of the same agent, commented out, at the
bottom of `investment_scanner/agent.py` -- a starting point for when this
project outgrows `create_agent`.

## 3. What's actually free vs. paid

This project deliberately avoids any billed API:

- **LangChain** (the framework) is free and open source. `LangSmith`,
  its optional observability/tracing platform, is a separate product with
  a free tier and paid tiers -- not required to build or run an agent.
- **Hosted LLM APIs** (OpenAI, Anthropic, etc.) bill per token with no
  meaningful free tier for real use -- true industry-wide, not specific to
  any one vendor.
- **Ollama** is the free path: it runs open-weight models (Llama, Qwen,
  Mistral, etc.) entirely on your own hardware. No API key, no per-call
  cost -- just your own compute/electricity. (Ollama also offers a
  separate paid "Ollama Cloud" hosted tier if you want bigger models
  without owning the hardware, but the local app used here is free.)
- **ddgs** (DuckDuckGo search) and **yfinance** (Yahoo Finance data) are
  both free, keyless libraries -- the other two ingredients that make this
  project cost nothing to run.

### Model choice for local tool calling

Not every Ollama model supports tool calling -- picking one that does is
non-negotiable for this project. Three that work, with real differences:

| Model | Maker | Notes |
|---|---|---|
| `llama3.1` | Meta | 8B/70B/405B sizes; most widely used, good general default (used in this repo) |
| `qwen2.5` | Alibaba | Wide size range (0.5B-72B); strong at coding/structured reasoning, multilingual |
| `mistral-nemo` | Mistral AI + NVIDIA | 12B; large 128k context window; Apache 2.0 license |

## 4. The project itself

**Investment Trends Scanner** -- scoped strictly to **equities and ETFs
only**. It's a news/trend summarizer, not a financial advisor: the system
prompt explicitly bans buy/sell recommendations and confident price
predictions, and closes every answer with an educational-use reminder.

Two tools:

1. `scan_market_news` -- a local `@tool`-decorated function using `ddgs`
   for free, keyless news search.
2. `get_stock_snapshot` -- loaded from a **separate MCP server**
   (`investment_scanner/mcp_server.py`), launched as its own subprocess
   and connected over stdio via `langchain-mcp-adapters`. This is the
   project's MCP use case: the agent treats it exactly like it would treat
   a third-party MCP server, even though we wrote it ourselves. It uses
   `yfinance` for free price data and rejects any ticker that isn't
   classified `EQUITY` or `ETF`, enforcing the project's scope at the tool
   level, not just in the prompt.

Memory uses `MemorySaver` (in-process only -- state resets when the
script exits), keyed by `thread_id`, so multiple questions in the same run
can build on each other.

## 5. Project layout

```
investment-scanner-agent/
├── README.md              # setup + run instructions
├── NOTES.md               # this file -- design log
├── LICENSE                # MIT
├── .gitignore
├── requirements.txt
├── run.py                 # entry point / demo script
└── investment_scanner/
    ├── __init__.py        # exposes build_agent(), ask()
    ├── agent.py            # tools, system prompt, create_agent
    │                       # (LangGraph rebuild commented at the bottom)
    └── mcp_server.py        # the MCP server (equity/ETF snapshot tool)
```

Set up with a `venv`, `pip install -r requirements.txt`, `ollama pull
llama3.1`, then `python run.py`. Full steps are in `README.md`.

### A real bug we hit

`ModuleNotFoundError: No module named 'ddgs'` on first run -- caused by
`pip install` and `python3 run.py` pointing at *different* Python
environments (the venv wasn't activated in the shell that ran `run.py`).
Fixed by confirming `which python3` pointed inside `.venv/` in the same
shell before both the install and the run. Worth remembering: this is the
single most common gotcha with any Python venv workflow, not specific to
this project.

## 6. Running it automatically

An agent is just a script -- no special deployment mechanism exists for
"agents" as a category. The same tools you'd use for any script apply:
`cron` or `launchd` (macOS-native, more reliable across sleep/wake than
cron) for scheduled runs, a webhook/queue listener for event-driven runs,
a `systemd`-managed long-running process for always-on polling. The one
agent-specific wrinkle: decide whether repeated runs should share a
`thread_id` (continued memory across runs) or start fresh each time, and
remember Ollama itself needs to be running in the background before a
scheduled job can reach it.

## 7. Tuning prompts for this agent

Small local models (8B-class, like `llama3.1`) are noticeably less
reliable than frontier hosted models at decomposing one compound ask into
several tool calls on their own. A single prompt asking for "presidential
announcements + Fed comments + economist forecasts + analyst upgrades,
each searched separately" may still collapse into one shallow search. The
more reliable pattern with a small local model is to ask several simpler,
single-topic questions in sequence (same `thread_id`, so context carries
over) rather than one large compound one, then optionally ask the agent to
synthesize the separate answers into a single digest at the end.

## 8. Where this could go next

- Fixed watchlist + scheduled runs + output to a file/digest instead of
  stdout, for genuine daily-use value instead of just a demo.
- A second data source, so one flaky feed (Yahoo Finance rate limits,
  DuckDuckGo hiccups) doesn't leave the agent with nothing.
- Uncomment the LangGraph version in `agent.py` to get explicit control
  over the graph -- useful before adding a second specialized sub-agent
  or a human-approval step.
- LangSmith tracing (free tier) if debugging *why* the agent chose a
  particular tool call becomes worth the setup.

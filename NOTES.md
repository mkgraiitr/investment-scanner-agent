# Build Notes: Investment Trends Scanner

A record of the design decisions and reasoning behind this project, kept
alongside the code so the "why" travels with the "what" on GitHub. Written
as a learning log from building an AI agent with free, local tooling --
and as a plain-language guide to what an AI agent actually is, using this
project as the working example throughout.

## 1. What an agent actually is: Model + Tools + Memory + Loop

An agent is a loop wrapped around a language model: instead of answering a
question once, the model repeatedly decides what to do next, takes an
action, observes the result, and decides again -- until the task is
actually done. This pattern is often called **ReAct** (Reason + Act). Four
things make it possible, and this project uses all four:

- **Model** -- the reasoning engine. It needs function/tool-calling
  ability: the capacity to output a structured request like
  `{"tool": "search", "args": {...}}` instead of just free text. In this
  project, that's a local Ollama model (`llama3.1` by default) via
  `ChatOllama` -- no API key, no cost, runs entirely on your own machine.
- **Tools** -- functions the model can invoke to actually touch the world
  (search, a calculator, an API call, file I/O). Each needs a clear name,
  description, and input schema; the description is the *only* thing the
  model sees when deciding whether/how to call it, so it has to read like
  real documentation. This project has two: `scan_market_news` (a local
  function using free DuckDuckGo search) and `get_stock_snapshot` (loaded
  from a small MCP server backed by free Yahoo Finance data).
- **Memory** -- what the agent remembers, and for how long. Short-term is
  the running conversation, fed back into the model on every turn of the
  loop. Long-term is anything that survives past one run -- external
  storage the agent can read and write across sessions. This project has
  both: `MemorySaver` gives it short-term, per-conversation memory, and
  `market_log.md` (Section 4) gives it real long-term memory -- a plain
  file on disk that outlives the process.
- **Loop** -- the control logic that actually runs the above three in a
  cycle: send the current state to the model, execute any tool call it
  requests, feed the result back in, repeat until the model gives a final
  answer or a stop condition is hit (max steps, timeout). This is the one
  piece you genuinely can't remove and still call it an agent -- without
  it, the model just answers once and stops, which is a chatbot with tools
  bolted on, not an agent. In this project, `create_agent()` builds this
  loop for you as a LangGraph graph (model -> tool -> model, looping) --
  you never have to write it by hand.

Two more things show up once tasks get harder, but they're supporting
layers, not core pillars: **planning** (breaking a goal into subtasks
before acting) and **guardrails** (iteration limits, permission checks,
human-approval steps before risky actions). A simple task doesn't need
either; they get added as the task or the risk grows.

This is all vendor-agnostic -- any frontier model (OpenAI, Anthropic,
Google, or an open-weight model like Llama or Qwen) can fill the
"reasoning engine" role. The framework around it barely changes by
provider.

### A quick word on automation (cron isn't one of the four)

You'll eventually want this agent to run on its own -- once a day, once an
hour -- and Section 6 covers exactly how (cron, launchd, a scheduler). But
it's worth being clear about what that layer actually is: cron doesn't
make a script more of an agent, it just decides *when* an already-complete
agent gets invoked. A cron job calling this script hourly is the same
relationship as a `for` loop calling a Python function ten times -- the
function isn't "more of a function" for being called on a schedule.
Automation is a deployment decision, not part of what makes something an
agent in the first place. Worth knowing, not worth dwelling on.

### Other formulas you'll see for the same idea

Different explanations name these same four things differently. Two common
ones, and how they map:

| Elsewhere | Maps to | Note |
|---|---|---|
| LLM / "Core" | Model | Same thing, different word |
| Shell / "Interface Layer" tool connectors | Tools | Shell is just the *specific* tool a CLI/coding agent uses to touch the world (running commands) -- other agents use search APIs, database calls, etc. instead |
| File System / "Memory Layer" | Memory | Some framings split this further into short-term, long-term (often a vector database), and episodic (logs of past runs) -- a useful refinement. This project's `market_log.md` is a concrete long-term/episodic example, just built as a plain file instead of a vector index (Section 4) |
| Orchestrator / control loop | Loop | Often the one piece left unnamed when people list "LLM + Shell + File System + Cron" -- easy to miss since it's implicit in "LLM," but it's what actually turns a tool call and a memory file into an agent's action, rather than just a tool call and a file sitting next to each other |
| Cron / Automation | *(not one of the four)* | Shows up in some formulas as a fourth ingredient, but per above it's a deployment concern, not an architectural one |

Whichever formula you run into, it's almost always these same four ideas
wearing different names, sometimes with a deployment detail (like Cron)
mixed in for good measure.

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

### File System as memory, for real this time

Section 6 below flagged that `MemorySaver` doesn't actually give this
project a File-System-backed memory -- it's RAM only, gone when the
process exits. We closed that gap: `scan_market_news` now checks
`market_log.md` (repo root) before searching, and appends a new dated
entry after every live search. A query found in the file within the last
`CACHE_FRESHNESS_HOURS` (4h default) is returned straight from disk, no
network call; older than that, and it's treated as a miss and re-searched.

This is a genuinely different kind of memory from `MemorySaver`, not a
duplicate of it: `MemorySaver` remembers a conversation (what did we
already say to each other), scoped to one running process. `market_log.md`
remembers search results (what did we already find out), persisted on
disk, surviving across completely separate runs of `run.py` on different
days. `get_stock_snapshot` deliberately does NOT get this treatment --
caching a price for 4 hours would make the tool actively wrong, whereas a
4-hour-old headline is usually still a fine answer. Matching is exact
(case/whitespace-normalized) text, so differently-phrased repeat questions
still count as new searches -- a known, accepted limitation, not a bug.

## 5. Project layout

```
investment-scanner-agent/
├── README.md              # setup + run instructions
├── NOTES.md               # this file -- design log
├── LICENSE                # MIT
├── .gitignore
├── requirements.txt
├── run.py                 # entry point / demo script
├── market_log.md          # generated on first run -- news cache (gitignored)
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

Whichever mechanism you pick, remember it's just the trigger -- the agent
itself (Section 1) is exactly as much of an agent whether it's invoked by
cron, by hand from a terminal, or by clicking Ask in `app.py`.

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

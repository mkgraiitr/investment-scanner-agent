"""
Investment Trends Scanner -- agent definition.

SCOPE: equities and ETFs only. This agent SUMMARIZES news and trends -- it
is not a financial advisor. It never recommends buying or selling anything,
and the system prompt below says so explicitly. Treat any output as a
starting point for your own research, not investment advice.

Architecture:
  - build_agent() below uses LangChain's create_agent(), which is built on
    LangGraph internally -- you get LangGraph's execution engine "for
    free" without touching its graph API directly.
  - A hand-written LangGraph version of this exact same agent is included,
    commented out, at the bottom of this file. It's kept commented so you
    have a working create_agent copy first; uncomment it (and stop using
    build_agent) when you're ready to take over the orchestration
    yourself -- e.g. to add custom routing between specialized sub-agents,
    a human-approval step, or deterministic non-LLM steps mixed into the
    flow.

Tools:
  1. scan_market_news   -- free DuckDuckGo news search (no API key, no
                            cost), scoped by its docstring to equities/ETF
                            topics.
  2. get_stock_snapshot -- loaded from a separate MCP server
                            (mcp_server.py, in this same package) over
                            stdio. This is the one MCP use case for this
                            project: a live price snapshot tool that lives
                            in its own process, wired in the same way
                            you'd wire up any third-party MCP server.

Model: a local Ollama model -- no API key, no per-token cost, runs
entirely on your own machine. You need Ollama installed and a
tool-calling-capable model pulled first:
    ollama pull llama3.1
(qwen2.5 and mistral-nemo are alternatives that also support tool calling.)
"""

from pathlib import Path

from ddgs import DDGS
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver

OLLAMA_MODEL = "llama3.1"  # must be a tool-calling-capable model you've pulled
MCP_SERVER_PATH = str(Path(__file__).parent / "mcp_server.py")

SYSTEM_PROMPT = """\
You are an Investment Trends Scanner, scoped ONLY to equities (stocks) and \
ETFs. You do not cover crypto, forex, commodities, bonds, or options -- if \
asked about those, say they're out of scope instead of guessing.

Your job is to surface and summarize recent news and trends relevant to \
equities and ETFs -- you are NOT a financial advisor. Follow these rules \
in every response:
- Never tell the user to buy, sell, or hold anything, and never predict \
future prices with confidence.
- When you cite a news item, mention its source and date if the tool gave \
you one.
- Clearly separate "here's what's being reported" from your own summary \
or interpretation of it.
- If a question falls outside equities/ETFs, say so and decline rather \
than answering anyway.
- End your answer with a short reminder that this is for educational \
purposes only and is not financial advice.
"""


# ---------------------------------------------------------------------------
# Tool 1: free news search, scoped to equities/ETFs by its docstring
# ---------------------------------------------------------------------------

@tool
def scan_market_news(query: str, max_results: int = 5) -> str:
    """
    Search recent news for a topic relevant to equities or ETFs.

    Use this for company news, sector trends, index/ETF flows, earnings,
    guidance, or macro stories that affect stocks or ETFs. Do NOT use this
    for crypto, forex, or commodities -- that's out of scope for this agent.

    Args:
        query: What to search for, e.g. "Nvidia earnings" or "semiconductor
            ETF outflows" or "Fed rate decision impact on equities".
        max_results: How many headlines to return (default 5).
    """
    try:
        results = DDGS().news(
            query=query, region="us-en", safesearch="off", max_results=max_results
        )
    except Exception as e:
        return f"News search failed: {e}"

    if not results:
        return f"No recent news found for '{query}'."

    lines = []
    for r in results:
        title = r.get("title", "")
        source = r.get("source", "unknown source")
        date = r.get("date", "unknown date")
        url = r.get("url", "")
        snippet = r.get("body", "")
        lines.append(f"- [{date}] {title} ({source}) -- {snippet}\n  {url}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Build and run the agent (LangChain create_agent, backed by LangGraph
# internally -- see the commented section at the bottom for the explicit
# LangGraph version of the same thing)
# ---------------------------------------------------------------------------

async def build_agent():
    model = ChatOllama(model=OLLAMA_MODEL, temperature=0.2)

    # MCP client: launches mcp_server.py as a subprocess over stdio and
    # exposes whatever @mcp.tool()-decorated functions it defines as
    # ordinary LangChain tools, ready to hand to create_agent.
    mcp_client = MultiServerMCPClient(
        {
            "equity_etf_snapshot": {
                "command": "python",
                "args": [MCP_SERVER_PATH],
                "transport": "stdio",
            }
        }
    )
    mcp_tools = await mcp_client.get_tools()

    tools = [scan_market_news, *mcp_tools]

    checkpointer = MemorySaver()

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
    return agent


async def ask(agent, message: str, thread_id: str = "scanner-demo"):
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": message}]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return result["messages"][-1].content


# =============================================================================
# LANGGRAPH VERSION -- commented out on purpose, uncomment when you're ready
# =============================================================================
# Everything above uses LangChain's create_agent(), which builds a LangGraph
# graph for you behind the scenes (model node -> tool node -> model node,
# looping until a final answer). Below is the same agent built by hand with
# LangGraph's explicit graph API. It behaves the same way as the
# create_agent version above -- the point of uncommenting this later isn't
# to change behavior, it's to have explicit control over the graph so you
# can extend it (add more nodes, branch conditionally, insert a
# human-approval interrupt, etc) in ways create_agent doesn't expose.
#
# from langgraph.graph import StateGraph, MessagesState, START
# from langgraph.prebuilt import ToolNode, tools_condition
#
# async def build_langgraph_agent():
#     model = ChatOllama(model=OLLAMA_MODEL, temperature=0.2)
#
#     mcp_client = MultiServerMCPClient(
#         {
#             "equity_etf_snapshot": {
#                 "command": "python",
#                 "args": [MCP_SERVER_PATH],
#                 "transport": "stdio",
#             }
#         }
#     )
#     mcp_tools = await mcp_client.get_tools()
#     tools = [scan_market_news, *mcp_tools]
#     model_with_tools = model.bind_tools(tools)
#
#     def call_model(state: MessagesState):
#         # create_agent handles the system prompt for you; here we have to
#         # prepend it ourselves on every call since we're driving the graph
#         # directly.
#         messages = [{"role": "system", "content": SYSTEM_PROMPT}] + state["messages"]
#         response = model_with_tools.invoke(messages)
#         return {"messages": response}
#
#     builder = StateGraph(MessagesState)
#     builder.add_node("call_model", call_model)
#     builder.add_node("tools", ToolNode(tools))
#     builder.add_edge(START, "call_model")
#     # tools_condition routes to "tools" if the model's last message asked
#     # for a tool call, otherwise straight to END.
#     builder.add_conditional_edges("call_model", tools_condition)
#     builder.add_edge("tools", "call_model")
#
#     checkpointer = MemorySaver()
#     graph = builder.compile(checkpointer=checkpointer)
#     return graph
#
#
# async def ask_langgraph(graph, message: str, thread_id: str = "scanner-demo"):
#     result = await graph.ainvoke(
#         {"messages": [{"role": "user", "content": message}]},
#         config={"configurable": {"thread_id": thread_id}},
#     )
#     return result["messages"][-1].content
# =============================================================================

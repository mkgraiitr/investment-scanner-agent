"""
Investment Trends Scanner -- educational LangChain + LangGraph + Ollama +
MCP example, scoped to equities and ETFs only.

Public API:
    build_agent() -- construct the agent (async, since it talks to the
                      MCP server to fetch tools).
    ask(agent, message, thread_id) -- send one message to a built agent.
"""

from investment_scanner.agent import ask, build_agent

__all__ = ["build_agent", "ask"]
__version__ = "0.1.0"

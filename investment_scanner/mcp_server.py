"""
MCP server exposing one tool: get_stock_snapshot(ticker).

This is the MCP use case for this project. Instead of writing the stock
lookup function directly inside the agent module, it lives here as its own
standalone process. The agent connects to it over stdio using
langchain-mcp-adapters -- the same way it could connect to any MCP server
written by someone else, in any language. Swapping this file out for a
different MCP server (e.g. one exposing a real broker API) would not
require changing the agent's code at all.

Data source: Yahoo Finance via the free `yfinance` library (no API key,
no cost). Scope: intentionally rejects anything that isn't classified as
an equity (stock) or ETF, to match this project's "equities and ETFs
only" scope -- e.g. asking it for a mutual fund, index, or currency
ticker returns a scope-rejection message instead of data.

You normally don't run this file directly -- investment_scanner/agent.py
launches it as a subprocess. Running it standalone (`python -m
investment_scanner.mcp_server`) just starts it listening on stdio, which
isn't useful by itself without an MCP client talking to it.
"""

from datetime import datetime, timezone

import yfinance as yf
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("equity-etf-snapshot")


@mcp.tool()
def get_stock_snapshot(ticker: str) -> str:
    """
    Get a current price snapshot for one equity (stock) or ETF ticker.

    Only covers equities and ETFs -- rejects other asset types (crypto,
    mutual funds, indices, currencies, etc.) by design, since this project
    is scoped to equities and ETFs only.

    Args:
        ticker: Exchange ticker symbol, e.g. "AAPL", "MSFT", "SPY", "QQQ".
    """
    try:
        info = yf.Ticker(ticker).info
    except Exception as e:
        return f"Could not fetch data for '{ticker}': {e}"

    if not info or (info.get("regularMarketPrice") is None and info.get("currentPrice") is None):
        return f"No data found for '{ticker}' -- check the ticker symbol."

    quote_type = info.get("quoteType", "UNKNOWN")
    if quote_type not in ("EQUITY", "ETF"):
        return (
            f"'{ticker}' is classified as {quote_type}, which is out of "
            "scope for this tool -- it only covers equities and ETFs."
        )

    price = info.get("currentPrice") or info.get("regularMarketPrice")
    prev_close = info.get("previousClose")
    pct_change = None
    if price is not None and prev_close:
        pct_change = round((price - prev_close) / prev_close * 100, 2)

    snapshot = {
        "ticker": ticker.upper(),
        "name": info.get("shortName") or info.get("longName"),
        "type": quote_type,
        "price": price,
        "previous_close": prev_close,
        "pct_change_today": pct_change,
        "day_high": info.get("dayHigh"),
        "day_low": info.get("dayLow"),
        "volume": info.get("volume"),
        "as_of_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return str(snapshot)


if __name__ == "__main__":
    mcp.run(transport="stdio")

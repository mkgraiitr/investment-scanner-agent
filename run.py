"""
Entry point / demo script.

Run from the repo root:
    python run.py

Edit the `questions` list below to try your own prompts.
"""

import asyncio

from investment_scanner import ask, build_agent


async def main():
    agent = await build_agent()

    questions = [
        "What are the latest developments affecting semiconductor equities "
        "and related ETFs like SMH or SOXX?",
        "Give me a quick price snapshot for AAPL and QQQ.",
    ]
    for q in questions:
        print(f"\n=== USER: {q}")
        answer = await ask(agent, q)
        print(f"=== AGENT: {answer}")


if __name__ == "__main__":
    asyncio.run(main())

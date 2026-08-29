"""
Loads runtime settings from config.ini in the repo root. Falls back to the
same defaults the project shipped with if the file, or a given key inside
it, is missing -- so the project still runs out of the box without one.
"""

import configparser
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.ini"

_parser = configparser.ConfigParser()
_parser.read(CONFIG_PATH)

OLLAMA_MODEL = _parser.get("agent", "ollama_model", fallback="llama3.1")

CACHE_FRESHNESS_HOURS = _parser.getfloat("cache", "freshness_hours", fallback=4)
LOG_PATH = Path(__file__).resolve().parent.parent / _parser.get(
    "cache", "log_path", fallback="market_log.md"
)

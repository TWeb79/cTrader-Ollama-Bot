"""Step 1: Data Preparation — load and format historical trade data for ML training.

Loads historical candles and trade events from the cTrader MCP server and
local events.json, producing clean feature arrays suitable for the ML model.

Author: Inventions4All - github:TWeb79
Version: 1.0.0  (deployment: 2026-07-30T17:53:53+02:00)
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np

log = logging.getLogger("ai_trader.data_prep")

EVENTS_LOG_FILE = "events.json"


async def fetch_historical_candles(
    client: Any,
    config: dict[str, Any],
    tool_names: dict[str, str],
) -> list[dict]:
    """Fetch historical candles from the cTrader MCP server.

    Args:
        client: cTrader MCP client instance.
        config: Configuration dictionary with symbol, history_months, timeframe.
        tool_names: Mapping of logical names to MCP tool names.

    Returns:
        List of candle dictionaries, or empty list on failure.
    """
    since = (
        datetime.utcnow()
        - timedelta(days=30 * config["history_months"])
    ).isoformat()
    history_args = {
        "symbolName": config["symbol"],
        "timeframe": config["candle_timeframe"],
        "from": since,
        "to": datetime.utcnow().isoformat(),
    }
    log.debug("Fetching historical candles: %s", history_args)
    raw = await client.call(tool_names["historical_candles"], history_args)
    candles = raw if isinstance(raw, list) else raw.get("bars", [])
    log.info(
        "Received %d candle(s) for %s (timeframe=%s)",
        len(candles),
        config["symbol"],
        config["candle_timeframe"],
    )
    return candles


def load_events(filepath: str = EVENTS_LOG_FILE) -> list[dict]:
    """Load trade events from the local JSON events file.

    Args:
        filepath: Path to the events JSON file.

    Returns:
        List of event dicts, or empty list if file is missing or invalid.
    """
    if not __import__("os").path.exists(filepath):
        log.warning("No events file found at %s", filepath)
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as fh:
            events = json.load(fh)
        log.info("Loaded %d event(s) from %s", len(events), filepath)
        return events
    except (json.JSONDecodeError, ValueError) as exc:
        log.warning("Failed to parse %s: %s", filepath, exc)
        return []


async def prepare_training_data(
    client: Any,
    config: dict[str, Any],
    tool_names: dict[str, str],
    events_file: str = EVENTS_LOG_FILE,
) -> tuple[list[dict], list[dict]]:
    """Prepare training data from historical candles and trade events.

    Combines:
    - Historical OHLCV candles from the MCP server.
    - Trade outcomes from events.json (profits, pips, close prices).

    Args:
        client: cTrader MCP client instance.
        config: Configuration dictionary.
        tool_names: Mapping of logical names to MCP tool names.
        events_file: Path to the events JSON file.

    Returns:
        Tuple of (candles, events) — both as lists of dicts.
    """
    candles = await fetch_historical_candles(client, config, tool_names)
    events = load_events(events_file)

    log.info(
        "Data preparation complete: %d candles, %d events",
        len(candles),
        len(events),
    )
    return candles, events
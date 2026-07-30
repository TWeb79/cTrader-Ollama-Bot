"""Compute high-level statistics from trade events."""

import json
import logging
import os
from datetime import datetime
from typing import Any

log = logging.getLogger("ai_trader.dashboard.stats")


def load_events(filepath: str) -> list[dict[str, Any]]:
    """Load trade events from JSON file."""
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, ValueError) as exc:
        log.warning("Failed to parse events file %s: %s", filepath, exc)
        return []


def compute_stats(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute P/L statistics and trade counts from events.

    Args:
        events: List of trade event dicts.

    Returns:
        Dict with total_trades, win_rate, net_pnl, avg_pnl, etc.
    """
    total = len(events)
    if total == 0:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "net_pnl": 0.0,
            "avg_pnl": 0.0,
            "max_win": 0.0,
            "max_loss": 0.0,
            "total_pips": 0.0,
            "avg_pips": 0.0,
            "last_trade": None,
        }

    pnls: list[float] = []
    pips: list[float] = []
    wins = 0
    losses = 0
    max_win = 0.0
    max_loss = 0.0

    for event in events:
        pnl = float(event.get("grossProfit", 0.0) or 0.0)
        pip_val = float(event.get("pips", 0.0) or 0.0)
        pnls.append(pnl)
        pips.append(pip_val)
        if pnl > 0:
            wins += 1
            max_win = max(max_win, pnl)
        elif pnl < 0:
            losses += 1
            max_loss = min(max_loss, pnl)

    net_pnl = sum(pnls)
    avg_pnl = net_pnl / total if total else 0.0
    win_rate = (wins / total) * 100 if total else 0.0
    total_pips = sum(pips)
    avg_pips = total_pips / total if total else 0.0

    last_event = events[-1] if events else None
    last_trade = None
    if last_event:
        last_trade = {
            "timestamp": last_event.get("timestamp"),
            "type": last_event.get("type"),
            "entry_price": last_event.get("entryPrice"),
            "close_price": last_event.get("closePrice"),
            "gross_profit": last_event.get("grossProfit"),
            "pips": last_event.get("pips"),
        }

    return {
        "total_trades": total,
        "win_rate": round(win_rate, 2),
        "net_pnl": round(net_pnl, 2),
        "avg_pnl": round(avg_pnl, 2),
        "max_win": round(max_win, 2),
        "max_loss": round(max_loss, 2),
        "total_pips": round(total_pips, 2),
        "avg_pips": round(avg_pips, 2),
        "last_trade": last_trade,
    }

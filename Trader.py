#!/usr/bin/env python3
"""
US500 AI Trading Assistant — Ollama (local LLM) + cTrader MCP server
======================================================================

WHAT THIS DOES
  1. Connects to your local cTrader MCP server via the Model Context Protocol (HTTP transport).
  2. Pulls ~6 months of historical candles for US500 and computes a simple
     indicator summary (SMA20/50, RSI14, range).
  3. Asks a local Ollama model to propose ONE simple rule-based strategy
     from that summary (JSON output).
  4. Enters a polling loop: fetches the live price, asks Ollama for a
     BUY / SELL / HOLD decision against the strategy, and places a market
     order with stop-loss and take-profit if signaled.

BEFORE YOU RUN THIS
  - Install deps:      pip install mcp ollama pandas
  - Pull a tool-capable model:  ollama pull qwen3.5:9b (or use qwen3.5:9b if already available)
  - The cTrader Desktop MCP server should be running on http://127.0.0.1:9876/mcp
  - Run once with dry_run=True (default) and inspect the logged "Available tools"
    list, then ensure TOOL_NAMES below match your server's actual tool names
    and argument shapes (they vary between implementations).
  - Use a DEMO account until you've verified every tool call works end to end.

DISCLAIMER
  This is a technical scaffold, not financial advice. AI-generated signals
  can be wrong, the model can misread indicators, and tool-calling models
  occasionally produce malformed arguments. You are responsible for
  reviewing every trade this places, for sizing risk appropriately, and
  for testing thoroughly on a demo account before ever pointing it at a
  live account. Nothing here guarantees profitability.
"""

import asyncio
import json
import logging
from contextlib import AsyncExitStack
from datetime import datetime, timedelta
from typing import Any, Optional


from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from volume_profile_strategy import (
    MarketContext,
    SchemaResult,
    calculate_position_size,
    compute_volume_profile,
    determine_schema,
    evaluate_entry,
    get_schema_direction,
    get_schema_targets,
    try_magnet_trade,
)

import ollama
import os

# ============================================================
# CONFIGURATION — edit before running
# ============================================================

CONFIG = {
    # --- cTrader MCP server connection ---
    "mcp_transport": "http",               # HTTP transport for cTrader Desktop MCP server
    "mcp_url": "http://127.0.0.1:9876/mcp",

    # --- Ollama ---
    "ollama_model": "qwen2.5-coder:7b", #"qwen3.5:9b",
    "ollama_host": "http://localhost:11434",

    # --- Strategy selection ---
    "strategy": "ollama",              # "ollama" or "volume_profile"

    # --- Instrument & analysis ---
    "symbol": "US500",
    "history_months": 6,
    "candle_timeframe": "m5",              # 5-minute candles for the 6-month strategy scan

    # --- Volume Profile Strategy parameters ---
    "vp_bin_size": None,                   # None = auto (price_range / 100)
    "vp_value_area_pct": 0.70,             # % of volume defining the value area
    "vp_zone_proximity_pips": 5.0,         # proximity in pips for schema detection
    "vp_min_risk_distance_pips": 30.0,     # min SL distance before trade is valid
    "vp_max_schema_entry_distance_pips": 15.0,  # max distance from VA boundary for inside-VA schemas
    "vp_max_sl_pips": 20.0,                # maximum stop-loss in pips
    "vp_min_rr_ratio": 1.2,                # minimum risk:reward ratio
    "vp_base_risk_pct": 2.0,               # base risk percentage per trade
    "vp_reduced_risk_pct": 0.5,            # reduced risk when daily/weekly target hit
    "vp_max_volume_lots": 10.0,            # hard ceiling on position size
    "vp_min_trade_volume": 0.1,            # minimum trade volume in lots
    "vp_volume_step": 0.1,                 # rounding step for volume
    "vp_volume_min_units": 0.01,           # broker minimum volume units
    "vp_volume_max_units": 100.0,          # broker maximum volume units
    "vp_pip_size": 0.01,                   # pip/point size for US500
    "vp_pip_value_per_unit": 1.0,          # value of one pip per unit (configurable per instrument)
    "vp_balance": 10000.0,                 # account balance for position sizing

    # --- Magnet mode (mean reversion fallback) ---
    "enable_magnet_mode": True,
    "magnet_start_hour": 17,
    "rsi_period": 14,
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    "bb_period": 20,
    "bb_std_dev": 2.0,

    # --- Live monitoring loop ---
    "poll_interval_seconds": 60,
    "max_loop_iterations": 1000,           # set an int (e.g. 5) for a quick test run

    # --- Ollama intraday data ---
    "ollama_intraday_hours": 4,             # hours of intraday candles to include in live signal prompt

    # --- Risk management ---
    "trade_volume": 0.1,                   # lots — fallback if risk-based sizing is disabled
    "stop_loss_points": 15.0,              # instrument points (not pips) — tune for US500's volatility
    "take_profit_points": 30.0,
    "max_open_positions": 1,
    "trade_risk_pct": 2.0,                 # risk percentage per trade for position sizing
    "trade_balance": 10000.0,              # account balance for position sizing
    "trade_pip_value_per_unit": 1.0,       # value of one pip per unit
    "trade_max_volume_lots": 10.0,         # hard ceiling on position size
    "trade_min_volume": 0.1,               # minimum trade volume
    "trade_volume_step": 0.1,              # rounding step for volume
    "trade_volume_min_units": 0.01,        # broker minimum volume units
    "trade_volume_max_units": 100.0,       # broker maximum volume units

    # --- Safety switch ---
    "dry_run": True,                       # True = never sends real orders, only logs intended trades
}

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("trader.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("ai_trader")


# ============================================================
# MCP CLIENT WRAPPER
# ============================================================

class CTraderMCPClient:
    """Thin wrapper around an MCP ClientSession talking to the cTrader MCP server."""

    def __init__(self, config: dict):
        self.config = config
        self._stack = AsyncExitStack()
        self.session: Optional[ClientSession] = None
        self.available_tools: dict[str, Any] = {}

    async def connect(self):
        if self.config["mcp_transport"] == "http":
            read, write = await self._stack.enter_async_context(
                streamable_http_client(self.config["mcp_url"])
            )
        else:
            raise ValueError(f"Unknown mcp_transport: {self.config['mcp_transport']}")

        self.session = await self._stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()

        tools_result = await self.session.list_tools()
        self.available_tools = {t.name: t for t in tools_result.tools}
        log.info("Connected to cTrader MCP server. Available tools:")
        for name, tool in self.available_tools.items():
            desc = (tool.description or "").strip().replace("\n", " ")[:100]
            log.info("  - %s: %s", name, desc)
        log.info(
            "If TOOL_NAMES below don't match the names above, edit TOOL_NAMES "
            "before this script will work correctly."
        )

    async def call(self, tool_name: str, arguments: dict) -> Any:
        if tool_name not in self.available_tools:
            raise RuntimeError(
                f"Tool '{tool_name}' not found on this MCP server. "
                f"Available: {list(self.available_tools.keys())}. "
                f"Update TOOL_NAMES in this script to match your server."
            )
        result = await self.session.call_tool(tool_name, arguments)
        # MCP tool results are a list of content blocks; most servers return one text/JSON block
        for block in result.content:
            if hasattr(block, "text"):
                try:
                    return json.loads(block.text)
                except (json.JSONDecodeError, TypeError):
                    return block.text
        return result

    async def close(self):
        await self._stack.aclose()


# ============================================================
# TOOL NAME MAPPING
# ============================================================
# MCP server tool names vary by implementation. After connecting once, check
# the logged "Available tools" list and correct these to match your server
# exactly (including expected argument keys, which also vary).

TOOL_NAMES = {
    "historical_candles": "get_trendbars",
    "current_price": "get_spot_prices",
    "place_order": "place_market_order",
    "list_positions": "get_positions",
    "close_position": "close_position",
    "list_pending_orders": "get_pending_orders",
    "list_deals": "get_deals",
    "list_order_history": "get_order_history",
    "amend_position": "amend_position",
}


# ============================================================
# OLLAMA ANALYSIS
# ============================================================

def compute_indicators(candles: list[dict]) -> dict:
    """Lightweight indicator summary so we don't have to dump raw OHLC into the prompt."""
    import pandas as pd

    log.debug("Computing indicators from %d candle(s)", len(candles))
    df = pd.DataFrame(candles)
    df = df.rename(columns={c: c.lower() for c in df.columns})
    log.debug("DataFrame columns after rename: %s", list(df.columns))
    close = df["close"].astype(float)

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))

    sma20_val = sma20.iloc[-1]
    sma50_val = sma50.iloc[-1]
    trend = "bullish" if (not pd.isna(sma20_val) and not pd.isna(sma50_val) and sma20_val > sma50_val) else "bearish"
    result = {
        "last_close": round(float(close.iloc[-1]), 2),
        "6m_high": round(float(close.max()), 2),
        "6m_low": round(float(close.min()), 2),
        "sma20": round(float(sma20.iloc[-1]), 2) if not sma20.isna().all() else None,
        "sma50": round(float(sma50.iloc[-1]), 2) if not sma50.isna().all() else None,
        "rsi14": round(float(rsi.iloc[-1]), 2) if not rsi.isna().all() else None,
        "trend_20_vs_50": trend,
    }
    log.debug("Indicators computed: %s", result)
    return result


async def fetch_intraday_candles(client: CTraderMCPClient, config: dict) -> list[dict]:
    """Fetch intraday candles for the current session."""
    now = datetime.utcnow()
    session_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    args = {
        "symbolName": config["symbol"],
        "timeframe": config["candle_timeframe"],
        "from": session_start.isoformat(),
        "to": now.isoformat(),
    }
    log.debug("Fetching intraday candles: %s", args)
    raw = await client.call(TOOL_NAMES["historical_candles"], args)
    candles = raw if isinstance(raw, list) else raw.get("bars", [])
    log.info("Fetched %d intraday candle(s) for %s", len(candles), config["symbol"])
    return candles


def format_intraday_summary(candles: list[dict]) -> str:
    """Create a compact text summary of intraday OHLCV candles for the LLM prompt."""
    import pandas as pd
    if not candles:
        return "No intraday candles available."
    df = pd.DataFrame(candles)
    df.columns = [c.lower() for c in df.columns]
    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df.get("high", close), errors="coerce")
    low = pd.to_numeric(df.get("low", close), errors="coerce")
    volume = pd.to_numeric(df.get("volume", pd.Series(1.0, index=close.index)), errors="coerce")
    lines = [f"Intraday {CONFIG['candle_timeframe']} candles (last {len(candles)}):"]
    lines.append(f"  Open: {float(close.iloc[0]):.2f} | Close: {float(close.iloc[-1]):.2f}")
    lines.append(f"  High: {float(high.max()):.2f} | Low: {float(low.min()):.2f}")
    lines.append(f"  Range: {float(high.max() - low.min()):.2f}")
    last = len(close) - 1
    for i in range(max(0, last - 4), last + 1):
        t = df.iloc[i].get("time") or df.iloc[i].get("timestamp") or f"bar_{i}"
        lines.append(f"  {t}: O={float(close.iloc[i]):.2f} H={float(high.iloc[i]):.2f} L={float(low.iloc[i]):.2f} V={float(volume.iloc[i]):.2f}")
    return "\n".join(lines)


def ask_ollama_for_strategy(model: str, symbol: str, indicators: dict) -> dict:
    """Ask the local model for a rule-based strategy given the indicator summary."""
    system_prompt = (
        "You are a cautious quantitative trading assistant. Given summarized market "
        "indicators, propose ONE simple, rule-based intraday strategy. "
        "Respond ONLY with valid JSON, no prose, no markdown fences, matching this schema:\n"
        '{"bias": "long"|"short"|"neutral", '
        '"reasoning": "short string", '
        '"entry_rule": "short string describing the price condition to enter", '
        '"invalidate_if": "short string describing when to skip/cancel", '
        '"confidence": 0-1 float}'
    )
    user_prompt = f"Symbol: {symbol}\n6-month indicator summary:\n{json.dumps(indicators, indent=2)}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    log.debug("Ollama strategy request → model=%s, messages=%s, options={temperature=0.1}", model, json.dumps(messages, indent=2))
    try:
        response = ollama.chat(
            model=model,
            messages=messages,
            options={"temperature": 0.1},
        )
    except Exception:
        log.exception("Ollama strategy request failed")
        return {"bias": "neutral", "reasoning": "ollama request failed", "confidence": 0.0}
    log.debug("Ollama strategy response ← raw=%s", repr(response))
    content = response["message"]["content"].strip()
    log.debug("Ollama strategy response ← content=%s", content)
    content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        result = json.loads(content)
        log.debug("Ollama strategy parsed result: %s", result)
        return result
    except json.JSONDecodeError:
        log.warning("Model did not return valid JSON, got:\n%s", content)
        return {"bias": "neutral", "reasoning": "unparseable model output", "confidence": 0.0}


def ask_ollama_for_live_signal(model: str, strategy: dict, symbol: str, current_price: float, intraday_candles: list[dict] = None) -> str:
    """Ask the model whether the strategy's entry condition is currently met. Returns BUY/SELL/HOLD."""
    system_prompt = (
        "You monitor a live price against a pre-agreed strategy and output exactly one word: "
        "BUY, SELL, or HOLD. Nothing else."
    )
    intraday_summary = ""
    if intraday_candles:
        intraday_summary = f"\nRecent intraday candles:\n{format_intraday_summary(intraday_candles)}"
    user_prompt = (
        f"Strategy: {json.dumps(strategy)}\n"
        f"Symbol: {symbol}\nCurrent price: {current_price}\n"
        f"{intraday_summary}"
        "Decision (one word only):"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    log.debug("Ollama live signal request → model=%s, messages=%s, options={temperature=0.0}", model, json.dumps(messages, indent=2))
    try:
        response = ollama.chat(
            model=model,
            messages=messages,
            options={"temperature": 0.0},
        )
    except Exception:
        log.exception("Ollama live signal request failed")
        return "HOLD"
    log.debug("Ollama live signal response ← raw=%s", repr(response))
    content = response["message"]["content"].strip()
    log.debug("Ollama live signal response ← content=%s", content)
    word = content.upper()
    for signal in ("BUY", "SELL", "HOLD"):
        if signal in word:
            log.debug("Ollama live signal resolved to: %s", signal)
            return signal
    log.debug("Ollama live signal resolved to: HOLD (no match in response)")
    return "HOLD"


async def ask_ollama_for_position_management(
    model: str, strategy: dict, symbol: str, current_price: float,
    positions: list[dict], pending_orders: list[dict] = None,
    intraday_candles: list[dict] = None,
) -> str:
    """Ask the model what to do with an existing open position.
    Returns TRAIL_SL, CLOSE, or HOLD."""
    system_prompt = (
        "You monitor an open trading position against its original strategy. "
        "Based on the current price, position details, and market conditions, "
        "decide the best action. Output exactly one word: TRAIL_SL, CLOSE, or HOLD. Nothing else."
    )
    intraday_summary = ""
    if intraday_candles:
        intraday_summary = f"\nRecent intraday candles:\n{format_intraday_summary(intraday_candles)}"
    pending_summary = ""
    if pending_orders:
        pending_summary = f"\nPending orders waiting to fill:\n{json.dumps(pending_orders, indent=2)}"
    positions_summary = json.dumps(positions, indent=2)
    user_prompt = (
        f"Strategy: {json.dumps(strategy)}\n"
        f"Symbol: {symbol}\n"
        f"Current price: {current_price}\n"
        f"Open positions:\n{positions_summary}"
        f"{pending_summary}"
        f"{intraday_summary}"
        "Decision (one word only - TRAIL_SL to move stop loss, CLOSE to close position, HOLD to do nothing):"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    log.debug("Ollama position management request → model=%s, messages=%s, options={temperature=0.0}", model, json.dumps(messages, indent=2))
    try:
        response = ollama.chat(
            model=model,
            messages=messages,
            options={"temperature": 0.0},
        )
    except Exception:
        log.exception("Ollama position management request failed")
        return "HOLD"
    log.debug("Ollama position management response ← raw=%s", repr(response))
    content = response["message"]["content"].strip()
    log.debug("Ollama position management response ← content=%s", content)
    word = content.upper()
    for signal in ("TRAIL_SL", "CLOSE", "HOLD"):
        if signal in word:
            log.debug("Ollama position management resolved to: %s", signal)
            return signal
    log.debug("Ollama position management resolved to: HOLD (no match in response)")
    return "HOLD"


def ask_ollama_for_sl_tp_recommendation(
    model: str, symbol: str, current_price: float, position: dict,
    strategy: dict, intraday_candles: list[dict] = None,
) -> dict:
    """Ask the model for recommended stop-loss and take-profit levels.
    Returns dict with stop_loss, take_profit, and reasoning, or None on failure."""
    system_prompt = (
        "You are a risk management assistant for trading. "
        "Given a new position and current market conditions, recommend optimal "
        "stop-loss and take-profit absolute price levels. "
        "Respond ONLY with valid JSON, no prose, no markdown fences, matching this schema:\n"
        '{"stop_loss": 5800.50, "take_profit": 5900.25, "reasoning": "short string"}'
    )
    intraday_summary = ""
    if intraday_candles:
        intraday_summary = f"\nRecent intraday candles:\n{format_intraday_summary(intraday_candles)}"

    pos_side = position.get("type", position.get("side", "Buy"))
    entry_price = position.get("entryPrice", position.get("openPrice", current_price))
    volume = position.get("volume", position.get("currentVolume", 0))

    user_prompt = (
        f"Symbol: {symbol}\n"
        f"Position: {pos_side} {volume} lots @ {entry_price}\n"
        f"Strategy: {json.dumps(strategy)}\n"
        f"Current price: {current_price}\n"
        f"{intraday_summary}"
        "Recommend stop-loss and take-profit absolute price levels:"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    log.debug("Ollama SL/TP recommendation request → model=%s, messages=%s, options={temperature=0.1}", model, json.dumps(messages, indent=2))
    try:
        response = ollama.chat(
            model=model,
            messages=messages,
            options={"temperature": 0.1},
        )
    except Exception:
        log.exception("Ollama SL/TP recommendation request failed")
        return None
    log.debug("Ollama SL/TP recommendation response ← raw=%s", repr(response))
    content = response["message"]["content"].strip()
    log.debug("Ollama SL/TP recommendation response ← content=%s", content)
    content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        result = json.loads(content)
        sl = result.get("stop_loss")
        tp = result.get("take_profit")
        reasoning = result.get("reasoning", "")
        if sl is None or tp is None:
            log.warning("Ollama SL/TP recommendation missing stop_loss or take_profit: %s", result)
            return None
        log.info("Ollama SL/TP recommendation: SL=%.2f, TP=%.2f, reasoning=%s", sl, tp, reasoning)
        return {"stop_loss": float(sl), "take_profit": float(tp), "reasoning": reasoning}
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        log.warning("Model did not return valid JSON for SL/TP, got:\n%s\nError: %s", content, e)
        return None


async def _apply_ollama_sl_tp_once(
    client: CTraderMCPClient, config: dict, position_id: str,
    current_price: float, strategy: dict, intraday_candles: list[dict],
    processed_set: set,
):
    """Fetch position details and apply one-time Ollama SL/TP recommendation."""
    if position_id in processed_set:
        return
    updated_positions = await client.call(TOOL_NAMES["list_positions"], {})
    if not isinstance(updated_positions, list):
        updated_positions = updated_positions.get("positions", updated_positions.get("items", [])) if isinstance(updated_positions, dict) else []

    new_position = None
    for pos in updated_positions:
        if str(pos.get("id", pos.get("positionId", ""))) == str(position_id):
            new_position = pos
            break

    if not new_position:
        log.warning("Could not find newly placed position %s for Ollama SL/TP recommendation", position_id)
        processed_set.add(position_id)
        return

    sl_tp = ask_ollama_for_sl_tp_recommendation(
        config["ollama_model"], config["symbol"], current_price,
        new_position, strategy, intraday_candles,
    )
    if sl_tp:
        if config["dry_run"]:
            log.info("[DRY RUN] Would amend position %s with SL=%.2f, TP=%.2f", position_id, sl_tp["stop_loss"], sl_tp["take_profit"])
        else:
            await client.call(TOOL_NAMES["amend_position"], {
                "Id": position_id,
                "stopLoss": round(sl_tp["stop_loss"], 2),
                "takeProfit": round(sl_tp["take_profit"], 2),
            })
            log.info("Amended position %s with Ollama SL/TP: SL=%.2f, TP=%.2f", position_id, sl_tp["stop_loss"], sl_tp["take_profit"])
    processed_set.add(position_id)


# ============================================================
# TRADE EXECUTION
# ============================================================

async def place_trade(client: CTraderMCPClient, side: str, config: dict):
    sl_pips = config["stop_loss_points"]
    volume = _calculate_trade_volume(sl_pips, config)

    if volume <= 0:
        log.warning("Calculated trade volume is %.2f — skipping trade", volume)
        return None

    order_args = {
        "symbolName": config["symbol"],
        "side": "buy" if side == "BUY" else "sell",
        "volume": volume,
        "volumeType": "lots",
        "stopLossPips": round(sl_pips, 2),
        "takeProfitPips": round(config["take_profit_points"], 2),
    }

    log.debug("Preparing order with args: %s", order_args)

    if config["dry_run"]:
        log.info("[DRY RUN] Would place order: %s", order_args)
        return {"dry_run": True, "order": order_args}

    log.warning("Placing LIVE order: %s", order_args)
    result = await client.call(TOOL_NAMES["place_order"], order_args)
    log.info("Order result: %s", result)

    # Cache the volume by position ID so event logging can use it if MCP returns 0
    if isinstance(result, dict):
        position_id = result.get("id")
        if position_id is not None:
            _trade_volumes[str(position_id)] = volume

    return result


EVENTS_LOG_FILE = "events.json"

# Cache for trade volumes sent to the MCP server, keyed by position ID.
# Used to populate volume in events.json when the MCP returns 0 or missing volume.
_trade_volumes: dict = {}


def _ms_timestamp() -> int:
    return int(datetime.utcnow().timestamp() * 1000)


def _calculate_trade_volume(sl_pips: float, config: dict) -> float:
    """Calculate position size based on risk percentage and account balance."""
    volume = calculate_position_size(
        sl_pips=sl_pips,
        risk_pct=config.get("trade_risk_pct", config.get("vp_base_risk_pct", 2.0)),
        balance=config.get("trade_balance", config.get("vp_balance", 10000.0)),
        pip_value_per_unit=config.get("trade_pip_value_per_unit", config.get("vp_pip_value_per_unit", 1.0)),
        max_volume_lots=config.get("trade_max_volume_lots", config.get("vp_max_volume_lots", 10.0)),
        min_trade_volume=config.get("trade_min_volume", config.get("vp_min_trade_volume", 0.1)),
        volume_step=config.get("trade_volume_step", config.get("vp_volume_step", 0.1)),
        volume_min_units=config.get("trade_volume_min_units", config.get("vp_volume_min_units", 0.01)),
        volume_max_units=config.get("trade_volume_max_units", config.get("vp_volume_max_units", 100.0)),
    )
    return volume


def _last_serial() -> int:
    if os.path.exists(EVENTS_LOG_FILE):
        try:
            with open(EVENTS_LOG_FILE, "r") as f:
                events = json.load(f)
            if events:
                return max(e.get("serial", 0) for e in events)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    return -1


def _coerce_int_time(val: Any, fallback: int) -> int:
    """Ensure a timestamp value is an integer (milliseconds)."""
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
            return int(dt.timestamp() * 1000)
        except (ValueError, TypeError):
            return fallback
    return fallback


def _position_to_event(position: dict, serial: int, event_type: str = "Create Position") -> dict:
    pos_type = position.get("type", position.get("side", "Buy"))
    if isinstance(pos_type, str):
        pos_type = pos_type.capitalize()
    if pos_type not in ("Buy", "Sell"):
        pos_type = "Buy"

    pos_id = str(position.get("id", position.get("positionId", 0)))
    cached_volume = _trade_volumes.get(pos_id)
    raw_volume = position.get("volume", position.get("currentVolume", cached_volume if cached_volume is not None else 0))
    volume = raw_volume if raw_volume is not None else (cached_volume if cached_volume is not None else 0)

    return {
        "serial": serial,
        "orderId": None,
        "positionId": pos_id,
        "event": event_type,
        "time": _ms_timestamp(),
        "volume": volume,
        "quantity": volume,
        "type": pos_type,
        "entryPrice": position.get("entryPrice", position.get("openPrice", 0)),
        "tp": position.get("takeProfit", position.get("tp", None)),
        "sl": position.get("stopLoss", position.get("sl", None)),
        "closePrice": None,
        "grossProfit": 0,
        "pips": 0,
        "balance": None,
        "equity": None,
    }


async def log_trades(client: CTraderMCPClient, config: dict):
    """Fetch current trade state from cTrader MCP and persist as cTrader events.json format."""
    existing_events = []
    if os.path.exists(EVENTS_LOG_FILE):
        try:
            with open(EVENTS_LOG_FILE, "r") as f:
                existing_events = json.load(f)
        except (json.JSONDecodeError, ValueError):
            existing_events = []

    next_serial = _last_serial() + 1
    new_events = []
    now_ms = _ms_timestamp()

    # Fetch open positions and create/create-modify events
    try:
        positions = await client.call(TOOL_NAMES["list_positions"], {})
        if not isinstance(positions, list):
            positions = positions.get("positions", positions.get("items", [])) if isinstance(positions, dict) else []
    except Exception:
        log.exception("Failed to fetch positions from MCP server")
        positions = []

    # Fetch pending orders
    try:
        pending = await client.call(TOOL_NAMES["list_pending_orders"], {})
        if not isinstance(pending, list):
            pending = pending.get("pendingOrders", pending.get("items", [])) if isinstance(pending, dict) else []
    except Exception:
        log.exception("Failed to fetch pending orders from MCP server")
        pending = []

    # Filter to US500
    positions = [
        p for p in positions
        if str(p.get("symbol", p.get("symbolName", ""))).upper() == "US500"
    ]
    pending = [
        o for o in pending
        if str(o.get("symbol", o.get("symbolName", ""))).upper() == "US500"
    ]

    # Build events for positions
    for pos in positions:
        pos_id = pos.get("id", pos.get("positionId", 0))
        # Check if this position already exists in our events
        existing_create_ids = {
            str(e.get("positionId"))
            for e in existing_events
            if e.get("event") == "Create Position"
        }
        if str(pos_id) not in existing_create_ids:
            # New position — CREATE event
            new_events.append(_position_to_event(pos, next_serial, "Create Position"))
            next_serial += 1
        else:
            # Existing position — check if SL/TP was modified
            existing_pos_event = next(
                (
                    e
                    for e in reversed(existing_events)
                    if str(e.get("positionId")) == str(pos_id)
                    and e.get("event") == "Create Position"
                ),
                None,
            )
            if existing_pos_event is not None:
                existing_sl = existing_pos_event.get("sl")
                current_sl = pos.get("stopLoss", pos.get("sl", None))
                if current_sl is not None and existing_sl != current_sl:
                    new_events.append(_position_to_event(pos, next_serial, "Position Modified (S/L)"))
                    next_serial += 1

    # Build events for pending orders (treat as position creates)
    for order in pending:
        order_id = order.get("id", order.get("orderId", 0))
        existing_pending_ids = {str(e.get("positionId")) for e in existing_events}
        if str(order_id) not in existing_pending_ids:
            new_events.append(_position_to_event(order, next_serial, "Create Position"))
            next_serial += 1

    # Fetch deals (closed/filled trades)
    try:
        deals = await client.call(TOOL_NAMES["list_deals"], {})
        if not isinstance(deals, list):
            deals = deals.get("deals", deals.get("items", [])) if isinstance(deals, dict) else []
    except Exception:
        log.exception("Failed to fetch deals from MCP server")
        deals = []

    deals = [
        d for d in deals
        if str(d.get("symbol", d.get("symbolName", ""))).upper() == "US500"
    ]

    for deal in deals:
        deal_id = deal.get("id", deal.get("dealId", 0))
        if str(deal_id) not in {str(e.get("serial")) for e in existing_events}:
            # Determine event type based on deal data
            close_price = deal.get("closePrice", deal.get("closePrice", None))
            profit = deal.get("profit", deal.get("grossProfit", 0))
            pips = deal.get("pips", 0)

            if close_price is None:
                event_type = "Create Position"
            elif profit < 0:
                event_type = "Stop Loss Hit"
            else:
                event_type = "Position closed"

            pos_type = deal.get("type", deal.get("side", "Buy"))
            if isinstance(pos_type, str):
                pos_type = pos_type.capitalize()
            if pos_type not in ("Buy", "Sell"):
                pos_type = "Buy"

            # Get the positionId from deal or use a fallback
            pos_id = deal.get("positionId", deal.get("position", 0)) or deal_id

            event = {
                "serial": next_serial,
                "orderId": None,
                "positionId": pos_id,
                "event": event_type,
                "time": _coerce_int_time(deal.get("time", deal.get("closeTime", now_ms)), now_ms),
                "volume": deal.get("volume", deal.get("quantity", 0)),
                "quantity": deal.get("volume", deal.get("quantity", 0)),
                "type": pos_type,
                "entryPrice": deal.get("entryPrice", deal.get("openPrice", 0)),
                "tp": deal.get("takeProfit", deal.get("tp", None)),
                "sl": deal.get("stopLoss", deal.get("sl", None)),
                "closePrice": close_price,
                "grossProfit": profit,
                "pips": pips,
                "balance": deal.get("balance", None),
                "equity": deal.get("equity", None),
            }
            new_events.append(event)
            next_serial += 1

    # Fetch order history (closed orders)
    try:
        history = await client.call(TOOL_NAMES["list_order_history"], {})
        if not isinstance(history, list):
            history = history.get("history", history.get("items", [])) if isinstance(history, dict) else []
    except Exception:
        log.exception("Failed to fetch order history from MCP server")
        history = []

    history = [
        h for h in history
        if str(h.get("symbol", h.get("symbolName", ""))).upper() == "US500"
    ]

    for h_item in history:
        h_id = h_item.get("id", h_item.get("orderId", 0))
        if str(h_id) not in {str(e.get("serial")) for e in existing_events}:
            close_price = h_item.get("closePrice", h_item.get("closePrice", None))
            profit = h_item.get("profit", h_item.get("grossProfit", 0))
            pips = h_item.get("pips", 0)

            if close_price is None:
                event_type = "Create Position"
            elif profit < 0:
                event_type = "Stop Loss Hit"
            else:
                event_type = "Position closed"

            pos_type = h_item.get("type", h_item.get("side", "Buy"))
            if isinstance(pos_type, str):
                pos_type = pos_type.capitalize()
            if pos_type not in ("Buy", "Sell"):
                pos_type = "Buy"

            pos_id = h_item.get("positionId", h_item.get("position", 0)) or h_id

            event = {
                "serial": next_serial,
                "orderId": None,
                "positionId": pos_id,
                "event": event_type,
                "time": _coerce_int_time(h_item.get("time", h_item.get("closeTime", now_ms)), now_ms),
                "volume": h_item.get("volume", h_item.get("quantity", 0)),
                "quantity": h_item.get("volume", h_item.get("quantity", 0)),
                "type": pos_type,
                "entryPrice": h_item.get("entryPrice", h_item.get("openPrice", 0)),
                "tp": h_item.get("takeProfit", h_item.get("tp", None)),
                "sl": h_item.get("stopLoss", h_item.get("sl", None)),
                "closePrice": close_price,
                "grossProfit": profit,
                "pips": pips,
                "balance": h_item.get("balance", None),
                "equity": h_item.get("equity", None),
            }
            new_events.append(event)
            next_serial += 1

    if new_events:
        all_events = existing_events + new_events
        with open(EVENTS_LOG_FILE, "w") as f:
            json.dump(all_events, f, indent=2, default=str)
        log.info("Wrote %d event(s) to %s (total: %d)", len(new_events), EVENTS_LOG_FILE, len(all_events))


# ============================================================
# VOLUME PROFILE STRATEGY WORKFLOW
# ============================================================


async def _build_market_context(client: CTraderMCPClient, vp: dict, config: dict) -> Optional[MarketContext]:
    """Fetch the live price and combine it with the volume profile data."""
    quote = await client.call(TOOL_NAMES["current_price"], {"symbolName": config["symbol"]})
    log.debug("Raw price quote for MarketContext: %s", repr(quote)[:1000])
    price = quote.get("bid") or quote.get("price") if isinstance(quote, dict) else quote
    if price is None:
        log.warning("Could not determine live price from quote: %s", quote)
        return None
    return MarketContext(
        price=float(price),
        poc=vp.get("poc", 0.0),
        vah=vp.get("vah", 0.0),
        val=vp.get("val", 0.0),
        prev_close=vp.get("prev_close", 0.0),
        prev_high=vp.get("prev_high", 0.0),
        prev_low=vp.get("prev_low", 0.0),
    )


async def run_volume_profile_strategy(client: CTraderMCPClient, config: dict, candles: list[dict]):
    """
    Volume profile strategy workflow.

    1. Compute volume profile from historical candles.
    2. In the polling loop, fetch the live price, determine the schema,
       check entry conditions, and place a trade if valid.
    3. Also supports magnet mode (mean reversion) when no position is active.
    """
    # --- Step 1: compute volume profile from historical data ---
    log.info("Computing volume profile from %d candle(s) ...", len(candles))
    vp = compute_volume_profile(
        candles,
        bin_size=config.get("vp_bin_size"),
        value_area_pct=config.get("vp_value_area_pct", 0.70),
    )
    log.info("Volume profile — POC: %.2f | VAH: %.2f | VAL: %.2f", vp["poc"], vp["vah"], vp["val"])
    log.debug("Volume profile full result: %s", vp)

    if vp["poc"] == 0.0 or vp["vah"] == 0.0 or vp["val"] == 0.0:
        log.error("Volume profile computation returned zero values — check candle data.")
        return

    # Pre-compute RSI and Bollinger Bands from candles for magnet mode fallback
    import pandas as pd
    df = pd.DataFrame(candles)
    df = df.rename(columns={c: c.lower() for c in df.columns})
    close = df["close"].astype(float)
    rsi_value = None
    bb_upper = None
    bb_lower = None
    if len(close) >= 20:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, float("nan"))
        rsi_series = 100 - (100 / (1 + rs))
        rsi_value = round(float(rsi_series.iloc[-1]), 2) if not rsi_series.isna().all() else None
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        bb_upper = round(float((sma20 + 2.0 * std20).iloc[-1]), 2) if not sma20.isna().all() else None
        bb_lower = round(float((sma20 - 2.0 * std20).iloc[-1]), 2) if not sma20.isna().all() else None
        log.debug("Magnet indicators — RSI: %s, BB upper: %s, BB lower: %s", rsi_value, bb_upper, bb_lower)

    # --- Step 2: live monitoring loop ---
    iterations = 0
    active_position_id: Optional[str] = None
    ollama_sl_tp_processed = set()

    while True:
        log.debug("--- Volume Profile Polling iteration %d ---", iterations + 1)

        # Check open positions
        positions = await client.call(TOOL_NAMES["list_positions"], {})
        log.debug("Raw positions response: %s", repr(positions)[:1000])
        open_count = len(positions) if isinstance(positions, list) else len(positions.get("positions", positions.get("items", [])))
        log.info("Open positions: %d / %d", open_count, config["max_open_positions"])

        if open_count > config["max_open_positions"]:
            for pos in positions[config["max_open_positions"]:]:
                pos_id = pos.get("id", pos.get("positionId", "unknown"))
                log.warning("Closing extra position %s to respect max_open_positions=%d", pos_id, config["max_open_positions"])
                if config["dry_run"]:
                    log.info("[DRY RUN] Would close position %s", pos_id)
                else:
                    await client.call(TOOL_NAMES["close_position"], {"Id": pos_id})
                    log.info("Closed extra position %s", pos_id)

        if open_count >= config["max_open_positions"]:
            log.info("Max open positions reached (%s). Checking position management.", open_count)
            quote_pos = await client.call(TOOL_NAMES["current_price"], {"symbolName": config["symbol"]})
            price_pos = quote_pos.get("bid") or quote_pos.get("price") if isinstance(quote_pos, dict) else quote_pos
            log.info("Current %s price: %s (position management)", config["symbol"], price_pos)

            pending_orders = []
            try:
                po = await client.call(TOOL_NAMES["list_pending_orders"], {})
                if isinstance(po, list):
                    pending_orders = po
                elif isinstance(po, dict):
                    pending_orders = po.get("pendingOrders", po.get("items", []))
            except Exception:
                log.exception("Failed to fetch pending orders for position management")

            intraday_candles = await fetch_intraday_candles(client, config)

            mgmt_signal = await ask_ollama_for_position_management(
                config["ollama_model"], {}, config["symbol"], price_pos,
                positions, pending_orders, intraday_candles,
            )
            log.info("Position management signal: %s", mgmt_signal)

            if mgmt_signal == "TRAIL_SL":
                log.info("Trailing stop loss for position")
                if not config["dry_run"] and positions:
                    for pos in positions:
                        pos_id = pos.get("id", pos.get("positionId", "unknown"))
                        current_sl = pos.get("stopLoss", pos.get("sl", 0))
                        pos_side = pos.get("type", pos.get("side", "Buy"))
                        new_sl = price_pos - config["stop_loss_points"] if pos_side == "Buy" else price_pos + config["stop_loss_points"]
                        if pos_side == "Buy" and new_sl > current_sl or pos_side != "Buy" and new_sl < current_sl:
                            log.info("Updating SL for position %s from %.2f to %.2f", pos_id, current_sl, new_sl)
                            await client.call(TOOL_NAMES["amend_position"], {
                                "Id": pos_id,
                                "stopLoss": round(new_sl, 2),
                            })
                        else:
                            log.debug("SL trail skipped for position %s — not favorable (current SL=%.2f, proposed=%.2f)", pos_id, current_sl, new_sl)
            elif mgmt_signal == "CLOSE":
                log.info("Closing position per model decision")
                if not config["dry_run"] and positions:
                    for pos in positions:
                        pos_id = pos.get("id", pos.get("positionId", "unknown"))
                        result = await client.call(TOOL_NAMES["close_position"], {"Id": pos_id})
                        log.info("Position %s closed: %s", pos_id, result)
            else:
                log.debug("Position management signal is HOLD — no action taken.")

        else:
            # Build market context from live price + volume profile
            ctx = await _build_market_context(client, vp, config)
            if ctx is None:
                log.warning("Could not build market context — skipping this iteration.")
            else:
                schema = determine_schema(
                    ctx.price, ctx.poc, ctx.vah, ctx.val,
                    proximity_pips=config.get("vp_zone_proximity_pips", 5.0),
                )
                direction = get_schema_direction(schema)
                log.info(
                    "Schema %d | Direction: %s | Price: %.2f | VAH: %.2f | VAL: %.2f | POC: %.2f",
                    schema, direction, ctx.price, ctx.vah, ctx.val, ctx.poc,
                )

                # Try schema-based entry
                vp_config = {
                    "pip_size": config.get("vp_pip_size", 0.01),
                    "pip_value_per_unit": config.get("vp_pip_value_per_unit", 1.0),
                    "balance": config.get("vp_balance", 10000.0),
                    "base_risk_pct": config.get("vp_base_risk_pct", 2.0),
                    "reduced_risk_pct": config.get("vp_reduced_risk_pct", 0.5),
                    "max_volume_lots": config.get("vp_max_volume_lots", 10.0),
                    "min_trade_volume": config.get("vp_min_trade_volume", 0.1),
                    "volume_step": config.get("vp_volume_step", 0.1),
                    "volume_min_units": config.get("vp_volume_min_units", 0.01),
                    "volume_max_units": config.get("vp_volume_max_units", 100.0),
                    "min_risk_distance_pips": config.get("vp_min_risk_distance_pips", 30.0),
                    "max_schema_entry_distance_pips": config.get("vp_max_schema_entry_distance_pips", 15.0),
                    "max_sl_pips": config.get("vp_max_sl_pips", 20.0),
                    "min_rr_ratio": config.get("vp_min_rr_ratio", 1.2),
                    "enable_magnet_mode": config.get("enable_magnet_mode", True),
                    "magnet_start_hour": config.get("magnet_start_hour", 17),
                    "rsi_overbought": config.get("rsi_overbought", 70),
                    "rsi_oversold": config.get("rsi_oversold", 30),
                }

                result = evaluate_entry(ctx, schema, vp_config)

                if result is not None and active_position_id is None:
                    log.info("Volume profile strategy signals %s — placing trade.", result.direction)
                    if config["dry_run"]:
                        log.info("[DRY RUN] Volume profile trade: %s", json.dumps({
                            "schema": result.schema,
                            "direction": result.direction,
                            "entry": result.entry,
                            "sl": result.sl,
                            "targets": result.targets,
                            "volume": calculate_position_size(
                                sl_pips=result.risk_pips,
                                risk_pct=vp_config["base_risk_pct"],
                                balance=vp_config["balance"],
                                pip_value_per_unit=vp_config["pip_value_per_unit"],
                                max_volume_lots=vp_config["max_volume_lots"],
                                min_trade_volume=vp_config["min_trade_volume"],
                                volume_step=vp_config["volume_step"],
                                volume_min_units=vp_config["volume_min_units"],
                                volume_max_units=vp_config["volume_max_units"],
                            ),
                        }, indent=2))
                    else:
                        volume = calculate_position_size(
                            sl_pips=result.risk_pips,
                            risk_pct=vp_config["base_risk_pct"],
                            balance=vp_config["balance"],
                            pip_value_per_unit=vp_config["pip_value_per_unit"],
                            max_volume_lots=vp_config["max_volume_lots"],
                            min_trade_volume=vp_config["min_trade_volume"],
                            volume_step=vp_config["volume_step"],
                            volume_min_units=vp_config["volume_min_units"],
                            volume_max_units=vp_config["volume_max_units"],
                        )
                        log.warning("Placing LIVE volume profile order: %s %s @ %.2f, SL=%.2f, TP=%.2f, Volume=%.2f",
                                    result.direction, config["symbol"], result.entry, result.sl, result.targets[0], volume)
                        order_args = {
                            "symbolName": config["symbol"],
                            "side": "buy" if result.direction == "BUY" else "sell",
                            "volume": volume,
                            "volumeType": "lots",
                            "stopLossPips": round(result.risk_pips, 2),
                            "takeProfitPips": round(result.tp1_pips, 2),
                        }
                        trade_result = await client.call(TOOL_NAMES["place_order"], order_args)
                        log.info("Volume profile order result: %s", trade_result)
                        active_position_id = str(trade_result.get("id", "unknown"))
                        if active_position_id != "unknown":
                            _trade_volumes[active_position_id] = volume
                        intraday_candles_vp = await fetch_intraday_candles(client, config)
                        await _apply_ollama_sl_tp_once(
                            client, config, active_position_id, ctx.price,
                            {}, intraday_candles_vp, ollama_sl_tp_processed,
                        )
                elif result is None and active_position_id is None:
                    # Try magnet mode as fallback
                    log.debug("Schema entry not valid — checking magnet mode.")
                    magnet_result = try_magnet_trade(ctx, rsi_value, bb_upper, bb_lower, vp_config)
                    if magnet_result is not None:
                        log.info("Magnet mode signals %s — placing trade.", magnet_result.direction)
                        if config["dry_run"]:
                            log.info("[DRY RUN] Magnet trade: %s", json.dumps({
                                "direction": magnet_result.direction,
                                "entry": magnet_result.entry,
                                "sl": magnet_result.sl,
                                "targets": magnet_result.targets,
                            }, indent=2))
                        else:
                            volume = calculate_position_size(
                            sl_pips=magnet_result.risk_pips,
                            risk_pct=vp_config["base_risk_pct"],
                            balance=vp_config["balance"],
                            pip_value_per_unit=vp_config["pip_value_per_unit"],
                            max_volume_lots=vp_config["max_volume_lots"],
                            min_trade_volume=vp_config["min_trade_volume"],
                            volume_step=vp_config["volume_step"],
                            volume_min_units=vp_config["volume_min_units"],
                            volume_max_units=vp_config["volume_max_units"],
                            )
                            log.warning("Placing LIVE magnet order: %s %s @ %.2f, SL=%.2f, TP=%.2f, Volume=%.2f",
                            magnet_result.direction, config["symbol"], ctx.price, magnet_result.sl, magnet_result.targets[0], volume)
                            order_args = {
                            "symbolName": config["symbol"],
                            "side": "buy" if magnet_result.direction == "BUY" else "sell",
                            "volume": volume,
                            "volumeType": "lots",
                            "stopLossPips": round(magnet_result.risk_pips, 2),
                            "takeProfitPips": round(magnet_result.tp1_pips, 2),
                            }
                            trade_result = await client.call(TOOL_NAMES["place_order"], order_args)
                            log.info("Magnet order result: %s", trade_result)
                            active_position_id = str(trade_result.get("id", "unknown"))
                            if active_position_id != "unknown":
                                    _trade_volumes[active_position_id] = volume
                            intraday_candles_mag = await fetch_intraday_candles(client, config)
                            await _apply_ollama_sl_tp_once(
                            client, config, active_position_id, ctx.price,
                            {}, intraday_candles_mag, ollama_sl_tp_processed,
                            )

        iterations += 1
        log.debug("Iteration %d of max_loop_iterations=%d complete.", iterations, config["max_loop_iterations"])
        if config["max_loop_iterations"] and iterations >= config["max_loop_iterations"]:
            log.info("Reached max_loop_iterations (%d), stopping.", config["max_loop_iterations"])
            break
        log.debug("Sleeping for %d seconds before next poll...", config["poll_interval_seconds"])
        await log_trades(client, config)
        await asyncio.sleep(config["poll_interval_seconds"])


# ============================================================
# MAIN WORKFLOW
# ============================================================

async def main():
    config = CONFIG
    log.debug("Starting AI trader with config: strategy=%s, symbol=%s, timeframe=%s, history_months=%s, dry_run=%s, max_loop_iterations=%d, poll_interval=%d",
              config["strategy"], config["symbol"], config["candle_timeframe"], config["history_months"],
              config["dry_run"], config["max_loop_iterations"], config["poll_interval_seconds"])
    client = CTraderMCPClient(config)
    await client.connect()

    try:
        # --- Step 1: pull last 6 months of history ---
        since = (datetime.utcnow() - timedelta(days=30 * config["history_months"])).isoformat()
        history_args = {
            "symbolName": config["symbol"],
            "timeframe": config["candle_timeframe"],
            "from": since,
            "to": datetime.utcnow().isoformat(),
        }
        log.info("Fetching %s months of history for %s ...", config["history_months"], config["symbol"])
        log.debug("History args: %s", history_args)
        candles_raw = await client.call(TOOL_NAMES["historical_candles"], history_args)
        log.debug("Raw MCP response for historical_candles: %s", repr(candles_raw)[:2000])
        candles = candles_raw if isinstance(candles_raw, list) else candles_raw.get("bars", [])
        if not candles:
            log.error("No candle data returned — check TOOL_NAMES['historical_candles'] and its arguments.")
            return

        log.info(
            "Received %d candle(s) for %s (timeframe=%s). Range: %s to %s",
            len(candles),
            config["symbol"],
            config["candle_timeframe"],
            candles[0].get("time") or candles[0].get("timestamp") or candles[0].get("date"),
            candles[-1].get("time") or candles[-1].get("timestamp") or candles[-1].get("date"),
        )
        log.debug("First candle: %s", candles[0])
        log.debug("Last candle: %s", candles[-1])
        log.debug("Sample candles (first 3): %s", candles[:3])

        # --- Strategy dispatch ---
        if config["strategy"] == "volume_profile":
            await run_volume_profile_strategy(client, config, candles)
            return

        # --- Ollama strategy (default) ---
        indicators = compute_indicators(candles)
        log.info("Indicator summary: %s", indicators)
        log.debug(
            "Indicators detail — last_close=%s, 6m_high=%s, 6m_low=%s, sma20=%s, sma50=%s, rsi14=%s, trend=%s",
            indicators["last_close"],
            indicators["6m_high"],
            indicators["6m_low"],
            indicators["sma20"],
            indicators["sma50"],
            indicators["rsi14"],
            indicators["trend_20_vs_50"],
        )

        log.debug("Calling Ollama for strategy with model=%s, symbol=%s", config["ollama_model"], config["symbol"])
        log.debug("Ollama strategy prompt indicators: %s", json.dumps(indicators, indent=2))
        strategy = ask_ollama_for_strategy(config["ollama_model"], config["symbol"], indicators)
        log.info("Proposed strategy: %s", json.dumps(strategy, indent=2))
        log.debug("Strategy bias=%s, confidence=%s", strategy.get("bias"), strategy.get("confidence"))

        if strategy.get("bias") == "neutral" or strategy.get("confidence", 0) < 0.3:
            log.info(
                "Model has low conviction (bias=%s, confidence=%s) — stopping before live monitoring. Adjust prompt or re-run later.",
                strategy.get("bias"),
                strategy.get("confidence"),
            )
            return

        # --- Step 2: watch live price and act on the strategy ---
        iterations = 0
        ollama_sl_tp_processed = set()
        while True:
            log.debug("--- Polling iteration %d ---", iterations + 1)
            positions = await client.call(TOOL_NAMES["list_positions"], {})
            log.debug("Raw positions response: %s", repr(positions)[:1000])
            open_count = len(positions) if isinstance(positions, list) else len(positions.get("positions", positions.get("items", [])))
            log.info("Open positions: %d / %d", open_count, config["max_open_positions"])

            # Close extra positions beyond max_open_positions
            if open_count > config["max_open_positions"]:
                for pos in positions[config["max_open_positions"]:]:
                    pos_id = pos.get("id", pos.get("positionId", "unknown"))
                    log.warning("Closing extra position %s to respect max_open_positions=%d", pos_id, config["max_open_positions"])
                    if config["dry_run"]:
                        log.info("[DRY RUN] Would close position %s", pos_id)
                    else:
                        await client.call(TOOL_NAMES["close_position"], {"Id": pos_id})
                        log.info("Closed extra position %s", pos_id)

            if open_count >= config["max_open_positions"]:
                # Position exists — ask Ollama about trailing SL or closing
                quote = await client.call(TOOL_NAMES["current_price"], {"symbolName": config["symbol"]})
                log.debug("Raw price quote response: %s", repr(quote)[:1000])
                price = quote.get("bid") or quote.get("price") if isinstance(quote, dict) else quote
                log.info("Current %s price: %s (position management)", config["symbol"], price)

                pending_orders = []
                try:
                    po = await client.call(TOOL_NAMES["list_pending_orders"], {})
                    if isinstance(po, list):
                        pending_orders = po
                    elif isinstance(po, dict):
                        pending_orders = po.get("pendingOrders", po.get("items", []))
                except Exception:
                    log.exception("Failed to fetch pending orders for position management")

                intraday_candles = await fetch_intraday_candles(client, config)

                log.debug("Calling Ollama for position management (model=%s)", config["ollama_model"])
                mgmt_signal = await ask_ollama_for_position_management(
                    config["ollama_model"], strategy, config["symbol"], price,
                    positions, pending_orders, intraday_candles,
                )
                log.info("Position management signal: %s", mgmt_signal)

                if mgmt_signal == "TRAIL_SL":
                    log.info("Trailing stop loss for position")
                    if not config["dry_run"] and positions:
                        for pos in positions:
                            pos_id = pos.get("id", pos.get("positionId", "unknown"))
                            current_sl = pos.get("stopLoss", pos.get("sl", 0))
                            new_sl = price - config["stop_loss_points"] if pos.get("type", pos.get("side", "Buy")) == "Buy" else price + config["stop_loss_points"]
                            if new_sl > current_sl if pos.get("type", pos.get("side", "Buy")) == "Buy" else new_sl < current_sl:
                                log.info("Updating SL for position %s from %.2f to %.2f", pos_id, current_sl, new_sl)
                                await client.call(TOOL_NAMES["amend_position"], {
                                    "Id": pos_id,
                                    "stopLoss": round(new_sl, 2),
                                })
                            else:
                                log.debug("SL trail skipped for position %s — not favorable (current SL=%.2f, proposed=%.2f)", pos_id, current_sl, new_sl)
                elif mgmt_signal == "CLOSE":
                    log.info("Closing position per model decision")
                    if not config["dry_run"] and positions:
                        for pos in positions:
                            pos_id = pos.get("id", pos.get("positionId", "unknown"))
                            result = await client.call(TOOL_NAMES["close_position"], {"Id": pos_id})
                            log.info("Position %s closed: %s", pos_id, result)
                else:
                    log.debug("Position management signal is HOLD — no action taken.")

            else:
                # No position — ask Ollama for entry signal
                quote = await client.call(TOOL_NAMES["current_price"], {"symbolName": config["symbol"]})
                log.debug("Raw price quote response: %s", repr(quote)[:1000])
                price = quote.get("bid") or quote.get("price") if isinstance(quote, dict) else quote
                log.info("Current %s price: %s", config["symbol"], price)
                log.debug("Price type: %s", type(price).__name__)

                log.debug("Calling Ollama for live signal (model=%s)", config["ollama_model"])
                intraday_candles = await fetch_intraday_candles(client, config)
                signal = ask_ollama_for_live_signal(
                    config["ollama_model"], strategy, config["symbol"], price, intraday_candles
                )
                log.info("Model signal: %s", signal)
                log.debug("Signal received: %s", signal)

                if signal in ("BUY", "SELL"):
                    log.info("Signal is %s — attempting to place trade.", signal)
                    trade_result = await place_trade(client, signal, config)
                    position_id = trade_result.get("id") if isinstance(trade_result, dict) else None
                    if position_id:
                        await _apply_ollama_sl_tp_once(
                            client, config, str(position_id), price,
                            strategy, intraday_candles, ollama_sl_tp_processed,
                        )
                else:
                    log.debug("Signal is HOLD — no trade action taken.")

            iterations += 1
            log.debug("Iteration %d of max_loop_iterations=%d complete.", iterations, config["max_loop_iterations"])
            if config["max_loop_iterations"] and iterations >= config["max_loop_iterations"]:
                log.info("Reached max_loop_iterations (%d), stopping.", config["max_loop_iterations"])
                break
            log.debug("Sleeping for %d seconds before next poll...", config["poll_interval_seconds"])
            await log_trades(client, config)
            await asyncio.sleep(config["poll_interval_seconds"])

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
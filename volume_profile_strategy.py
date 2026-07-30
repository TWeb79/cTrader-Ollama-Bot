#!/usr/bin/env python3
"""
Volume Profile Strategy — Python port of cTrader BOT14 EliteAuctionScaling.
=============================================================================

Translates the cAlgo volume profile strategy into a Python module usable
by Trader.py.  Computes POC / VAH / VAL from historical candle data,
detects one of four schemas, checks entry conditions, calculates stop-loss
and take-profit levels, and sizes the position by risk.

Schemas
-------
  2  Price above VAH  →  Short  (target: POC → VAL → PrevLow)
  5  Price in lower VA (below POC, above VAL)  →  Long  (target: POC → VAH)
  6  Price below VAL  →  Long  (target: VAL → POC → VAH → PrevHigh)
  7  Price in upper VA (above POC, below VAH)  →  Short  (target: POC → VAL)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger("ai_trader.vp_strategy")

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class MarketContext:
    price: float
    poc: float
    vah: float
    val: float
    prev_close: float
    prev_high: float
    prev_low: float


@dataclass
class SchemaResult:
    schema: int
    direction: str  # "BUY" or "SELL"
    entry: float
    sl: float
    targets: list[float]
    risk_pips: float
    tp1_pips: float
    rr_ratio: float
    reason: str


# ---------------------------------------------------------------------------
# Volume profile computation
# ---------------------------------------------------------------------------


def compute_volume_profile(
    candles: list[dict],
    bin_size: Optional[float] = None,
    value_area_pct: float = 0.70,
) -> dict:
    """
    Compute Point of Control (POC), Value Area High (VAH), and Value Area
    Low (VAL) from a list of candle dicts.

    Parameters
    ----------
    candles : list[dict]
        Each dict must contain at least ``close`` and optionally ``high``,
        ``low``, ``volume``.  Keys are case-insensitive (lowercased internally).
    bin_size : float, optional
        Price width of each histogram bin.  If None, auto-calculated as
        (max − min) / 100.
    value_area_pct : float
        Fraction of total volume that defines the value area (default 0.70).

    Returns
    -------
    dict with keys ``poc``, ``vah``, ``val``, ``prev_close``,
    ``prev_high``, ``prev_low``.  Values are 0.0 when computation fails.
    """
    df = pd.DataFrame(candles)
    df.columns = [c.lower() for c in df.columns]

    if "close" not in df.columns:
        log.error("No 'close' column in candles — cannot compute volume profile")
        return _empty_context()

    close = pd.to_numeric(df["close"], errors="coerce").dropna()
    if close.empty:
        log.error("No valid close prices — cannot compute volume profile")
        return _empty_context()

    # Use typical price (H+L+C)/3 for binning when available, else close
    if "high" in df.columns and "low" in df.columns:
        high = pd.to_numeric(df["high"], errors="coerce")
        low = pd.to_numeric(df["low"], errors="coerce")
        typical = (high + low + close) / 3.0
    else:
        typical = close

    # Volume: use candle volume if present, else count candles per bin
    has_volume = "volume" in df.columns
    if has_volume:
        volumes = pd.to_numeric(df["volume"], errors="coerce").fillna(1.0)
    else:
        volumes = pd.Series(1.0, index=close.index)

    # Determine bin size
    if bin_size is None:
        price_range = close.max() - close.min()
        if price_range <= 0:
            return _empty_context()
        bin_size = price_range / 100.0

    # Build histogram using numpy digitize for reliable integer bin indices
    min_price = typical.min()
    max_price = typical.max()
    bin_edges = np.arange(min_price, max_price + bin_size, bin_size)
    if len(bin_edges) < 2:
        return _empty_context()
    bin_indices = np.digitize(typical, bin_edges) - 1

    # Aggregate volume per bin
    hist = np.zeros(len(bin_edges) - 1, dtype=float)
    for idx, vol in zip(bin_indices, volumes):
        if 0 <= idx < len(hist):
            hist[idx] += vol

    # POC = bin with highest volume
    poc_bin = int(np.argmax(hist))
    poc = (bin_edges[poc_bin] + bin_edges[poc_bin + 1]) / 2.0

    # Value area: accumulate bins from POC outward until value_area_pct of total volume
    total_volume = hist.sum()
    if total_volume <= 0:
        return _empty_context()

    va_volume = hist[poc_bin]
    va_low_idx = poc_bin
    va_high_idx = poc_bin

    left = poc_bin - 1
    right = poc_bin + 1
    while va_volume / total_volume < value_area_pct and (left >= 0 or right < len(hist)):
        left_vol = hist[left] if left >= 0 else 0.0
        right_vol = hist[right] if right < len(hist) else 0.0

        if left_vol >= right_vol and left >= 0:
            va_volume += left_vol
            va_low_idx = left
            left -= 1
        elif right < len(hist):
            va_volume += right_vol
            va_high_idx = right
            right += 1
        elif left >= 0:
            va_volume += left_vol
            va_low_idx = left
            left -= 1
        else:
            break

    vah = bin_edges[va_high_idx + 1]
    val = bin_edges[va_low_idx]

    # Previous day context — use last candle's close/high/low as proxy
    prev_close = float(close.iloc[-1])
    prev_high = float(df["high"].iloc[-1]) if "high" in df.columns else prev_close
    prev_low = float(df["low"].iloc[-1]) if "low" in df.columns else prev_close

    return {
        "poc": round(float(poc), 2),
        "vah": round(float(vah), 2),
        "val": round(float(val), 2),
        "prev_close": round(float(prev_close), 2),
        "prev_high": round(float(prev_high), 2),
        "prev_low": round(float(prev_low), 2),
    }


def _empty_context() -> dict:
    return {
        "poc": 0.0,
        "vah": 0.0,
        "val": 0.0,
        "prev_close": 0.0,
        "prev_high": 0.0,
        "prev_low": 0.0,
    }


# ---------------------------------------------------------------------------
# Schema detection
# ---------------------------------------------------------------------------


def determine_schema(
    price: float,
    poc: float,
    vah: float,
    val: float,
    proximity_pips: float = 5.0,
) -> int:
    """
    Determine which volume-profile schema applies.

    Mirrors BOT14 EliteAuctionScaling.DetermineSchema().

    Returns
    -------
    int — schema number (2, 5, 6, 7) or 0 if undetermined.
    """
    tol = proximity_pips
    if price > vah + tol:
        return 2
    if price < val - tol:
        return 6
    if price > poc and price <= vah:
        return 7
    if price < poc and price >= val:
        return 5
    # Default fallback
    return 5


def get_schema_direction(schema: int) -> Optional[str]:
    """Return 'BUY' or 'SELL' for the given schema, or None."""
    directions = {2: "SELL", 5: "BUY", 6: "BUY", 7: "SELL"}
    return directions.get(schema)


def get_schema_targets(schema: int, ctx: MarketContext) -> list[float]:
    """
    Return ordered target prices for the given schema.
    Mirrors BOT14 EliteAuctionScaling.GetSchemaTargets().
    """
    targets = []
    if schema == 2:
        targets.append(ctx.poc)
        targets.append(ctx.val)
        if ctx.prev_low > 0:
            targets.append(ctx.prev_low)
    elif schema == 5:
        targets.append(ctx.poc)
        targets.append(ctx.vah)
    elif schema == 6:
        targets.append(ctx.val)
        targets.append(ctx.poc)
        targets.append(ctx.vah)
        if ctx.prev_high > 0:
            targets.append(ctx.prev_high)
    elif schema == 7:
        targets.append(ctx.poc)
        targets.append(ctx.val)
    return [t for t in targets if t > 0]


# ---------------------------------------------------------------------------
# Entry conditions
# ---------------------------------------------------------------------------


def can_execute(schema: int, price: float, vah: float, val: float, poc: float, proximity_pips: float = 5.0) -> bool:
    """
    Check whether the current price satisfies the entry condition for the
    detected schema.  Mirrors BOT14 EliteAuctionScaling.CanExecuteNow().
    """
    tol = proximity_pips
    if schema == 2:
        dist_above = price - vah
        return dist_above > tol and dist_above < 50 * tol
    if schema == 5:
        return price > val and price < poc
    if schema == 6:
        return price < val - tol
    if schema == 7:
        return price < vah and price > poc
    return True


# ---------------------------------------------------------------------------
# Stop-loss calculation
# ---------------------------------------------------------------------------


def get_schema_stop_loss(schema: int, trade_type: str, ctx: MarketContext, proximity_pips: float = 3.0) -> float:
    """
    Calculate stop-loss price for the given schema and direction.
    Mirrors BOT14 EliteAuctionScaling.GetSchemaStopLoss().
    """
    prox = proximity_pips
    if schema == 2:
        return ctx.vah + prox
    if schema == 5:
        return ctx.val - prox
    if schema == 6:
        if trade_type == "BUY":
            return ctx.val - prox
        else:
            return ctx.vah + prox
    if schema == 7:
        if trade_type == "SELL":
            return ctx.vah + prox
        else:
            return ctx.val - prox
    # Default
    if trade_type == "BUY":
        return ctx.val - prox
    return ctx.vah + prox


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------


def calculate_position_size(
    sl_pips: float,
    risk_pct: float,
    balance: float,
    pip_value_per_unit: float,
    max_volume_lots: float = 10.0,
    min_trade_volume: float = 0.1,
    volume_step: float = 0.1,
    volume_min_units: float = 0.01,
    volume_max_units: float = 100.0,
) -> float:
    """
    Risk-based position sizing with a hard volume cap.

    Formula (corrected from BOT14 v5.1 → v6.0):
        volume = risk_amount / (sl_pips × pip_value_per_unit)

    Parameters
    ----------
    sl_pips : float
        Stop-loss distance in pips/points.
    risk_pct : float
        Risk as a percentage of account balance (e.g. 2.0 for 2%).
    balance : float
        Current account balance.
    pip_value_per_unit : float
        Value of one pip per unit of the instrument.
    max_volume_lots : float
        Hard ceiling on position size in lots.
    min_trade_volume : float
        Minimum trade volume to accept.
    volume_step : float
        Rounding step for volume.
    volume_min_units : float
        Minimum volume units (broker minimum).
    volume_max_units : float
        Maximum volume units (broker maximum).

    Returns
    -------
    float — rounded position volume in lots, or 0.0 if calculation is invalid.
    """
    if sl_pips <= 0 or pip_value_per_unit <= 0:
        log.warning("Invalid sl_pips or pip_value_per_unit — cannot size position")
        return 0.0

    risk_amount = balance * (risk_pct / 100.0)
    raw_volume = risk_amount / (sl_pips * pip_value_per_unit)

    # Apply hard lot cap before any rounding
    max_units = max_volume_lots
    raw_volume = min(raw_volume, max_units)

    # Apply broker min/max and step rounding
    raw_volume = max(raw_volume, volume_min_units)
    raw_volume = min(raw_volume, volume_max_units)
    raw_volume = (raw_volume // volume_step) * volume_step
    raw_volume = round(raw_volume, 1)

    if raw_volume < min_trade_volume:
        log.debug("Position size %.2f below minimum %.2f — trade skipped", raw_volume, min_trade_volume)
        return 0.0

    return raw_volume


# ---------------------------------------------------------------------------
# Full entry evaluation
# ---------------------------------------------------------------------------


def evaluate_entry(
    ctx: MarketContext,
    schema: int,
    config: dict,
) -> Optional[SchemaResult]:
    """
    Full entry evaluation: check conditions, compute SL/TP/position size.

    Returns a :class:`SchemaResult` if the entry is valid, or ``None`` if
    the setup should be skipped.
    """
    price = ctx.price
    direction = get_schema_direction(schema)
    if direction is None:
        log.debug("Schema %d has no direction — skipping", schema)
        return None

    if not can_execute(
        schema, price, ctx.vah, ctx.val, ctx.poc,
        proximity_pips=config.get("vp_zone_proximity_pips", 5.0),
    ):
        log.debug("Schema %d entry conditions not met at price %.2f", schema, price)
        return None

    sl_price = get_schema_stop_loss(schema, direction, ctx, proximity_pips=3.0)
    risk_pips = abs(price - sl_price) / config.get("pip_size", 0.01)

    # FIX 3 equivalent: skip if structure SL is too tight
    min_risk_distance = config.get("min_risk_distance_pips", 30.0)
    if risk_pips < min_risk_distance:
        log.debug(
            "Schema %d skipped: structure SL only %.1f pips (min %d). No artificial widening.",
            schema, risk_pips, min_risk_distance,
        )
        return None

    # FIX 4 equivalent: skip inside-VA schemas when price is too far from boundary
    max_entry_dist = config.get("max_schema_entry_distance_pips", 15.0)
    if schema in (5, 7):
        if schema == 5:
            boundary_dist = (price - ctx.val) / config.get("pip_size", 0.01)
        else:
            boundary_dist = (ctx.vah - price) / config.get("pip_size", 0.01)
        if boundary_dist > max_entry_dist:
            log.debug(
                "Schema %d skipped: price %.1f pips from VA boundary (max %d).",
                schema, boundary_dist, max_entry_dist,
            )
            return None

    # Check RR ratio against first target
    targets = get_schema_targets(schema, ctx)
    if not targets:
        log.debug("Schema %d has no valid targets — skipping", schema)
        return None

    tp1_price = targets[0]
    tp1_pips = abs(tp1_price - price) / config.get("pip_size", 0.01)
    min_rr = config.get("min_rr_ratio", 1.2)

    if tp1_pips < risk_pips * min_rr:
        log.debug(
            "Schema %d RR too low: %.1fp profit vs %.1fp risk (min %.1f:1)",
            schema, tp1_pips, risk_pips, min_rr,
        )
        return None

    # Check max SL
    max_sl = config.get("max_sl_pips", 20.0)
    if risk_pips > max_sl:
        log.debug("Schema %d SL %.1fp exceeds MaxSL %.1fp — skipped", schema, risk_pips, max_sl)
        return None

    # Position sizing
    balance = config.get("balance", 0.0)
    risk_pct = config.get("base_risk_pct", 2.0)
    pip_value_per_unit = config.get("pip_value_per_unit", 1.0)
    volume = calculate_position_size(
        sl_pips=risk_pips,
        risk_pct=risk_pct,
        balance=balance,
        pip_value_per_unit=pip_value_per_unit,
        max_volume_lots=config.get("max_volume_lots", 10.0),
        min_trade_volume=config.get("min_trade_volume", 0.1),
        volume_step=config.get("volume_step", 0.1),
        volume_min_units=config.get("volume_min_units", 0.01),
        volume_max_units=config.get("volume_max_units", 100.0),
    )

    if volume <= 0:
        log.debug("Schema %d position size is 0 — trade skipped", schema)
        return None

    log.info(
        "Schema %s entry valid — %s | Price: %.2f | SL: %.2f (%.1fp) | TP1: %.2f (%.1fp) | RR: %.2f | Volume: %.2f",
        schema, direction, price, sl_price, risk_pips, tp1_price, tp1_pips, tp1_pips / risk_pips, volume,
    )

    return SchemaResult(
        schema=schema,
        direction=direction,
        entry=price,
        sl=sl_price,
        targets=targets,
        risk_pips=risk_pips,
        tp1_pips=tp1_pips,
        rr_ratio=tp1_pips / risk_pips,
        reason=f"Schema {schema} {direction}",
    )


# ---------------------------------------------------------------------------
# Magnet mode (mean reversion fallback)
# ---------------------------------------------------------------------------


def try_magnet_trade(
    ctx: MarketContext,
    rsi_value: Optional[float],
    bb_upper: Optional[float],
    bb_lower: Optional[float],
    config: dict,
) -> Optional[SchemaResult]:
    """
    Magnet mode: mean-reversion trades when no active position.
    Mirrors BOT14 EliteAuctionScaling.TryMagnetTrade().
    """
    enable_magnet = config.get("enable_magnet_mode", True)
    if not enable_magnet:
        return None

    magnet_start_hour = config.get("magnet_start_hour", 17)
    # We don't have broker hour in Python; skip hour check and let caller decide

    price = ctx.price
    poc = ctx.poc
    vah = ctx.vah
    val = ctx.val

    trade_type = None
    target = 0.0

    # Short setup: overbought + above VAH
    if price > vah and bb_upper is not None and price >= bb_upper and rsi_value is not None and rsi_value >= config.get("rsi_overbought", 70):
        trade_type = "SELL"
        target = poc

    # Long setup: oversold + below VAL
    if price < val and bb_lower is not None and price <= bb_lower and rsi_value is not None and rsi_value <= config.get("rsi_oversold", 30):
        trade_type = "BUY"
        target = poc

    if trade_type is None or target == 0:
        return None

    entry = price  # Use current price as entry
    sl_prox = 3.0 * config.get("pip_size", 0.01)
    sl = val - sl_prox if trade_type == "BUY" else vah + sl_prox

    risk_pips = abs(entry - sl) / config.get("pip_size", 0.01)
    tp_pips = abs(target - entry) / config.get("pip_size", 0.01)

    min_risk_distance = config.get("min_risk_distance_pips", 30.0)
    if risk_pips < min_risk_distance or risk_pips > config.get("max_sl_pips", 20.0):
        return None

    if tp_pips < risk_pips * config.get("min_rr_ratio", 1.2):
        return None

    balance = config.get("balance", 0.0)
    risk_pct = config.get("base_risk_pct", 2.0)
    pip_value_per_unit = config.get("pip_value_per_unit", 1.0)
    volume = calculate_position_size(
        sl_pips=risk_pips,
        risk_pct=risk_pct,
        balance=balance,
        pip_value_per_unit=pip_value_per_unit,
        max_volume_lots=config.get("max_volume_lots", 10.0),
        min_trade_volume=config.get("min_trade_volume", 0.1),
        volume_step=config.get("volume_step", 0.1),
        volume_min_units=config.get("volume_min_units", 0.01),
        volume_max_units=config.get("volume_max_units", 100.0),
    )

    if volume <= 0:
        return None

    log.info(
        "[MAGNET MODE] %s → POC | Entry: %.2f | SL: %.2f | TP: %.2f | Volume: %.2f",
        trade_type, entry, sl, target, volume,
    )

    return SchemaResult(
        schema=0,
        direction=trade_type,
        entry=entry,
        sl=sl,
        targets=[target],
        risk_pips=risk_pips,
        tp1_pips=tp_pips,
        rr_ratio=tp_pips / risk_pips,
        reason="Magnet mode mean reversion",
    )
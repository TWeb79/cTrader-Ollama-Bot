#!/usr/bin/env python3
"""US500 AI Trading Assistant — Local ML Model + cTrader MCP server.

Replaces the Ollama LLM approach with a local ML model (GradientBoosting)
trained on historical trade data. Follows the plan.md pipeline:

1. Data Preparation — load historical candles and trade events.
2. Model Training  — train a local model on historical outcomes.
3. Integration     — use the trained model for live BUY/SELL/HOLD decisions.
4. Feedback Loop   — record trade outcomes and retrain periodically.

Port allocations (project 55, per RULES_ports.md):
    8055 → Web dashboard
    8155 → FastAPI service
    8255 → Database
    8955 → LLM / ML service

Author: Inventions4All - github:TWeb79
Version: 1.0.0  (deployment: 2026-07-30T17:53:53+02:00)
"""

import asyncio
import json
import logging
import os as _os
from typing import Any

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

from trader_config import CONFIG, get_port_mapping, load_env_ports
from data_preparation import fetch_historical_candles, prepare_training_data
from ai_model import ModelTrainer, MODEL_PATH, SCALER_PATH
from feedback_loop import process_trade_feedback
from trade_executor import CTraderMCPClient, TOOL_NAMES
from trade_logger import log_trades

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("trader.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("ai_trader")


def _compute_live_indicators(candles: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute indicator summary from intraday candles for ML prediction."""
    if not candles:
        return {}
    import pandas as pd

    df = pd.DataFrame(candles)
    df = df.rename(columns={c: c.lower() for c in df.columns})
    close = df["close"].astype(float)

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))

    sma20_val = sma20.iloc[-1] if not sma20.isna().all() else None
    sma50_val = sma50.iloc[-1] if not sma50.isna().all() else None
    trend = (
        "bullish"
        if sma20_val is not None
        and sma50_val is not None
        and sma20_val > sma50_val
        else "bearish"
    )

    return {
        "last_close": round(float(close.iloc[-1]), 2),
        "6m_high": round(float(close.max()), 2),
        "6m_low": round(float(close.min()), 2),
        "sma20": round(float(sma20_val), 2) if sma20_val is not None else None,
        "sma50": round(float(sma50_val), 2) if sma50_val is not None else None,
        "rsi14": round(float(rsi.iloc[-1]), 2) if not rsi.isna().all() else None,
        "trend_20_vs_50": trend,
    }


def predict_signal(model_trainer: ModelTrainer, indicators: dict[str, Any]) -> str:
    """Use the trained ML model to predict BUY/SELL/HOLD."""
    prediction = model_trainer.predict(indicators)
    bias = prediction.get("bias", "neutral")
    log.info(
        "ML signal: bias=%s confidence=%.2f",
        bias,
        prediction.get("confidence", 0),
    )
    return bias.upper()


def predict_position_management(
    model_trainer: ModelTrainer,
    _strategy_hint: dict[str, Any],
    current_price: float,
    pending_orders: list[dict[str, Any]],
) -> str:
    """Use the ML model to decide TRAIL_SL, CLOSE, or HOLD."""
    if model_trainer.model is None:
        return "HOLD"

    import numpy as _np

    features = {
        "last_close": current_price,
        "entry_price": 0.0,
        "current_sl": 0.0,
        "current_tp": 0.0,
        "volume": 0.0,
        "unrealized_pnl": 0.0,
    }

    try:
        feature_vec = _np.array([[
            features["last_close"],
            features["entry_price"],
            features["current_sl"],
            features["current_tp"],
            features["volume"],
            features["unrealized_pnl"],
        ]])
        prediction = model_trainer.model.predict(feature_vec)[0]
        labels = (
            model_trainer.label_map_inv
            if hasattr(model_trainer, "label_map_inv")
            else {}
        )
        action = labels.get(int(prediction), "HOLD").upper()
        if action not in ("TRAIL_SL", "CLOSE", "HOLD"):
            action = "HOLD"
        return action
    except Exception:
        log.exception("ML position management prediction failed")
        return "HOLD"


async def run_volume_profile_strategy(
    client: CTraderMCPClient,
    config: dict[str, Any],
    model_trainer: ModelTrainer,
    candles: list[dict[str, Any]],
) -> None:
    """Run the volume profile strategy with ML-assisted position management."""
    log.info("Computing volume profile from %d candle(s) ...", len(candles))
    vp = compute_volume_profile(
        candles,
        bin_size=config.get("vp_bin_size"),
        value_area_pct=config.get("vp_value_area_pct", 0.70),
    )
    log.info("Volume profile — POC: %.2f | VAH: %.2f | VAL: %.2f", vp["poc"], vp["vah"], vp["val"])

    if vp["poc"] == 0.0 or vp["vah"] == 0.0 or vp["val"] == 0.0:
        log.error("Volume profile computation returned zero values — check candle data.")
        return

    import pandas as _pd

    df = _pd.DataFrame(candles)
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

    iterations = 0
    active_position_id = None

    while True:
        log.debug("--- Volume Profile Polling iteration %d ---", iterations + 1)
        positions = await client.call(TOOL_NAMES["list_positions"], {})
        open_count = len(positions) if isinstance(positions, list) else len(
            positions.get("positions", positions.get("items", []))
        )
        log.info("Open positions: %d / %d", open_count, config["max_open_positions"])

        if open_count > config["max_open_positions"]:
            for pos in positions[config["max_open_positions"]:]:
                pos_id = pos.get("id", pos.get("positionId", "unknown"))
                log.warning("Closing extra position %s", pos_id)
                if config["dry_run"]:
                    log.info("[DRY RUN] Would close position %s", pos_id)
                else:
                    await client.call(TOOL_NAMES["close_position"], {"Id": pos_id})

        if open_count >= config["max_open_positions"]:
            log.info("Max open positions reached. Checking position management.")
            quote_pos = await client.call(TOOL_NAMES["current_price"], {"symbolName": config["symbol"]})
            price_pos = quote_pos.get("bid") or quote_pos.get("price") if isinstance(quote_pos, dict) else quote_pos

            pending_orders = []
            try:
                po = await client.call(TOOL_NAMES["list_pending_orders"], {})
                if isinstance(po, list):
                    pending_orders = po
                elif isinstance(po, dict):
                    pending_orders = po.get("pendingOrders", po.get("items", []))
            except Exception:
                log.exception("Failed to fetch pending orders")

            intraday_candles = await fetch_historical_candles(client, config, TOOL_NAMES)
            mgmt_signal = predict_position_management(
                model_trainer, {}, price_pos, pending_orders,
            )

            if mgmt_signal == "TRAIL_SL":
                if not config["dry_run"] and positions:
                    for pos in positions:
                        pos_id = pos.get("id", pos.get("positionId", "unknown"))
                        current_sl = pos.get("stopLoss", pos.get("sl", 0))
                        pos_side = pos.get("type", pos.get("side", "Buy"))
                        new_sl = (
                            price_pos - config["stop_loss_points"]
                            if pos_side == "Buy"
                            else price_pos + config["stop_loss_points"]
                        )
                        if pos_side == "Buy" and new_sl > current_sl or pos_side != "Buy" and new_sl < current_sl:
                            log.info("Updating SL for %s to %.2f", pos_id, new_sl)
                            await client.call(TOOL_NAMES["amend_position"], {
                                "Id": pos_id,
                                "stopLoss": round(new_sl, 2),
                            })
            elif mgmt_signal == "CLOSE":
                if not config["dry_run"] and positions:
                    for pos in positions:
                        pos_id = pos.get("id", pos.get("positionId", "unknown"))
                        await client.call(TOOL_NAMES["close_position"], {"Id": pos_id})
        else:
            ctx = await _build_market_context(client, vp, config)
            if ctx is None:
                log.warning("Could not build market context — skipping.")
            else:
                schema = determine_schema(
                    ctx.price, ctx.poc, ctx.vah, ctx.val,
                    proximity_pips=config.get("vp_zone_proximity_pips", 5.0),
                )
                direction = get_schema_direction(schema)
                log.info(
                    "Schema %d | Dir: %s | Price: %.2f | VAH: %.2f | VAL: %.2f | POC: %.2f",
                    schema, direction, ctx.price, ctx.vah, ctx.val, ctx.poc,
                )
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
                    log.info("Volume profile signals %s — placing trade.", result.direction)
                    if config["dry_run"]:
                        log.info("[DRY RUN] VP trade: %s", json.dumps({
                            "schema": result.schema,
                            "direction": result.direction,
                            "entry": result.entry,
                            "sl": result.sl,
                            "targets": result.targets,
                        }, indent=2))
                    else:
                        order_args = {
                            "symbolName": config["symbol"],
                            "side": "buy" if result.direction == "BUY" else "sell",
                            "volume": config["trade_volume"],
                            "volumeType": "lots",
                            "stopLossPips": round(result.risk_pips, 2),
                            "takeProfitPips": round(result.tp1_pips, 2),
                        }
                        trade_result = await client.call(TOOL_NAMES["place_order"], order_args)
                        log.info("Volume profile order result: %s", trade_result)
                        active_position_id = str(trade_result.get("id", "unknown"))

        iterations += 1
        if config["max_loop_iterations"] and iterations >= config["max_loop_iterations"]:
            log.info("Reached max_loop_iterations, stopping.")
            break
        await log_trades(client, config)
        await asyncio.sleep(config["poll_interval_seconds"])


async def _build_market_context(
    client: CTraderMCPClient,
    vp: dict[str, Any],
    config: dict[str, Any],
) -> MarketContext | None:
    """Fetch live price and combine with volume profile data."""
    quote = await client.call(TOOL_NAMES["current_price"], {"symbolName": config["symbol"]})
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


async def main() -> None:
    """Main orchestrator — implements the plan.md 4-step ML pipeline.

    Step 1: Data Preparation — load historical candles and trade events.
    Step 2: Model Training  — train the local ML model on historical data.
    Step 3: Integration     — use the trained model for live trading decisions.
    Step 4: Feedback Loop   — record outcomes and retrain periodically.
    """
    import os as _os

    ports = load_env_ports("ports.env")
    port_mapping = get_port_mapping()
    log.info("Project %d port mapping: %s", 55, port_mapping)

    config = CONFIG
    client = CTraderMCPClient(config)

    max_retries = 10
    retry_delay = 5

    for attempt in range(1, max_retries + 1):
        try:
            log.info("Connecting to cTrader MCP server (attempt %d/%d)...", attempt, max_retries)
            await client.connect()
            log.info("Connected to cTrader MCP server.")
            break
        except (Exception, RuntimeError) as exc:
            log.warning("Connection attempt %d failed: %s", attempt, exc)
            if attempt >= max_retries:
                log.error("Max connection retries reached — exiting.")
                return
            import time
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 60)

    try:
        # Step 1: Data Preparation
        log.info("=== Step 1: Data Preparation ===")
        candles = await fetch_historical_candles(client, config, TOOL_NAMES)
        if not candles:
            log.error("No candle data returned — check MCP connection.")
            return
        events = _load_events_file("events.json")
        log.info("Loaded %d candle(s) and %d event(s).", len(candles), len(events))

        # Step 2: Model Training
        log.info("=== Step 2: Model Training ===")
        model_trainer = ModelTrainer(
            model_path=config.get("ml_model_path", MODEL_PATH),
            scaler_path=config.get("ml_scaler_path", SCALER_PATH),
        )
        if not model_trainer.load():
            log.info("No saved model — training new model from historical data.")
            model_trainer.train(events_file="events.json")

        # Step 3: Integration — live trading loop with ML predictions
        log.info("=== Step 3: Integration — Live Trading Loop ===")
        iterations = 0

        while True:
            try:
                log.debug("--- Live Monitoring iteration %d ---", iterations + 1)
                quote = await client.call(TOOL_NAMES["current_price"], {"symbolName": config["symbol"]})
                current_price = quote.get("bid") or quote.get("price") if isinstance(quote, dict) else quote

                intraday_candles = await fetch_historical_candles(client, config, TOOL_NAMES)
                indicators = _compute_live_indicators(intraday_candles)

                signal = predict_signal(model_trainer, indicators)
                log.info("ML trading signal: %s", signal)

                if signal in ("BUY", "SELL") and config.get("max_open_positions", 1) > 0:
                    log.info("ML signal %s — executing trade.", signal)
                    if config["dry_run"]:
                        log.info("[DRY RUN] ML trade: %s %s", signal, config["symbol"])
                    else:
                        sl_pips = config["stop_loss_points"]
                        tp_pips = config["take_profit_points"]
                        order_args = {
                            "symbolName": config["symbol"],
                            "side": "buy" if signal == "BUY" else "sell",
                            "volume": config["trade_volume"],
                            "volumeType": "lots",
                            "stopLossPips": round(sl_pips, 2),
                            "takeProfitPips": round(tp_pips, 2),
                        }
                        result = await client.call(TOOL_NAMES["place_order"], order_args)
                        log.info("Order result: %s", result)

                # Step 4: Feedback Loop
                log.info("=== Step 4: Feedback Loop ===")
                deals = await client.call(TOOL_NAMES["list_deals"], {})
                deal_list = deals if isinstance(deals, list) else deals.get("deals", deals.get("items", [])) if isinstance(deals, dict) else []
                for deal in deal_list:
                    if str(deal.get("symbol", deal.get("symbolName", ""))).upper() == "US500":
                        process_trade_feedback(model_trainer, deal, config)

                await log_trades(client, config)

                iterations += 1
                if config["max_loop_iterations"] and iterations >= config["max_loop_iterations"]:
                    log.info("Reached max_loop_iterations, stopping.")
                    break
                import time
                time.sleep(config["poll_interval_seconds"])
            except (Exception, RuntimeError) as exc:
                log.exception("Error in trading loop iteration: %s", exc)
                import time
                time.sleep(retry_delay)

    finally:
        await client.close()


def _load_events_file(filepath: str) -> list[dict[str, Any]]:
    """Load trade events from the local JSON events file."""
    if not _os.path.exists(filepath):
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, ValueError):
        return []


if __name__ == "__main__":
    asyncio.run(main())
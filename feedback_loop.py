"""Step 4: Feedback Loop — update the ML model based on trade outcomes.

Designs and implements a feedback mechanism that:
- Records trade outcomes (especially failures) into trade_history.json.
- Periodically retrains the model when the retrain interval is reached.
- Prioritizes learning from failed trades (negative PnL).

Author: Inventions4All - github:TWeb79
Version: 1.0.0  (deployment: 2026-07-30T17:53:53+02:00)
"""

import json
import logging
from typing import Any

from ai_model import ModelTrainer

log = logging.getLogger("ai_trader.feedback")


def classify_trade_outcome(
    gross_profit: float,
    pips: float,
    trade_type: str,
) -> str:
    """Classify a trade outcome for the feedback loop.

    Args:
        gross_profit: The gross profit from the trade.
        pips: The pip movement of the trade.
        trade_type: 'Buy' or 'Sell'.

    Returns:
        Outcome label: 'buy', 'sell', or 'hold'.
    """
    if gross_profit > 0:
        return trade_type.lower()
    if gross_profit < 0:
        log.info(
            "Failed trade recorded — type=%s, profit=%.2f, pips=%.1f",
            trade_type, gross_profit, pips,
        )
    return "hold"


def compute_failure_weight(features: dict[str, Any], outcome: str) -> float:
    """Compute a weight for a training sample based on outcome quality.

    Failed trades receive higher weight so the model learns from mistakes.

    Args:
        features: Feature dict for the trade.
        outcome: Trade outcome label.

    Returns:
        Weight multiplier (>= 1.0 for failures, 1.0 otherwise).
    """
    if outcome == "hold" and features.get("grossProfit", 0) < 0:
        return 2.0
    return 1.0


async def process_trade_feedback(
    model_trainer: ModelTrainer,
    trade_record: dict[str, Any],
    config: dict[str, Any],
) -> None:
    """Process a completed trade's feedback for the feedback loop.

    Records the outcome, computes a weighted label, and triggers
    retraining if the retrain interval is reached.

    Args:
        model_trainer: The active ModelTrainer instance.
        trade_record: Dict with trade outcome data (profit, pips, type, entryPrice).
        config: Configuration dict with ml_retrain_interval and ml_min_samples.
    """
    gross_profit = trade_record.get("grossProfit", 0)
    pips = trade_record.get("pips", 0)
    trade_type = trade_record.get("type", trade_record.get("side", "Buy"))
    outcome = classify_trade_outcome(gross_profit, pips, trade_type)

    features = {
        "entryPrice": trade_record.get("entryPrice", 0),
        "closePrice": trade_record.get("closePrice", 0),
        "grossProfit": gross_profit,
        "pips": pips,
        "type": trade_type,
    }

    model_trainer.record_trade_outcome(features, outcome, config)
    log.info(
        "Trade feedback recorded — type=%s outcome=%s profit=%.2f",
        trade_type, outcome, gross_profit,
    )
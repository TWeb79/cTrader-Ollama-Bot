"""Step 2 & 3: ML Model Training — local model training and live prediction.

Trains a GradientBoostingClassifier on historical trade outcomes and uses
the trained model for live BUY/SELL/HOLD/TRAIL_SL/CLOSE decisions.

Author: Inventions4All - github:TWeb79
Version: 1.0.0  (deployment: 2026-07-30T17:53:53+02:00)
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Optional

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger("ai_trader.ai_model")

MODEL_PATH = "trading_model.joblib"
SCALER_PATH = "trading_scaler.joblib"
TRADES_HISTORY_PATH = "trade_history.json"
EVENTS_LOG_FILE = "events.json"


class TradeEvent:
    """Represents a single trade event with features and outcome."""

    def __init__(self, features: dict[str, Any], outcome: str) -> None:
        self.features = features
        self.outcome = outcome


class ModelTrainer:
    """Trains and manages a local ML model for trading decisions.

    Implements the plan.md pipeline:
    1. Data Preparation — load historical data from events.json + candles.
    2. Feature Engineering — compute indicators as features.
    3. Model Training — train GradientBoostingClassifier on outcomes.
    4. Prediction — use the trained model for live decisions.
    5. Feedback — record outcomes and retrain periodically.
    """

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        scaler_path: str = SCALER_PATH,
    ) -> None:
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model: Optional[Pipeline] = None
        self.scaler: Optional[StandardScaler] = None
        self.trade_history: list[TradeEvent] = []
        self.trade_count = 0
        self.label_map: dict[str, int] = {}
        self.label_map_inv: dict[int, str] = {}

    def load(self) -> bool:
        """Load a previously trained model from disk.

        Returns:
            True if model was loaded successfully, False otherwise.
        """
        if not os.path.exists(self.model_path):
            log.info("No saved model found at %s", self.model_path)
            return False
        try:
            self.model = joblib.load(self.model_path)
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
            log.info("Loaded ML model from %s", self.model_path)
            return True
        except Exception:
            log.exception("Failed to load ML model from %s", self.model_path)
            return False

    def save(self) -> None:
        """Save the trained model and scaler to disk."""
        try:
            if self.model is not None:
                joblib.dump(self.model, self.model_path)
                log.info("Saved ML model to %s", self.model_path)
            if self.scaler is not None:
                joblib.dump(self.scaler, self.scaler_path)
                log.info("Saved scaler to %s", self.scaler_path)
        except Exception:
            log.exception("Failed to save ML model")

    def _extract_features(self, indicators: dict[str, Any]) -> Optional[np.ndarray]:
        """Convert indicator summary dict to feature array for prediction.

        Args:
            indicators: Dict with keys matching ml_features config.

        Returns:
            2D numpy array of features, or None if data is missing.
        """
        feature_names = [
            "last_close", "6m_high", "6m_low",
            "sma20", "sma50", "rsi14",
        ]
        values: list[float] = []
        for name in feature_names:
            val = indicators.get(name)
            if val is None:
                return None
            values.append(float(val))
        return np.array([values])

    def prepare_training_data(
        self, events_file: str = EVENTS_LOG_FILE
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load historical trade events and compute features + labels.

        Labels are derived from trade outcomes:
        - 'buy': profitable long trade (grossProfit > 0)
        - 'sell': profitable short trade (grossProfit > 0)
        - 'hold': unprofitable or break-even trade

        Args:
            events_file: Path to the events JSON file.

        Returns:
            Tuple of (feature_matrix, label_array).
        """
        if not os.path.exists(events_file):
            log.warning("No events file at %s — cannot prepare training data", events_file)
            return np.empty((0, 6)), np.array([])

        try:
            with open(events_file, "r", encoding="utf-8") as fh:
                events = json.load(fh)
        except (json.JSONDecodeError, ValueError):
            log.warning("Failed to parse %s — skipping training data", events_file)
            return np.empty((0, 6)), np.array([])

        features: list[list[float]] = []
        labels: list[str] = []
        for event in events:
            entry_price = event.get("entryPrice", 0)
            close_price = event.get("closePrice")
            gross_profit = event.get("grossProfit", 0)
            pips = event.get("pips", 0)
            trade_type = event.get("type", "Buy")

            if close_price is None or entry_price == 0:
                continue

            if gross_profit > 0:
                label = trade_type.lower()
            else:
                label = "hold"

            price_change_pct = (
                (close_price - entry_price) / entry_price * 100
                if entry_price != 0
                else 0.0
            )
            features.append([
                entry_price,
                close_price,
                price_change_pct,
                pips,
                gross_profit,
                1.0 if trade_type == "Buy" else 0.0,
            ])
            labels.append(label)

        if not features:
            log.warning("No valid training samples found in %s", events_file)
            return np.empty((0, 6)), np.array([])

        X = np.array(features)
        y = np.array(labels)
        log.info("Prepared training data: %d samples from %s", len(X), events_file)
        return X, y

    def train(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        events_file: str = EVENTS_LOG_FILE,
    ) -> bool:
        """Train the ML model on historical trade outcomes.

        If X and y are not provided, loads them from events.json.

        Args:
            X: Optional pre-computed feature matrix.
            y: Optional pre-computed label array.
            events_file: Path to events JSON for auto-loading data.

        Returns:
            True if training succeeded, False otherwise.
        """
        if X is None or y is None:
            X, y = self.prepare_training_data(events_file)

        if len(X) == 0:
            log.warning("No training data available — cannot train model")
            return False

        if len(X) < 5:
            log.warning(
                "Only %d training samples — model may not generalize well", len(X),
            )

        unique_labels = sorted(set(y))
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        y_encoded = np.array([self.label_map[label] for label in y])

        try:
            self.model = Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=42,
                )),
            ])
            self.model.fit(X, y_encoded)
            self.label_map_inv = {idx: label for label, idx in self.label_map.items()}
            self.trade_count = 0
            log.info("Trained ML model on %d samples with %d classes", len(X), len(unique_labels))
            self.save()
            return True
        except Exception:
            log.exception("Failed to train ML model")
            return False

    def predict(self, indicators: dict[str, Any]) -> dict[str, Any]:
        """Predict trading decision from indicator summary.

        Args:
            indicators: Dict with indicator values for feature extraction.

        Returns:
            Dict with bias, confidence, reasoning, and predicted_label.
        """
        if self.model is None:
            log.warning("No ML model loaded — returning neutral default")
            return {"bias": "neutral", "reasoning": "model not trained", "confidence": 0.0}

        features = self._extract_features(indicators)
        if features is None:
            log.warning("Cannot extract features from indicators — missing data")
            return {
                "bias": "neutral",
                "reasoning": "insufficient indicator data",
                "confidence": 0.0,
            }

        try:
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            confidence = float(max(probabilities))
            predicted_label = self.label_map_inv.get(int(prediction), "neutral")

            bias_map = {
                "buy": "long", "sell": "short",
                "hold": "neutral", "neutral": "neutral",
            }
            bias = bias_map.get(predicted_label, "neutral")

            result = {
                "bias": bias,
                "reasoning": (
                    f"ML model prediction: {predicted_label}"
                    f" (confidence={confidence:.2f})"
                ),
                "confidence": round(confidence, 2),
                "predicted_label": predicted_label,
            }
            log.debug("ML model prediction: %s", result)
            return result
        except Exception:
            log.exception("ML model prediction failed")
            return {"bias": "neutral", "reasoning": "prediction error", "confidence": 0.0}

    def record_trade_outcome(
        self, features: dict[str, Any], outcome: str, config: dict[str, Any]
    ) -> None:
        """Record a trade outcome for the feedback loop.

        Persists to trade_history.json and triggers retraining when the
        retrain interval or minimum sample count is reached.

        Args:
            features: Feature dict from the trade.
            outcome: Trade outcome label ('buy', 'sell', 'hold').
            config: Configuration dict with ml_retrain_interval and ml_min_samples.
        """
        self.trade_history.append(TradeEvent(features, outcome))
        self.trade_count += 1

        try:
            history: list[dict[str, Any]] = []
            if os.path.exists(TRADES_HISTORY_PATH):
                with open(TRADES_HISTORY_PATH, "r", encoding="utf-8") as fh:
                    history = json.load(fh)
            history.append({
                "features": features,
                "outcome": outcome,
                "timestamp": datetime.utcnow().isoformat(),
            })
            with open(TRADES_HISTORY_PATH, "w", encoding="utf-8") as fh:
                json.dump(history[-500:], fh, indent=2, default=str)
        except Exception:
            log.exception("Failed to record trade outcome")

        retrain_interval = config.get("ml_retrain_interval", 10)
        min_samples = config.get("ml_min_samples", 50)
        if self.trade_count >= retrain_interval and len(self.trade_history) >= min_samples:
            log.info("Retraining ML model after %d trades...", self.trade_count)
            self.retrain()
            self.trade_count = 0

    def retrain(self, events_file: str = EVENTS_LOG_FILE) -> None:
        """Retrain the model using accumulated trade data."""
        log.info("Retraining ML model...")
        X, y = self.prepare_training_data(events_file)
        if len(X) == 0:
            log.warning("No retraining data available")
            return
        self.train(X=X, y=y)
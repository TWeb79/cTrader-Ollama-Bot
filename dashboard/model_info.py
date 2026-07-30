"""Inspect trained ML model metadata for dashboard display."""

import logging
import os
from typing import Any

import joblib

log = logging.getLogger("ai_trader.dashboard.model_info")


def get_model_info(model_path: str, scaler_path: str) -> dict[str, Any]:
    """Extract metadata from saved model and scaler.

    Args:
        model_path: Path to joblib model file.
        scaler_path: Path to joblib scaler file.

    Returns:
        Dict with model type, training info, feature names, etc.
    """
    info: dict[str, Any] = {
        "model_path": model_path,
        "scaler_path": scaler_path,
        "model_exists": os.path.exists(model_path),
        "scaler_exists": os.path.exists(scaler_path),
        "model_type": None,
        "feature_names": [],
        "n_features": 0,
        "classes": [],
        "last_modified": None,
    }

    if not info["model_exists"]:
        return info

    try:
        stat = os.stat(model_path)
        info["last_modified"] = datetime.utcfromtimestamp(stat.st_mtime).isoformat() + "Z"
    except OSError:
        pass

    try:
        model = joblib.load(model_path)
        info["model_type"] = type(model).__name__

        if hasattr(model, "named_steps"):
            for name, step in model.named_steps.items():
                info[f"step_{name}_type"] = type(step).__name__
                if hasattr(step, "feature_names_in_"):
                    info["feature_names"] = list(step.feature_names_in_)
                    info["n_features"] = len(info["feature_names"])
                if hasattr(step, "classes_"):
                    info["classes"] = list(step.classes_)
        elif hasattr(model, "feature_names_in_"):
            info["feature_names"] = list(model.feature_names_in_)
            info["n_features"] = len(info["feature_names"])
        if hasattr(model, "classes_"):
            info["classes"] = list(model.classes_)
    except Exception as exc:
        log.warning("Failed to inspect model: %s", exc)

    return info

"""Configuration for the US500 AI Trading Assistant (project 55).

Port allocations (RULES_ports.md, project ID 55):
    8055 → Web dashboard
    8155 → FastAPI service
    8255 → Database
    8955 → LLM / ML service

Author: Inventions4All - github:TWeb79
Version: 1.0.0  (deployment: 2026-07-30T17:53:53+02:00)
"""

import os

PROJECT_ID = 55
PROJECT_NAME = "cTraderMCP"

DASHBOARD_PORT = 8055
FASTAPI_PORT = 8155
DATABASE_PORT = 8255
LLM_PORT = 8955

VERSION = "1.0.0"
DEPLOYMENT_DATETIME = "2026-07-30T17:53:53+02:00"

CONFIG = {
    # --- cTrader MCP server connection ---
    "mcp_transport": "http",
    "mcp_url": "http://127.0.0.1:9876/mcp",

    # --- ML Model Configuration ---
    "ml_model_path": "trading_model.joblib",
    "ml_scaler_path": "trading_scaler.joblib",
    "ml_history_path": "trade_history.json",
    "ml_retrain_interval": 10,
    "ml_min_samples": 50,
    "ml_features": [
        "last_close", "6m_high", "6m_low",
        "sma20", "sma50", "rsi14", "trend_20_vs_50",
    ],

    # --- Intraday data ---
    "intraday_hours": 4,

    # --- Instrument & analysis ---
    "symbol": "US500",
    "history_months": 6,
    "candle_timeframe": "m5",

    # --- Volume Profile Strategy parameters ---
    "vp_bin_size": None,
    "vp_value_area_pct": 0.70,
    "vp_zone_proximity_pips": 5.0,
    "vp_min_risk_distance_pips": 30.0,
    "vp_max_schema_entry_distance_pips": 15.0,
    "vp_max_sl_pips": 20.0,
    "vp_min_rr_ratio": 1.2,
    "vp_base_risk_pct": 2.0,
    "vp_reduced_risk_pct": 0.5,
    "vp_max_volume_lots": 10.0,
    "vp_min_trade_volume": 0.1,
    "vp_volume_step": 0.1,
    "vp_volume_min_units": 0.01,
    "vp_volume_max_units": 100.0,
    "vp_pip_size": 0.01,
    "vp_pip_value_per_unit": 1.0,
    "vp_balance": 10000.0,

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
    "max_loop_iterations": 1000,

    # --- Risk management ---
    "trade_volume": 0.1,
    "stop_loss_points": 15.0,
    "take_profit_points": 30.0,
    "max_open_positions": 1,

    # --- Safety switch ---
    "dry_run": True,
}


def get_port_mapping() -> dict[str, int]:
    """Return the port allocation for this project."""
    return {
        "dashboard": DASHBOARD_PORT,
        "fastapi": FASTAPI_PORT,
        "database": DATABASE_PORT,
        "llm": LLM_PORT,
    }


def load_env_ports(env_path: str = "ports.env") -> dict[str, int]:
    """Load port overrides from a ports.env file if it exists."""
    ports = get_port_mapping()
    if not os.path.exists(env_path):
        return ports
    with open(env_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key in ports and value.isdigit():
                    ports[key] = int(value)
    return ports
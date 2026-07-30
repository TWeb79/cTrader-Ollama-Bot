"""FastAPI backend for the cTraderMCP web dashboard."""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from dashboard.log_reader import get_log_stats, tail_log
from dashboard.model_info import get_model_info
from dashboard.stats import compute_stats, load_events

log = logging.getLogger("ai_trader.dashboard")

app = FastAPI(title="cTraderMCP Dashboard", version="1.0.0")

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "dashboard"
STATIC_DIR = TEMPLATES_DIR / "static"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _get_env_path(key: str, default: str) -> str:
    """Read file path from environment variable or use default."""
    return os.environ.get(key, default)


def _get_docker_client():
    """Create a Docker client from environment if socket is available."""
    try:
        import docker
        return docker.from_env()
    except Exception as exc:
        log.warning("Docker client unavailable: %s", exc)
        return None


def _get_trader_container_name() -> str:
    return os.environ.get("TRADER_CONTAINER", "55-cTraderMCP-trader")


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request) -> Any:
    """Serve the dashboard SPA."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/stats")
async def api_stats() -> Any:
    """Return high-level P/L statistics from events.json."""
    events_file = _get_env_path("EVENTS_FILE", str(BASE_DIR / "events.json"))
    events = load_events(events_file)
    stats = compute_stats(events)
    stats["generated_at"] = datetime.utcnow().isoformat() + "Z"
    return JSONResponse(stats)


@app.get("/api/trades")
async def api_trades(limit: int = 100) -> Any:
    """Return recent trade events."""
    events_file = _get_env_path("EVENTS_FILE", str(BASE_DIR / "events.json"))
    events = load_events(events_file)
    events = events[-limit:] if limit > 0 else events
    return JSONResponse({"trades": events, "count": len(events)})


@app.get("/api/model-info")
async def api_model_info() -> Any:
    """Return trained model metadata."""
    model_path = _get_env_path("MODEL_PATH", str(BASE_DIR / "trading_model.joblib"))
    scaler_path = _get_env_path("SCALER_PATH", str(BASE_DIR / "trading_scaler.joblib"))
    info = get_model_info(model_path, scaler_path)
    info["generated_at"] = datetime.utcnow().isoformat() + "Z"
    return JSONResponse(info)


@app.get("/api/logs")
async def api_logs(lines: int = 200) -> Any:
    """Return tail of trader.log."""
    log_file = _get_env_path("LOG_FILE", str(BASE_DIR / "trader.log"))
    log_lines = tail_log(log_file, max_lines=lines)
    log_stats = get_log_stats(log_file)
    return JSONResponse({"lines": log_lines, "stats": log_stats})


@app.get("/api/chart")
async def api_chart() -> Any:
    """Return chart data points derived from trade events.

    Constructs a simple price timeline by linking trade entry and exit prices.
    Each trade becomes two points: entry at timestamp T, exit at timestamp T+1.
    """
    events_file = _get_env_path("EVENTS_FILE", str(BASE_DIR / "events.json"))
    events = load_events(events_file)
    points: list[dict[str, Any]] = []
    markers: list[dict[str, Any]] = []

    if not events:
        return JSONResponse({"points": points, "markers": markers})

    sorted_events = sorted(events, key=lambda e: e.get("timestamp", ""))

    for idx, event in enumerate(sorted_events):
        raw_ts = event.get("timestamp") or event.get("time") or ""
        if isinstance(raw_ts, (int, float)):
            ts = datetime.utcfromtimestamp(float(raw_ts) / 1000.0).isoformat() + "Z"
        else:
            ts = str(raw_ts)
        entry = event.get("entryPrice")
        close = event.get("closePrice")
        trade_type = event.get("type", "Buy")
        gross_profit = event.get("grossProfit", 0)

        if entry is None:
            continue

        points.append({"x": ts, "y": float(entry)})
        markers.append({
            "x": ts,
            "y": float(entry),
            "type": "entry",
            "trade_type": trade_type,
            "index": idx,
        })

        if close is not None:
            points.append({"x": ts, "y": float(close)})
            markers.append({
                "x": ts,
                "y": float(close),
                "type": "exit",
                "trade_type": trade_type,
                "pnl": gross_profit,
                "index": idx,
            })

    return JSONResponse({"points": points, "markers": markers})


@app.get("/api/trader/status")
async def api_trader_status() -> Any:
    """Return the current status of the trader container."""
    client = _get_docker_client()
    if client is None:
        return JSONResponse({"status": "unavailable", "detail": "Docker client not available"})

    container_name = _get_trader_container_name()
    try:
        container = client.containers.get(container_name)
        status = container.status
        return JSONResponse({"status": status, "container": container_name})
    except Exception as exc:
        log.warning("Failed to get trader container status: %s", exc)
        return JSONResponse({"status": "not_found", "detail": str(exc)})


@app.post("/api/trader/start")
async def api_trader_start() -> Any:
    """Start the trader container."""
    client = _get_docker_client()
    if client is None:
        return JSONResponse({"ok": False, "detail": "Docker client not available"})

    container_name = _get_trader_container_name()
    try:
        container = client.containers.get(container_name)
        if container.status == "running":
            return JSONResponse({"ok": True, "status": "running", "detail": "Already running"})
        container.start()
        return JSONResponse({"ok": True, "status": "starting", "detail": "Start signal sent"})
    except Exception as exc:
        log.exception("Failed to start trader container")
        return JSONResponse({"ok": False, "detail": str(exc)})


@app.post("/api/trader/stop")
async def api_trader_stop() -> Any:
    """Stop the trader container."""
    client = _get_docker_client()
    if client is None:
        return JSONResponse({"ok": False, "detail": "Docker client not available"})

    container_name = _get_trader_container_name()
    try:
        container = client.containers.get(container_name)
        if container.status != "running":
            return JSONResponse({"ok": True, "status": container.status, "detail": "Not running"})
        container.stop()
        return JSONResponse({"ok": True, "status": "stopping", "detail": "Stop signal sent"})
    except Exception as exc:
        log.exception("Failed to stop trader container")
        return JSONResponse({"ok": False, "detail": str(exc)})


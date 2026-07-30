# Implementation Plan: Docker Setup + Web Dashboard for cTraderMCP

## Objective
Add a Dockerized deployment and a modern web dashboard (port 8055) that visualizes:
- Trained model metadata
- Price chart with trade entry/exit markers
- Live log streaming
- High-level P/L statistics and trade counts from events.json

## Directory Structure
```
55-cTraderMCP/
├── docker/
│   ├── Dockerfile.trader
│   └── Dockerfile.dashboard
├── dashboard/
│   ├── __init__.py
│   ├── main.py            # FastAPI backend
│   ├── index.html         # SPA dashboard
│   └── static/
│       ├── css/
│       │   └── dashboard.css
│       └── js/
│           └── dashboard.js
├── docker-compose.yml
├── requirements-dashboard.txt
└── ...
```

## Step 1: Docker Setup
- Create `docker/Dockerfile.trader` based on `debian:12-slim` with Python 3.10
- Create `docker/Dockerfile.dashboard` based on `debian:12-slim` with Python 3.10 + FastAPI/uvicorn
- Create `docker-compose.yml` with two services: `trader` and `dashboard`
- Use volume mounts for `events.json`, `trader.log`, `trade_history.json`, `trading_model.joblib`, `trading_scaler.joblib`
- Expose port 8055 for dashboard, keep existing ports for trader

## Step 2: FastAPI Dashboard Backend
Endpoints:
- `GET /` — serve `index.html`
- `GET /api/stats` — P/L summary, trade count, win rate from events.json
- `GET /api/trades` — recent trade list with entry/exit prices, P/L
- `GET /api/model-info` — model metadata (type, training date, feature importance if available)
- `GET /api/logs` — tail of trader.log
- `GET /api/candles` — historical candles for chart (fallback to static sample if none)

## Step 3: Frontend Dashboard
- Single-page app with Tailwind CSS (CDN) + Chart.js
- Sections:
  1. **Header**: Project name, version, deployment datetime
  2. **Stats Cards**: Total trades, win rate, net P/L, avg P/L per trade
  3. **Chart**: Candlestick-style line chart with markers for trade entries (green ▲) and exits (red ▼)
  4. **Model Info**: Model type, training sample count, last trained timestamp
  5. **Logs**: Scrollable log viewer with auto-refresh

## Step 4: Data Adapters
- `dashboard/stats.py` — compute stats from events.json
- `dashboard/model_info.py` — inspect joblib model metadata
- `dashboard/log_reader.py` — tail trader.log safely

## Step 5: Integration
- Update `TraderAI.py` to optionally expose metrics endpoint OR keep dashboard as separate reader
- Ensure file paths are configurable via env vars for Docker
- Update `.gitignore` for Docker artifacts if needed

## Step 6: Documentation
- Update README.md with Docker instructions
- Update ARCHITECTURE.md with dashboard service

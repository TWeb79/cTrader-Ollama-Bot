# US500 AI Trading Assistant

## Overview
This project implements an AI-powered trading assistant that connects to a cTrader MCP server and uses a local ML model (GradientBoostingClassifier) to make trading decisions on the US500 instrument (S&P 500 index). It replaces the previous Ollama LLM approach with a model trained on historical trade outcomes.

## Features
- Connects to cTrader MCP server via HTTP transport
- Retrieves historical price data (6 months of candles) and trade events
- Trains a local ML model on historical trade outcomes
- Uses the trained model for live BUY/SELL/HOLD/TRAIL_SL/CLOSE decisions
- Implements a feedback loop that records outcomes and retrains periodically
- Includes risk management with configurable stop-loss and take-profit
- Dry-run mode for safe testing

## Port Allocations (project 55)

| Port | Service Type | Purpose |
| ---- | ------------ | ------- |
| 8055 | Web dashboard | Project web UI |
| 8155 | FastAPI service | API backend |
| 8255 | Database | Data store |
| 8955 | LLM / ML service | Local ML model inference |

## Prerequisites
Before running this script, ensure you have:

1. **Python 3.10+** installed (tested with 3.10.20)
2. **cTrader Desktop** with MCP server running (default: http://127.0.0.1:9876/mcp)
3. Required Python packages (see Installation)

## Installation

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start cTrader MCP Server
Ensure your cTrader Desktop application is running and the MCP server is active on port 9876.

### 3. Verify ports.env
Check `ports.env` for the project's port assignments.

## Configuration

Edit `trader_config.py` to match your setup. Key settings:

```python
CONFIG = {
    "mcp_transport": "http",
    "mcp_url": "http://127.0.0.1:9876/mcp",
    "ml_model_path": "trading_model.joblib",
    "ml_retrain_interval": 10,
    "ml_min_samples": 50,
    "symbol": "US500",
    "history_months": 6,
    "candle_timeframe": "m5",
    "dry_run": True,
}
```

## Usage

### 1. First Run (Recommended - Dry Run)
```bash
python3.10 TraderAI.py
```

The script will:
1. Load historical candles and trade events
2. Train the ML model (or load a saved model)
3. Enter the live monitoring loop
4. Log all decisions to `trader.log`

### 2. Live Trading (After Verification)
Change `"dry_run": False` in the CONFIG and ensure you are using a demo account.

### 3. Test Run
Set `"max_loop_iterations": 5` in CONFIG for a limited test run.

## How It Works

1. **Data Preparation** — Loads historical OHLCV candles from cTrader MCP and trade events from `events.json`.
2. **Model Training** — Computes features (SMA20/50, RSI14, price ratios) and trains a GradientBoostingClassifier on outcomes.
3. **Integration** — In the live loop, fetches current price, computes indicators, and uses the trained model to predict BUY/SELL/HOLD.
4. **Feedback Loop** — Records each trade's outcome (profit, pips) and triggers retraining when the retrain interval is reached.

## Safety Features

- **Dry-run Mode**: Default mode logs intended trades without executing
- **Position Limits**: Configurable maximum concurrent positions
- **Risk Parameters**: Adjustable stop-loss and take-profit levels
- **Manual Review**: All trading decisions are logged for review

## Project Files

| File | Purpose |
| ---- | ------- |
| TraderAI.py | Main orchestrator (4-step ML pipeline) |
| trader_config.py | Configuration and port mappings |
| data_preparation.py | Step 1: load and format training data |
| ai_model.py | Step 2: train model + Step 3: predict |
| feedback_loop.py | Step 4: record outcomes and retrain |
| trade_executor.py | cTrader MCP client wrapper |
| trade_logger.py | Trade state persistence (events.json) |
| volume_profile_strategy.py | Volume profile strategy module |

## Important Disclaimers

⚠️ **THIS IS NOT FINANCIAL ADVICE**
This tool is a technical demonstration only. Trading involves substantial risk of loss. Past performance is not indicative of future results.

⚠️ **ALWAYS TEST ON DEMO FIRST**
Never run with real money until you have thoroughly tested in dry-run mode.

## License
This project is provided for educational purposes only. Use at your own risk.
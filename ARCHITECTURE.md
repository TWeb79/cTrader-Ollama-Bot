# Architecture — US500 AI Trading Assistant (project 55)

## System Overview

The US500 AI Trading Assistant replaces the previous Ollama LLM approach with a
local ML model (GradientBoostingClassifier) trained on historical trade data.
The system connects to a cTrader MCP server for live market data and order
execution, and uses a four-step ML pipeline for decision-making.

## Module Responsibilities

```
TraderAI.py          → Main orchestrator (plan.md 4-step pipeline)
trader_config.py     → Configuration, port mappings (project 55)
data_preparation.py  → Step 1: Load candles & trade events for training
ai_model.py          → Step 2: Train ML model + Step 3: Live predictions
feedback_loop.py     → Step 4: Record outcomes & trigger retraining
trade_executor.py    → cTrader MCP client wrapper + TOOL_NAMES mapping
trade_logger.py      → Trade state persistence (events.json formatting)
volume_profile_strategy.py → Existing volume profile strategy module
```

## Data Flow

1. cTrader MCP Server → fetch_historical_candles() → raw candle data
2. events.json → load_events() → historical trade outcomes
3. candles + events → ModelTrainer.prepare_training_data() → feature matrix X, labels y
4. ModelTrainer.train(X, y) → trained GradientBoostingClassifier → trading_model.joblib
5. Live price + indicators → ModelTrainer.predict() → BUY/SELL/HOLD signal
6. Trade outcome → feedback_loop.process_trade_feedback() → retrain on schedule

## Port Allocation (project 55)

| Port | Service Type | Purpose |
| ---- | ------------ | ------- |
| 8055 | Web dashboard | Project web UI (if added) |
| 8155 | FastAPI service | API layer (if added) |
| 8255 | Database | Data store (if added) |
| 8955 | LLM / ML service | Local ML model inference |

## External Dependencies

- **cTrader MCP Server**: Provides market data and order execution via HTTP transport
- **scikit-learn**: GradientBoostingClassifier for trade outcome prediction
- **pandas**: OHLCV candle data manipulation and indicator computation
- **numpy**: Numerical operations for feature arrays
- **joblib**: Model serialization/deserialization

## Safety

- `dry_run=True` by default — logs intended trades without executing
- `max_open_positions` limits concurrent positions
- Model confidence threshold prevents low-certainty trades
- All decisions are logged to `trader.log`
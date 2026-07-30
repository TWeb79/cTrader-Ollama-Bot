# US500 AI Trading Assistant

## Overview
This project implements an AI-powered trading assistant that connects to a cTrader MCP server and uses a local Ollama LLM to make trading decisions on the US500 instrument (S&P 500 index).

## Features
- Connects to cTrader MCP server via HTTP transport
- Retrieves historical price data (6 months of daily candles)
- Calculates technical indicators (SMA20, SMA50, RSI14)
- Uses Ollama LLM to generate a rule-based trading strategy
- Monitors live prices and executes trades based on AI signals
- Includes risk management with configurable stop-loss and take-profit
- Dry-run mode for safe testing

## Prerequisites
Before running this script, ensure you have:

1. **Python 3.10+** installed (tested with 3.10.20)
2. **cTrader Desktop** with MCP server running (default: http://127.0.0.1:9876/mcp)
3. **Ollama** running locally with a capable model installed
4. Required Python packages: `mcp`, `ollama`, `pandas`

## Installation

### 1. Install Python Dependencies
```bash
pip install mcp ollama pandas
```

### 2. Install and Configure Ollama
```bash
# Install Ollama (if not already installed)
# Visit https://ollama.com/download for installation instructions

# Pull the recommended model
ollama pull qwen3.5:9b

# Verify the model is available
ollama list
```

### 3. Start cTrader MCP Server
Ensure your cTrader Desktop application is running and the MCP server is active on port 9876.

## Configuration

Edit the `CONFIG` section in `Trader.py` to match your setup:

```python
CONFIG = {
    # --- cTrader MCP server connection ---
    "mcp_transport": "http",               # HTTP transport for cTrader Desktop MCP server
    "mcp_url": "http://127.0.0.1:9876/mcp",

    # --- Ollama ---
    "ollama_model": "qwen3.5:9b",          # Ensure this model is pulled in Ollama
    "ollama_host": "http://localhost:11434",

    # --- Instrument & analysis ---
    "symbol": "US500",                     # S&P 500 index
    "history_months": 6,                   # Lookback period for strategy formation
    "candle_timeframe": "d1",              # Daily candles

    # --- Live monitoring loop ---
    "poll_interval_seconds": 60,           # How often to check price (seconds)
    "max_loop_iterations": 1,              # Set to None for continuous running, or integer for test runs

    # --- Risk management ---
    "trade_volume": 0.1,                   # Position size in lots
    "stop_loss_points": 15.0,              # Stop loss in price points
    "take_profit_points": 30.0,            # Take profit in price points
    "max_open_positions": 1,               # Maximum concurrent positions

    # --- Safety switch ---
    "dry_run": True,                       # Set to False for live trading (USE WITH EXTREME CAUTION)
}
```

## Usage

### 1. First Run (Recommended - Dry Run)
For your first execution, keep `dry_run = True` to see what the bot would do without placing real trades:

```bash
python3.10 Trader.py
```

You should see output showing:
- Connection to cTrader MCP server
- List of available tools
- Historical data retrieval
- Indicator calculations
- AI-generated strategy
- Simulated trade decisions (in dry-run mode)

### 2. Live Trading (After Verification)
⚠️ **WARNING**: Only proceed to live trading after thoroughly testing in dry-run mode and verifying on a demo account.

1. Change `"dry_run": False` in the CONFIG
2. Ensure you're using a demo account in cTrader
3. Run the script:
```bash
python3.10 Trader.py
```

### 3. Test Run
To run a limited number of iterations for testing:
- Set `"max_loop_iterations": 5` in CONFIG
- This will run the monitoring loop 5 times before exiting

## How It Works

1. **Initialization**: Connects to cTrader MCP server and discovers available tools
2. **Strategy Formation**:
   - Downloads 6 months of daily price data for US500
   - Calculates SMA-20, SMA-50, and RSI-14 indicators
   - Sends indicator summary to Ollama LLM to generate a trading strategy
3. **Live Trading Loop**:
   - Periodically checks current price (every 60 seconds by default)
   - Asks Ollama whether to BUY, SELL, or HOLD based on the strategy
   - If signal is BUY/SELL and position limits allow, places a market order
   - Orders include stop-loss and take-profit levels
   - Continues until interrupted or max iterations reached

## Safety Features

- **Dry-run Mode**: Default mode logs intended trades without executing them
- **Position Limits**: Configurable maximum concurrent positions
- **Risk Parameters**: Adjustable stop-loss and take-profit levels
- **Manual Review**: All trading decisions are logged for review

## Important Disclaimers

⚠️ **THIS IS NOT FINANCIAL ADVICE**  
This tool is a technical demonstration only. Trading involves substantial risk of loss. Past performance is not indicative of future results.

⚠️ **ALWAYS TEST ON DEMO FIRST**  
Never run with real money until you have:
- Verified the bot's behavior in dry-run mode
- Tested extensively on a demo account
- Understood all risks involved
- Started with minimal position sizes

⚠️ **MONITOR CONTINUOUSLY**  
Automated trading systems require ongoing supervision. Be prepared to intervene manually if needed.

## Troubleshooting

### Connection Issues
- Verify cTrader Desktop is running and MCP server is accessible at http://127.0.0.1:9876/mcp
- Check firewall settings allowing localhost connections
- Try accessing the URL directly in a browser or with curl

### Model Issues
- Ensure Ollama is running: `ollama serve`
- Verify model is available: `ollama list`
- Try pulling the model again: `ollama pull qwen3.5:9b`

### Dependency Problems
- Use Python 3.10 specifically: `python3.10 Trader.py`
- Create a virtual environment if experiencing package conflicts
- Reinstall packages: `pip install --force-reinstall mcp ollama pandas`

### No Trades Occurring
- Check if market is open (US500 trades during NYSE hours)
- Verify signal generation logic in the logs
- Ensure position limits haven't been reached
- Confirm price data is being received correctly

## Customization

### Adjusting Strategy Parameters
- Modify `history_months` for different lookback periods
- Change `candle_timeframe` to use different timeframes (m5, h1, etc.)
- Adjust `stop_loss_points` and `take_profit_points` for risk tolerance

### Changing Instruments
- Update `"symbol": "US500"` to other supported symbols (check cTrader for available instruments)
- Note: Different instruments may have different point/pip values

## License
This project is provided for educational purposes only. Use at your own risk.

## Support
For issues or questions, please refer to the project documentation or consult with a qualified financial advisor before using this tool for live trading.# cTrader-Ollama-Bot

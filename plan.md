# Implementation Plan: Ollama SL/TP Recommendation + Trailing SL Fix

## Objective
Implement one-time Ollama-based stop-loss and take-profit recommendation immediately after a position is opened, then use only trailing stop-loss for position management.

## Changes Required

### 1. Fix Trailing SL Bug
- In `run_volume_profile_strategy` (line ~824): change `close_position` to `amend_position`
- In `main()` ollama loop (line ~1081): change `close_position` to `amend_position`

### 2. Add `ask_ollama_for_sl_tp_recommendation` Function
- Create a new Ollama prompt function that returns JSON: `{"stop_loss": float, "take_profit": float, "reasoning": str}`
- Inputs: model, symbol, current_price, position details, strategy, intraday candles
- Outputs: recommended absolute SL/TP price levels

### 3. Integrate in Volume Profile Strategy Path
- After placing a volume profile trade (line ~914) and magnet trade (line ~939):
  - Fetch updated positions to get new position details
  - If position not yet processed for Ollama SL/TP:
    - Call `ask_ollama_for_sl_tp_recommendation`
    - Amend position with recommended SL and TP
    - Mark position as processed
- Track processed positions with a set (`ollama_sl_tp_processed`)

### 4. Integrate in Main Ollama Strategy Path
- After `place_trade` in main loop (line ~1115):
  - Capture trade result and extract position ID
  - Fetch updated positions
  - If position not yet processed for Ollama SL/TP:
    - Call `ask_ollama_for_sl_tp_recommendation`
    - Amend position with recommended SL and TP
    - Mark position as processed
- Track processed positions with a set (`ollama_sl_tp_processed`)

### 5. Position Management Behavior
- After initial Ollama SL/TP is applied, position management only trails SL
- No further TP adjustments after initial recommendation
- Existing `ask_ollama_for_position_management` already returns TRAIL_SL/CLOSE/HOLD — no changes needed there

## Verification
- Run lint/typecheck if available
- Review logic for both strategy paths
- Confirm dry_run logging is correct

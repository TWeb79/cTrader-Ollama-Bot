import json
import ollama
from datetime import datetime, timedelta

# Simulate what the indicators might look like
indicators = {
    'last_close': 7300.5,
    '6m_high': 7800.0,
    '6m_low': 6500.0,
    'sma20': 7250.0,
    'sma50': 7100.0,
    'rsi14': 65.5,
    'trend_20_vs_50': 'bullish'
}

print('Testing Ollama call...')
start = datetime.now()
try:
    response = ollama.chat(
        model='qwen3.5:9b',
        messages=[
            {'role': 'system', 'content': 'You are a cautious quantitative trading assistant. Given summarized market indicators, propose ONE simple, rule-based intraday strategy. Respond ONLY with valid JSON, no prose, no markdown fences, matching this schema: {"bias": "long"|"short"|"neutral", "reasoning": "short string", "entry_rule": "short string describing the price condition to enter", "invalidate_if": "short string describing when to skip/cancel", "confidence": 0-1 float}'},
            {'role': 'user', 'content': 'Symbol: US500\n6-month indicator summary:\n' + json.dumps(indicators, indent=2)}
        ],
        options={'temperature': 0.1}
    )
    end = datetime.now()
    print(f'Call took {(end-start).total_seconds():.2f} seconds')
    print(f'Response: {response}')
    if 'message' in response:
        content = response['message']['content'].strip()
        print(f'Content: {content}')
        content = content.removeprefix('```json').removeprefix('```').removesuffix('```').strip()
        print(f'Cleaned content: {content}')
        try:
            result = json.loads(content)
            print(f'Parsed result: {result}')
        except json.JSONDecodeError as e:
            print(f'JSON decode error: {e}')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()

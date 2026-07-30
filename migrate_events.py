#!/usr/bin/env python3
"""Migrate existing events.json to match events2.json format.

Fixes:
1. Converts ISO 8601 time strings to integer milliseconds.
2. Removes duplicate Create Position events for the same positionId.
3. Resets serial numbers to start at 0 and be sequential.
4. Ensures all numeric fields are proper numbers, not strings.
"""

import json
from datetime import datetime
from pathlib import Path


EVENTS_FILE = Path("events.json")
REFERENCE_FILE = Path("events 2.json")


def iso_to_ms(val) -> int:
    """Convert ISO 8601 string or int to milliseconds since epoch."""
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        try:
            dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
            return int(dt.timestamp() * 1000)
        except (ValueError, TypeError):
            pass
    return 0


def migrate():
    if not EVENTS_FILE.exists():
        print(f"ERROR: {EVENTS_FILE} not found")
        return

    with open(EVENTS_FILE, "r", encoding="utf-8") as f:
        events = json.load(f)

    if not isinstance(events, list):
        print("ERROR: events.json is not a JSON array")
        return

    seen_position_creates = set()
    migrated = []

    for event in events:
        # Skip duplicate Create Position events for the same positionId
        if event.get("event") == "Create Position":
            pos_id = event.get("positionId")
            if pos_id in seen_position_creates:
                continue
            seen_position_creates.add(pos_id)

        # Fix time to int milliseconds
        event = dict(event)
        event["time"] = iso_to_ms(event.get("time", 0))

        # Ensure numeric fields are numbers, not strings
        for field in ("serial", "positionId", "volume", "quantity",
                      "entryPrice", "grossProfit", "pips"):
            val = event.get(field)
            if isinstance(val, str):
                try:
                    event[field] = float(val) if "." in val else int(val)
                except (ValueError, TypeError):
                    pass

        for field in ("tp", "sl", "closePrice", "balance", "equity"):
            val = event.get(field)
            if isinstance(val, str) and val.lower() != "null":
                try:
                    event[field] = float(val) if "." in val else int(val)
                except (ValueError, TypeError):
                    pass
            elif isinstance(val, str) and val.lower() == "null":
                event[field] = None

        migrated.append(event)

    # Reset serials to start at 0, sequential
    for idx, event in enumerate(migrated):
        event["serial"] = idx

    with open(EVENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(migrated, f, indent=2, default=str)

    print(f"Migrated {len(events)} events -> {len(migrated)} events")
    print(f"Removed {len(events) - len(migrated)} duplicate Create Position events")
    print(f"Serial range: 0 - {len(migrated) - 1}")


if __name__ == "__main__":
    migrate()

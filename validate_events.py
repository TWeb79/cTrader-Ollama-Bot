#!/usr/bin/env python3
"""Validator for cTrader events.json format.

Compares events.json against the reference events2.json schema and
reports any structural, typing, or formatting mismatches.
"""

import json
import sys
from pathlib import Path

EXPECTED_KEYS = [
    "serial",
    "orderId",
    "positionId",
    "event",
    "time",
    "volume",
    "quantity",
    "type",
    "entryPrice",
    "tp",
    "sl",
    "closePrice",
    "grossProfit",
    "pips",
    "balance",
    "equity",
]

ALLOWED_EVENTS = {
    "Create Position",
    "Position Modified (S/L)",
    "Stop Loss Hit",
    "Take Profit",
    "Position closed",
}


def load_events(path: str) -> list:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array, got {type(data).__name__}")
    return data


def validate_events(events: list, strict: bool = True) -> list:
    errors = []

    if not events:
        errors.append("Events list is empty")
        return errors

    serials = []
    for idx, event in enumerate(events):
        if not isinstance(event, dict):
            errors.append(f"Event {idx}: not a dict")
            continue

        keys = list(event.keys())
        if keys != EXPECTED_KEYS:
            errors.append(
                f"Event {idx} (serial={event.get('serial')}): "
                f"key order mismatch. Expected {EXPECTED_KEYS}, got {keys}"
            )

        for key in EXPECTED_KEYS:
            if key not in event:
                errors.append(f"Event {idx}: missing key '{key}'")

        serial = event.get("serial")
        if not isinstance(serial, int):
            errors.append(f"Event {idx}: serial must be int, got {type(serial).__name__}")
        else:
            serials.append(serial)

        time_val = event.get("time")
        if not isinstance(time_val, int):
            errors.append(
                f"Event {idx}: time must be int (milliseconds), "
                f"got {type(time_val).__name__} = {time_val!r}"
            )

        event_type = event.get("event")
        if event_type not in ALLOWED_EVENTS:
            errors.append(
                f"Event {idx}: event type '{event_type}' not in allowed set {ALLOWED_EVENTS}"
            )

        numeric_fields = [
            "positionId",
            "volume",
            "quantity",
            "entryPrice",
            "grossProfit",
            "pips",
        ]
        for field in numeric_fields:
            val = event.get(field)
            if val is not None and not isinstance(val, (int, float)):
                errors.append(
                    f"Event {idx}: {field} must be numeric or null, "
                    f"got {type(val).__name__}"
                )

        nullable_numeric = ["tp", "sl", "closePrice", "balance", "equity"]
        for field in nullable_numeric:
            val = event.get(field)
            if val is not None and not isinstance(val, (int, float)):
                errors.append(
                    f"Event {idx}: {field} must be numeric or null, "
                    f"got {type(val).__name__}"
                )

        if event.get("orderId") is not None and not isinstance(event.get("orderId"), int):
            errors.append(
                f"Event {idx}: orderId must be int or null, "
                f"got {type(event.get('orderId')).__name__}"
            )

        type_val = event.get("type")
        if type_val not in ("Buy", "Sell"):
            errors.append(
                f"Event {idx}: type must be 'Buy' or 'Sell', got {type_val!r}"
            )

    if serials:
        if strict:
            if serials[0] != 0:
                errors.append(f"Serial does not start at 0 (first serial={serials[0]})")
            expected = list(range(serials[0], serials[0] + len(serials)))
            if serials != expected:
                errors.append(
                    f"Serial numbers are not sequential starting from {serials[0]}. "
                    f"Got {serials[:10]}..."
                )

        dupes = [s for s in set(serials) if serials.count(s) > 1]
        if dupes:
            errors.append(f"Duplicate serial numbers found: {dupes}")

    return errors


def compare_with_reference(events: list, ref_path: str) -> list:
    ref_events = load_events(ref_path)
    diffs = []

    ref_keys = list(ref_events[0].keys()) if ref_events else []
    if events:
        actual_keys = list(events[0].keys())
        if actual_keys != ref_keys:
            diffs.append(
                f"Key order mismatch vs reference. "
                f"Reference: {ref_keys}, Actual: {actual_keys}"
            )

    ref_types = sorted(set(e.get("event") for e in ref_events))
    actual_types = sorted(set(e.get("event") for e in events))
    if set(actual_types) - set(ref_types):
        diffs.append(
            f"Event type(s) not present in reference: "
            f"{sorted(set(actual_types) - set(ref_types))}"
        )

    ref_time_types = {type(e.get("time")).__name__ for e in ref_events[:10]}
    actual_time_types = {type(e.get("time")).__name__ for e in events[:10]}
    if ref_time_types != actual_time_types:
        diffs.append(
            f"Time type mismatch. Reference: {ref_time_types}, Actual: {actual_time_types}"
        )

    ref_serial_start = ref_events[0].get("serial") if ref_events else None
    actual_serial_start = events[0].get("serial") if events else None
    if ref_serial_start != actual_serial_start:
        diffs.append(
            f"Serial start mismatch. Reference: {ref_serial_start}, Actual: {actual_serial_start}"
        )

    return diffs


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Validate cTrader events.json format")
    parser.add_argument("file", help="Path to events.json to validate")
    parser.add_argument(
        "--reference",
        default="events 2.json",
        help="Path to reference events2.json (default: 'events 2.json')",
    )
    parser.add_argument(
        "--no-strict-serial",
        action="store_true",
        help="Skip strict serial sequential check",
    )
    args = parser.parse_args()

    try:
        events = load_events(args.file)
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    errors = validate_events(events, strict=not args.no_strict_serial)

    ref_path = Path(args.reference)
    if ref_path.exists():
        diffs = compare_with_reference(events, args.reference)
        if diffs:
            print("=== Differences from reference ===")
            for d in diffs:
                print(f"  - {d}")
            errors.extend(diffs)

    if errors:
        print("=== Validation FAILED ===")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print("=== Validation PASSED ===")
        print(f"Validated {len(events)} events against reference format.")


if __name__ == "__main__":
    main()

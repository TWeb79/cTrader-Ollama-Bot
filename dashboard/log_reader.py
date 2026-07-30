"""Read and tail log files for dashboard display."""

import logging
import os
from typing import Any

log = logging.getLogger("ai_trader.dashboard.log_reader")


def tail_log(filepath: str, max_lines: int = 200) -> list[str]:
    """Return the last N lines of a log file.

    Args:
        filepath: Path to log file.
        max_lines: Maximum number of lines to return.

    Returns:
        List of log lines.
    """
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
        return lines[-max_lines:]
    except OSError as exc:
        log.warning("Failed to read log file %s: %s", filepath, exc)
        return []


def get_log_stats(filepath: str) -> dict[str, Any]:
    """Return basic stats about the log file.

    Args:
        filepath: Path to log file.

    Returns:
        Dict with line count, file size, etc.
    """
    stats: dict[str, Any] = {
        "path": filepath,
        "exists": os.path.exists(filepath),
        "line_count": 0,
        "size_bytes": 0,
    }
    if not stats["exists"]:
        return stats
    try:
        stat = os.stat(filepath)
        stats["size_bytes"] = stat.st_size
        with open(filepath, "r", encoding="utf-8") as fh:
            stats["line_count"] = sum(1 for _ in fh)
    except OSError as exc:
        log.warning("Failed to stat log file %s: %s", filepath, exc)
    return stats

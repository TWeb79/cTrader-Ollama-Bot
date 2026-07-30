"""Trade logging utilities — persist trade state as cTrader events.json format.

Provides helpers for timestamp generation, event serialization,
and persisting trade events to disk.

Author: Inventions4All - github:TWeb79
Version: 1.0.0  (deployment: 2026-07-30T17:53:53+02:00)
"""

import json
import logging
import os
from contextlib import AsyncExitStack
from datetime import datetime
from typing import Any, Optional

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

try:
    import httpx2
except ImportError:
    httpx2 = None

log = logging.getLogger("ai_trader.logger")

EVENTS_LOG_FILE = "events.json"

TOOL_NAMES = {
    "historical_candles": "get_trendbars",
    "current_price": "get_spot_prices",
    "place_order": "place_market_order",
    "list_positions": "get_positions",
    "close_position": "close_position",
    "list_pending_orders": "get_pending_orders",
    "list_deals": "get_deals",
    "list_order_history": "get_order_history",
    "amend_position": "amend_position",
}


class CTraderMCPClient:
    """Thin wrapper around an MCP ClientSession talking to the cTrader MCP server."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._stack = AsyncExitStack()
        self.session: Optional[ClientSession] = None
        self.available_tools: dict[str, Any] = {}

    async def connect(self) -> None:
        """Connect to the cTrader MCP server and discover available tools."""
        if self.config["mcp_transport"] == "http":
            mcp_url = self.config["mcp_url"]
            http_client = None

            if httpx2 is not None:
                host_from_url = mcp_url.split("//")[-1].split("/")[0].split(":")[0]
                if host_from_url in ("127.0.0.1", "localhost"):
                    host_gateway = os.environ.get("HOST_GATEWAY", "192.168.65.254")
                    host_port = mcp_url.split("://")[1].split("/")[0].split(":")[-1]
                    base_url = f"http://{host_gateway}:{host_port}"
                    http_client = httpx2.AsyncClient(
                        base_url=base_url,
                        headers={"Host": f"{host_from_url}:{host_port}"},
                    )
                    mcp_url = mcp_url.replace(
                        f"{host_from_url}:{host_port}",
                        f"{host_gateway}:{host_port}",
                        1,
                    )

            if http_client is not None:
                read, write = await self._stack.enter_async_context(
                    streamable_http_client(mcp_url, http_client=http_client),
                )
            else:
                read, write = await self._stack.enter_async_context(
                    streamable_http_client(mcp_url),
                )
        else:
            raise ValueError(
                f"Unknown mcp_transport: {self.config['mcp_transport']}",
            )

        self.session = await self._stack.enter_async_context(
            ClientSession(read, write),
        )
        await self.session.initialize()

        tools_result = await self.session.list_tools()
        self.available_tools = {t.name: t for t in tools_result.tools}
        log.info("Connected to cTrader MCP server. Available tools:")
        for name, tool in self.available_tools.items():
            desc = (tool.description or "").strip().replace("\n", " ")[:100]
            log.info("  - %s: %s", name, desc)

    async def call(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool on the cTrader MCP server.

        Args:
            tool_name: Logical tool name mapped via TOOL_NAMES.
            arguments: Dict of arguments for the tool call.

        Returns:
            Parsed tool result (dict or list).

        Raises:
            RuntimeError: If the tool is not available on the server.
        """
        if tool_name not in self.available_tools:
            raise RuntimeError(
                f"Tool '{tool_name}' not found. "
                f"Available: {list(self.available_tools.keys())}. "
                "Update TOOL_NAMES to match your server.",
            )
        result = await self.session.call_tool(tool_name, arguments)
        for block in result.content:
            if hasattr(block, "text"):
                try:
                    return json.loads(block.text)
                except (json.JSONDecodeError, TypeError):
                    return block.text
        return result

    async def close(self) -> None:
        """Close the MCP connection."""
        await self._stack.aclose()


def _ms_timestamp() -> int:
    return int(datetime.utcnow().timestamp() * 1000)


def _last_serial(filepath: str = EVENTS_LOG_FILE) -> int:
    """Return the highest serial number in the events file."""
    import os as _os
    if not _os.path.exists(filepath):
        return -1
    try:
        with open(filepath, "r", encoding="utf-8") as fh:
            events = json.load(fh)
        if events:
            return max(e.get("serial", 0) for e in events)
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    return -1


def _coerce_int_time(val: Any, fallback: int) -> int:
    """Ensure a timestamp value is an integer (milliseconds)."""
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
            return int(dt.timestamp() * 1000)
        except (ValueError, TypeError):
            return fallback
    return fallback


def _position_to_event(
    position: dict[str, Any],
    serial: int,
    event_type: str = "Create Position",
) -> dict[str, Any]:
    """Convert a position dict to a cTrader events.json compatible event."""
    pos_type = position.get("type", position.get("side", "Buy"))
    if isinstance(pos_type, str):
        pos_type = pos_type.capitalize()
    if pos_type not in ("Buy", "Sell"):
        pos_type = "Buy"
    return {
        "serial": serial,
        "orderId": None,
        "positionId": position.get("id", position.get("positionId", 0)),
        "event": event_type,
        "time": _ms_timestamp(),
        "volume": position.get("volume", position.get("currentVolume", 0)),
        "quantity": position.get("volume", position.get("currentVolume", 0)),
        "type": pos_type,
        "entryPrice": position.get("entryPrice", position.get("openPrice", 0)),
        "tp": position.get("takeProfit", position.get("tp", None)),
        "sl": position.get("stopLoss", position.get("sl", None)),
        "closePrice": None,
        "grossProfit": 0,
        "pips": 0,
        "balance": None,
        "equity": None,
    }


async def log_trades(client: CTraderMCPClient, config: dict[str, Any]) -> None:
    """Fetch current trade state from cTrader MCP and persist as events.json.

    Creates CREATE, MODIFY, and CLOSE events for positions, pending orders,
    deals, and order history entries filtered to the configured symbol.

    Args:
        client: Connected cTrader MCP client.
        config: Configuration dictionary with symbol and dry_run flag.
    """
    import os as _os

    existing_events: list[dict[str, Any]] = []
    if _os.path.exists(EVENTS_LOG_FILE):
        try:
            with open(EVENTS_LOG_FILE, "r", encoding="utf-8") as fh:
                existing_events = json.load(fh)
        except (json.JSONDecodeError, ValueError):
            existing_events = []

    next_serial = _last_serial(EVENTS_LOG_FILE) + 1
    new_events: list[dict[str, Any]] = []
    now_ms = _ms_timestamp()
    symbol = config.get("symbol", "US500")

    positions = await _fetch_positions(client)
    pending = await _fetch_pending_orders(client)
    deals = await _fetch_deals(client)
    history = await _fetch_order_history(client)

    # Build position events
    for pos in positions:
        if str(pos.get("symbol", pos.get("symbolName", ""))).upper() != symbol:
            continue
        pos_id = pos.get("id", pos.get("positionId", 0))
        existing_create_ids = {
            str(e.get("positionId"))
            for e in existing_events
            if e.get("event") == "Create Position"
        }
        if str(pos_id) not in existing_create_ids:
            new_events.append(_position_to_event(pos, next_serial, "Create Position"))
            next_serial += 1

    # Build pending order events
    for order in pending:
        if str(order.get("symbol", order.get("symbolName", ""))).upper() != symbol:
            continue
        order_id = order.get("id", order.get("orderId", 0))
        existing_pending_ids = {str(e.get("positionId")) for e in existing_events}
        if str(order_id) not in existing_pending_ids:
            new_events.append(_position_to_event(order, next_serial, "Create Position"))
            next_serial += 1

    # Build deal events
    for deal in deals:
        if str(deal.get("symbol", deal.get("symbolName", ""))).upper() != symbol:
            continue
        deal_id = deal.get("id", deal.get("dealId", 0))
        if str(deal_id) not in {str(e.get("serial")) for e in existing_events}:
            event = _build_deal_event(deal, next_serial, now_ms)
            if event is not None:
                new_events.append(event)
                next_serial += 1

    # Build order history events
    for h_item in history:
        if str(h_item.get("symbol", h_item.get("symbolName", ""))).upper() != symbol:
            continue
        h_id = h_item.get("id", h_item.get("orderId", 0))
        if str(h_id) not in {str(e.get("serial")) for e in existing_events}:
            event = _build_history_event(h_item, next_serial, now_ms)
            if event is not None:
                new_events.append(event)
                next_serial += 1

    if new_events:
        all_events = existing_events + new_events
        with open(EVENTS_LOG_FILE, "w", encoding="utf-8") as fh:
            json.dump(all_events, fh, indent=2, default=str)
        log.info(
            "Wrote %d event(s) to %s (total: %d)",
            len(new_events),
            EVENTS_LOG_FILE,
            len(all_events),
        )


async def _fetch_positions(client: CTraderMCPClient) -> list[dict[str, Any]]:
    """Fetch open positions from the MCP server."""
    try:
        result = await client.call(TOOL_NAMES["list_positions"], {})
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return result.get("positions", result.get("items", []))
    except Exception:
        log.exception("Failed to fetch positions from MCP server")
    return []


async def _fetch_pending_orders(client: CTraderMCPClient) -> list[dict[str, Any]]:
    """Fetch pending orders from the MCP server."""
    try:
        result = await client.call(TOOL_NAMES["list_pending_orders"], {})
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return result.get("pendingOrders", result.get("items", []))
    except Exception:
        log.exception("Failed to fetch pending orders from MCP server")
    return []


async def _fetch_deals(client: CTraderMCPClient) -> list[dict[str, Any]]:
    """Fetch closed/filled deals from the MCP server."""
    try:
        result = await client.call(TOOL_NAMES["list_deals"], {})
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return result.get("deals", result.get("items", []))
    except Exception:
        log.exception("Failed to fetch deals from MCP server")
    return []


async def _fetch_order_history(client: CTraderMCPClient) -> list[dict[str, Any]]:
    """Fetch order history from the MCP server."""
    try:
        result = await client.call(TOOL_NAMES["list_order_history"], {})
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return result.get("history", result.get("items", []))
    except Exception:
        log.exception("Failed to fetch order history from MCP server")
    return []


def _build_deal_event(deal: dict[str, Any], serial: int, now_ms: int) -> Optional[dict[str, Any]]:
    """Build an event dict from a deal record."""
    pos_type = deal.get("type", deal.get("side", "Buy"))
    if isinstance(pos_type, str):
        pos_type = pos_type.capitalize()
    if pos_type not in ("Buy", "Sell"):
        pos_type = "Buy"

    close_price = deal.get("closePrice")
    profit = deal.get("profit", deal.get("grossProfit", 0))
    pips = deal.get("pips", 0)

    if close_price is None:
        event_type = "Create Position"
    elif profit < 0:
        event_type = "Stop Loss Hit"
    else:
        event_type = "Position closed"

    return {
        "serial": serial,
        "orderId": None,
        "positionId": deal.get("positionId", deal.get("position", 0)) or deal.get("id", deal.get("dealId", 0)),
        "event": event_type,
        "time": _coerce_int_time(deal.get("time", deal.get("closeTime", now_ms)), now_ms),
        "volume": deal.get("volume", deal.get("quantity", 0)),
        "quantity": deal.get("volume", deal.get("quantity", 0)),
        "type": pos_type,
        "entryPrice": deal.get("entryPrice", deal.get("openPrice", 0)),
        "tp": deal.get("takeProfit", deal.get("tp", None)),
        "sl": deal.get("stopLoss", deal.get("sl", None)),
        "closePrice": close_price,
        "grossProfit": profit,
        "pips": pips,
        "balance": deal.get("balance", None),
        "equity": deal.get("equity", None),
    }


def _build_history_event(h_item: dict[str, Any], serial: int, now_ms: int) -> Optional[dict[str, Any]]:
    """Build an event dict from an order history record."""
    pos_type = h_item.get("type", h_item.get("side", "Buy"))
    if isinstance(pos_type, str):
        pos_type = pos_type.capitalize()
    if pos_type not in ("Buy", "Sell"):
        pos_type = "Buy"

    close_price = h_item.get("closePrice")
    profit = h_item.get("profit", h_item.get("grossProfit", 0))
    pips = h_item.get("pips", 0)

    if close_price is None:
        event_type = "Create Position"
    elif profit < 0:
        event_type = "Stop Loss Hit"
    else:
        event_type = "Position closed"

    return {
        "serial": serial,
        "orderId": None,
        "positionId": h_item.get("positionId", h_item.get("position", 0)) or h_item.get("id", h_item.get("orderId", 0)),
        "event": event_type,
        "time": _coerce_int_time(h_item.get("time", h_item.get("closeTime", now_ms)), now_ms),
        "volume": h_item.get("volume", h_item.get("quantity", 0)),
        "quantity": h_item.get("volume", h_item.get("quantity", 0)),
        "type": pos_type,
        "entryPrice": h_item.get("entryPrice", h_item.get("openPrice", 0)),
        "tp": h_item.get("takeProfit", h_item.get("tp", None)),
        "sl": h_item.get("stopLoss", h_item.get("sl", None)),
        "closePrice": close_price,
        "grossProfit": profit,
        "pips": pips,
        "balance": h_item.get("balance", None),
        "equity": h_item.get("equity", None),
    }
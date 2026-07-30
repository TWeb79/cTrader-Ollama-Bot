"""Trade execution and MCP client wrapper for the cTrader trading system.

Provides the CTraderMCPClient wrapper and place_trade helper that the
main orchestrator uses to interact with the cTrader MCP server.

Author: Inventions4All - github:TWeb79
Version: 1.0.0  (deployment: 2026-07-30T17:53:53+02:00)
"""

import json
import logging
from contextlib import AsyncExitStack
from datetime import datetime
from typing import Any, Optional

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

log = logging.getLogger("ai_trader.executor")

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
            read, write = await self._stack.enter_async_context(
                streamable_http_client(self.config["mcp_url"]),
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
            tool_name: Logical tool name (mapped via TOOL_NAMES).
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
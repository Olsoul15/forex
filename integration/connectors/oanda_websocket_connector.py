"""
OandaWebSocketConnector for real-time price data streaming from OANDA API.

This module provides a WebSocket connector for streaming real-time price data
from the OANDA API, with support for:
- Multiple instruments
- Automatic reconnection with backoff
- Connection metrics tracking
- Integration with the DataRouter for processing
"""

import asyncio
import json
import logging
import time
import uuid
from enum import Enum
from typing import Dict, List, Optional, Any, Callable

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from forex_ai.core.data_router import DataRouter, DataType, Priority

# Configure logging
logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """WebSocket connection states."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


class OandaWebSocketConnector:
    """
    WebSocket connector for real-time price data streaming from OANDA API.

    This connector establishes a WebSocket connection to the OANDA API or a local proxy,
    subscribes to price updates for specified instruments, and routes the data to
    the DataRouter for processing.

    Features:
    - Real-time price data streaming
    - Automatic reconnection with exponential backoff
    - Connection metrics tracking
    - Support for multiple instruments
    """

    def __init__(self, config: Dict[str, Any], data_router: DataRouter):
        """
        Initialize the OANDA WebSocket connector.

        Args:
            config: Configuration dictionary containing connection settings
            data_router: DataRouter instance for routing received data
        """
        self.config = config
        self.data_router = data_router

        # Extract config with defaults
        self.use_proxy = config.get("use_proxy", True)
        self.proxy_url = config.get("proxy_url", "ws://localhost:8080/stream")
        self.oanda_url = config.get(
            "oanda_url", "wss://stream-fxtrade.oanda.com/v3/accounts/"
        )
        self.account_id = config.get("account_id", "")
        self.access_token = config.get("access_token", "")
        self.instruments = config.get("instruments", ["EUR_USD"])
        self.max_reconnect_attempts = config.get("max_reconnect_attempts", 5)
        self.reconnect_delay = config.get("initial_reconnect_delay", 1)

        # Connection state and metrics
        self.connection_state = ConnectionState.DISCONNECTED
        self.connection_metrics = {
            "total_messages": 0,
            "heartbeats": 0,
            "price_updates": 0,
            "errors": 0,
            "reconnections": 0,
            "last_message_time": 0,
            "connection_duration": 0,
            "start_time": 0,
        }

        # WebSocket connection
        self.websocket = None
        self.connection_task = None
        self.processing_task = None
        self.shutdown_event = asyncio.Event()

        # Message handlers
        self.handlers = {}

        # Register default handlers
        self._register_default_handlers()

    def _register_default_handlers(self):
        """Register default message handlers."""
        self.register_handler("PRICE", self._handle_price)
        self.register_handler("HEARTBEAT", self._handle_heartbeat)

    def register_handler(self, message_type: str, handler: Callable):
        """
        Register a handler for a specific message type.

        Args:
            message_type: Type of message to handle
            handler: Handler function
        """
        self.handlers[message_type] = handler

    async def connect(self) -> bool:
        """
        Connect to the OANDA WebSocket API.

        Returns:
            bool: True if connection was successful, False otherwise
        """
        if self.connection_state in [
            ConnectionState.CONNECTING,
            ConnectionState.CONNECTED,
        ]:
            logger.warning("Connection attempt while already connecting/connected")
            return True

        self.connection_state = ConnectionState.CONNECTING
        self.connection_metrics["start_time"] = time.time()

        try:
            # Build the connection URL
            if self.use_proxy:
                url = self.proxy_url
            else:
                # Format instruments for OANDA API
                instruments_param = "/".join(self.instruments)
                url = f"{self.oanda_url}{self.account_id}/price/stream?instruments={instruments_param}"

            # Set up headers for non-proxy connections
            headers = {}
            if not self.use_proxy and self.access_token:
                headers["Authorization"] = f"Bearer {self.access_token}"

            # Connect to WebSocket
            logger.info(f"Connecting to WebSocket at {url}")
            self.websocket = await websockets.connect(
                url, extra_headers=headers, ping_interval=20, ping_timeout=30
            )

            self.connection_state = ConnectionState.CONNECTED
            logger.info("Successfully connected to OANDA WebSocket")
            return True

        except (ConnectionError, WebSocketException) as e:
            self.connection_state = ConnectionState.FAILED
            self.connection_metrics["errors"] += 1
            logger.error(f"Failed to connect to WebSocket: {str(e)}")
            return False

    async def reconnect(self) -> bool:
        """
        Attempt to reconnect to the WebSocket with exponential backoff.

        Returns:
            bool: True if reconnection was successful, False otherwise
        """
        attempt = 0
        delay = self.reconnect_delay

        while attempt < self.max_reconnect_attempts:
            self.connection_state = ConnectionState.RECONNECTING
            self.connection_metrics["reconnections"] += 1

            logger.info(
                f"Reconnection attempt {attempt + 1}/{self.max_reconnect_attempts} (delay: {delay}s)"
            )
            await asyncio.sleep(delay)

            if await self.connect():
                return True

            # Exponential backoff with jitter
            delay = min(
                delay * 2 + (delay * 0.1 * (2 * asyncio.get_event_loop().time() % 1)),
                60,
            )
            attempt += 1

        self.connection_state = ConnectionState.FAILED
        logger.error(
            f"Failed to reconnect after {self.max_reconnect_attempts} attempts"
        )
        return False

    async def disconnect(self):
        """Disconnect from the WebSocket."""
        self.shutdown_event.set()

        if self.websocket and self.connection_state == ConnectionState.CONNECTED:
            logger.info("Disconnecting from WebSocket")
            await self.websocket.close()

        self.connection_state = ConnectionState.DISCONNECTED
        self.websocket = None

        # Calculate connection duration
        if self.connection_metrics["start_time"] > 0:
            self.connection_metrics["connection_duration"] = (
                time.time() - self.connection_metrics["start_time"]
            )

    async def _handle_price(self, message: Dict[str, Any]):
        """
        Handle price update messages.

        Args:
            message: Price update message
        """
        self.connection_metrics["price_updates"] += 1
        instrument = message.get("instrument", "")

        try:
            price_data = {
                "instrument": instrument,
                "time": message.get("time", ""),
                "bid": float(message.get("bids", [{}])[0].get("price", 0)),
                "ask": float(message.get("asks", [{}])[0].get("price", 0)),
                "spread": 0.0,  # Will be calculated below
                "tradeable": message.get("tradeable", False),
                "timestamp": time.time(),
            }

            # Calculate spread
            if price_data["bid"] > 0 and price_data["ask"] > 0:
                price_data["spread"] = price_data["ask"] - price_data["bid"]

            # Route the price tick data
            logger.debug(f"Routing price tick: {instrument}")
            await self.data_router.route(
                {
                    "type": DataType.PRICE_TICK.value,
                    "priority": Priority.HIGH.value,
                    "data": price_data,
                    "metadata": {"source": "oanda_websocket", "instrument": instrument},
                }
            )

        except Exception as e:
            logger.error(f"Error processing price update: {str(e)}")
            self.connection_metrics["errors"] += 1

    async def _handle_heartbeat(self, message: Dict[str, Any]):
        """
        Handle heartbeat messages.

        Args:
            message: Heartbeat message
        """
        self.connection_metrics["heartbeats"] += 1
        self.connection_metrics["last_message_time"] = time.time()

    async def process_message(self, message_str: str):
        """
        Process a raw WebSocket message.

        Args:
            message_str: Raw message string from WebSocket
        """
        try:
            message = json.loads(message_str)
            self.connection_metrics["total_messages"] += 1

            # Extract message type
            message_type = message.get("type", "")

            # Find and call the appropriate handler
            handler = self.handlers.get(message_type)
            if handler:
                await handler(message)
            else:
                logger.debug(f"No handler for message type: {message_type}")

        except json.JSONDecodeError:
            logger.error("Failed to parse WebSocket message as JSON")
            self.connection_metrics["errors"] += 1
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {str(e)}")
            self.connection_metrics["errors"] += 1

    async def start_processing(self):
        """Start the WebSocket connection and message processing."""
        if self.connection_task is not None:
            logger.warning("Processing already started")
            return

        self.shutdown_event.clear()
        self.connection_task = asyncio.create_task(self._connection_loop())

    async def stop_processing(self):
        """Stop the WebSocket connection and message processing."""
        if self.connection_task is None:
            logger.warning("Processing not started")
            return

        await self.disconnect()

        if self.connection_task:
            self.connection_task.cancel()
            try:
                await self.connection_task
            except asyncio.CancelledError:
                pass
            self.connection_task = None

    async def _connection_loop(self):
        """Main connection loop for WebSocket processing."""
        while not self.shutdown_event.is_set():
            try:
                # Ensure we're connected
                if self.connection_state != ConnectionState.CONNECTED:
                    if not await self.connect():
                        if not await self.reconnect():
                            logger.error(
                                "Failed to establish connection, stopping connection loop"
                            )
                            break

                # Process messages
                async for message in self.websocket:
                    if self.shutdown_event.is_set():
                        break

                    await self.process_message(message)
                    self.connection_metrics["last_message_time"] = time.time()

            except (ConnectionClosed, WebSocketException) as e:
                logger.error(f"WebSocket connection closed unexpectedly: {str(e)}")
                self.connection_state = ConnectionState.DISCONNECTED

                if not self.shutdown_event.is_set():
                    if not await self.reconnect():
                        break

            except Exception as e:
                logger.error(f"Unexpected error in connection loop: {str(e)}")
                self.connection_metrics["errors"] += 1

                if not self.shutdown_event.is_set():
                    await asyncio.sleep(
                        5
                    )  # Prevent rapid reconnection attempts on persistent errors

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get connection metrics.

        Returns:
            dict: Connection metrics
        """
        # Update connection duration if connected
        if (
            self.connection_state == ConnectionState.CONNECTED
            and self.connection_metrics["start_time"] > 0
        ):
            self.connection_metrics["connection_duration"] = (
                time.time() - self.connection_metrics["start_time"]
            )

        return {**self.connection_metrics, "state": self.connection_state.value}

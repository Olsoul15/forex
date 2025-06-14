"""
OANDA Proxy Server for WebSocket connections.

This module provides a proxy server that connects to OANDA's WebSocket API
and forwards the data to clients. It handles:
- Connection management
- Authentication
- Data forwarding
- Error handling and reconnection
"""

import asyncio
import json
import logging
import os
import time
from typing import Dict, List, Set, Any, Optional
from datetime import datetime

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException
from websockets.server import WebSocketServerProtocol

from forex_ai.utils.logging import setup_logging, get_logger

# Set up logging
setup_logging()
logger = get_logger(__name__)

# Configuration
OANDA_ACCOUNT_ID = os.environ.get("OANDA_ACCOUNT_ID", "")
OANDA_ACCESS_TOKEN = os.environ.get("OANDA_ACCESS_TOKEN", "")
OANDA_WS_URL = "wss://stream-fxtrade.oanda.com/v3/accounts/{}/pricing/stream"
DEFAULT_PORT = 8080
MAX_RECONNECT_ATTEMPTS = 5
INITIAL_RECONNECT_DELAY = 1


class OandaProxyServer:
    """Proxy server for OANDA WebSocket connections."""

    def __init__(self, port: int = DEFAULT_PORT):
        """Initialize the proxy server.

        Args:
            port: Port to listen on
        """
        self.port = port
        self.clients: Set[WebSocketServerProtocol] = set()
        self.oanda_ws: Optional[websockets.WebSocketClientProtocol] = None
        self.connection_task: Optional[asyncio.Task] = None
        self.shutdown_event = asyncio.Event()
        self.metrics = {
            "total_messages": 0,
            "client_connections": 0,
            "client_disconnections": 0,
            "oanda_connections": 0,
            "oanda_disconnections": 0,
            "errors": 0,
            "start_time": time.time(),
        }

    async def start(self):
        """Start the proxy server."""
        logger.info(f"Starting OANDA proxy server on port {self.port}")
        server = await websockets.serve(
            self.handle_client,
            "localhost",
            self.port,
            ping_interval=20,
            ping_timeout=30,
        )

        # Start OANDA connection
        self.connection_task = asyncio.create_task(self._maintain_oanda_connection())

        try:
            await server.wait_closed()
        except Exception as e:
            logger.error(f"Server error: {str(e)}")
        finally:
            await self.stop()

    async def stop(self):
        """Stop the proxy server."""
        logger.info("Stopping OANDA proxy server")
        self.shutdown_event.set()

        # Close OANDA connection
        if self.oanda_ws:
            await self.oanda_ws.close()
            self.oanda_ws = None

        # Cancel connection task
        if self.connection_task:
            self.connection_task.cancel()
            try:
                await self.connection_task
            except asyncio.CancelledError:
                pass
            self.connection_task = None

        # Close all client connections
        for client in self.clients.copy():
            try:
                await client.close()
            except Exception as e:
                logger.error(f"Error closing client connection: {str(e)}")

        self.clients.clear()

    async def handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """Handle a new client connection.

        Args:
            websocket: Client WebSocket connection
            path: Request path
        """
        logger.info(f"New client connection from {websocket.remote_address}")
        self.clients.add(websocket)
        self.metrics["client_connections"] += 1

        try:
            # Send initial connection success message
            await websocket.send(
                json.dumps(
                    {
                        "type": "CONNECTION_STATUS",
                        "status": "connected",
                        "timestamp": datetime.now().isoformat(),
                    }
                )
            )

            # Keep connection alive until client disconnects
            async for message in websocket:
                # Handle any client messages if needed
                logger.debug(f"Received client message: {message}")

        except ConnectionClosed:
            logger.info(f"Client disconnected: {websocket.remote_address}")
        except Exception as e:
            logger.error(f"Error handling client connection: {str(e)}")
            self.metrics["errors"] += 1
        finally:
            self.clients.remove(websocket)
            self.metrics["client_disconnections"] += 1

    async def _connect_to_oanda(self) -> bool:
        """Connect to OANDA WebSocket API.

        Returns:
            bool: True if connection successful, False otherwise
        """
        if not OANDA_ACCOUNT_ID or not OANDA_ACCESS_TOKEN:
            logger.error("OANDA credentials not configured")
            return False

        try:
            # Build connection URL
            url = OANDA_WS_URL.format(OANDA_ACCOUNT_ID)
            headers = {"Authorization": f"Bearer {OANDA_ACCESS_TOKEN}"}

            # Connect to OANDA
            logger.info("Connecting to OANDA WebSocket")
            self.oanda_ws = await websockets.connect(
                url, extra_headers=headers, ping_interval=20, ping_timeout=30
            )
            self.metrics["oanda_connections"] += 1
            logger.info("Successfully connected to OANDA WebSocket")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to OANDA WebSocket: {str(e)}")
            self.metrics["errors"] += 1
            return False

    async def _maintain_oanda_connection(self):
        """Maintain connection to OANDA WebSocket API."""
        while not self.shutdown_event.is_set():
            try:
                # Ensure we're connected
                if not self.oanda_ws:
                    if not await self._connect_to_oanda():
                        # Wait before retrying
                        await asyncio.sleep(INITIAL_RECONNECT_DELAY)
                        continue

                # Process messages from OANDA
                async for message in self.oanda_ws:
                    if self.shutdown_event.is_set():
                        break

                    # Forward message to all connected clients
                    self.metrics["total_messages"] += 1
                    await self._broadcast_message(message)

            except (ConnectionClosed, WebSocketException) as e:
                logger.error(f"OANDA WebSocket connection closed: {str(e)}")
                self.metrics["oanda_disconnections"] += 1
                self.oanda_ws = None

                if not self.shutdown_event.is_set():
                    # Attempt to reconnect
                    await asyncio.sleep(INITIAL_RECONNECT_DELAY)

            except Exception as e:
                logger.error(f"Error in OANDA connection loop: {str(e)}")
                self.metrics["errors"] += 1
                self.oanda_ws = None

                if not self.shutdown_event.is_set():
                    await asyncio.sleep(5)  # Prevent rapid reconnection attempts

    async def _broadcast_message(self, message: str):
        """Broadcast a message to all connected clients.

        Args:
            message: Message to broadcast
        """
        if not self.clients:
            return

        # Parse message to validate JSON
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            logger.error("Invalid JSON message from OANDA")
            self.metrics["errors"] += 1
            return

        # Forward to all connected clients
        disconnected_clients = set()
        for client in self.clients:
            try:
                await client.send(message)
            except ConnectionClosed:
                disconnected_clients.add(client)
            except Exception as e:
                logger.error(f"Error sending message to client: {str(e)}")
                self.metrics["errors"] += 1
                disconnected_clients.add(client)

        # Clean up disconnected clients
        for client in disconnected_clients:
            self.clients.remove(client)
            self.metrics["client_disconnections"] += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get server metrics.

        Returns:
            dict: Server metrics
        """
        uptime = time.time() - self.metrics["start_time"]
        return {
            **self.metrics,
            "uptime": uptime,
            "active_clients": len(self.clients),
            "oanda_connected": self.oanda_ws is not None,
        }


async def main():
    """Main entry point."""
    # Create and start the proxy server
    server = OandaProxyServer()
    try:
        await server.start()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
    finally:
        await server.stop()


if __name__ == "__main__":
    asyncio.run(main()) 
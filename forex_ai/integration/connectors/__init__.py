"""
Connectors for the AutoAgent integration with external data sources.

This package provides connectors for integrating with external data sources,
including real-time price feeds, economic data providers, and news services.
"""

from forex_ai.integration.connectors.oanda_websocket_connector import (
    OandaWebSocketConnector,
)

__all__ = [
    "OandaWebSocketConnector",
]

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to run the OANDA proxy server.

This script starts the OANDA proxy server that handles WebSocket connections
and forwards data from OANDA to clients.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from forex_ai.proxy.oanda_proxy_server import OandaProxyServer
from forex_ai.utils.logging import setup_logging, get_logger

# Set up logging
setup_logging()
logger = get_logger(__name__)


async def main():
    """Main entry point."""
    # Check for required environment variables
    if not os.environ.get("OANDA_ACCOUNT_ID"):
        logger.error("OANDA_ACCOUNT_ID environment variable not set")
        sys.exit(1)

    if not os.environ.get("OANDA_ACCESS_TOKEN"):
        logger.error("OANDA_ACCESS_TOKEN environment variable not set")
        sys.exit(1)

    # Create and start the proxy server
    server = OandaProxyServer()
    try:
        logger.info("Starting OANDA proxy server...")
        await server.start()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
    finally:
        await server.stop()


if __name__ == "__main__":
    asyncio.run(main()) 
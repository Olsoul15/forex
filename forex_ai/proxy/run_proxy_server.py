#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to run the OANDA proxy server.

This script is for DEVELOPMENT USE ONLY. It starts the OANDA proxy server
that handles WebSocket connections and forwards data from OANDA to clients.

In production, the proxy server should be started with user credentials
loaded from the database, not from environment variables.
"""

import asyncio
import os
import sys
import argparse
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from forex_ai.proxy.oanda_proxy_server import OandaProxyServer
from forex_ai.utils.logging import setup_logging, get_logger
from forex_ai.models.broker_models import OandaCredentials, BrokerType, OandaEnvironment

# Set up logging
setup_logging()
logger = get_logger(__name__)


async def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run OANDA proxy server (DEVELOPMENT USE ONLY)")
    parser.add_argument("--user-id", help="User ID to load credentials for")
    parser.add_argument("--access-token", help="OANDA API access token")
    parser.add_argument("--account-id", help="OANDA account ID")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument("--dev", action="store_true", help="Use environment variables (DEVELOPMENT ONLY)")
    args = parser.parse_args()
    
    # Check if we have credentials from command line
    if args.access_token and args.account_id:
        logger.info("Using credentials from command line")
        server = OandaProxyServer(
            port=args.port,
            access_token=args.access_token,
            account_id=args.account_id
        )
    elif args.user_id:
        logger.info(f"Loading credentials for user {args.user_id}")
        server = OandaProxyServer(
            port=args.port,
            user_id=args.user_id
        )
    elif args.dev:
        # Explicitly use environment variables for development
        logger.warning("DEVELOPMENT MODE: Using environment variables for credentials")
        
        # Check for required environment variables
        if not os.environ.get("OANDA_ACCOUNT_ID"):
            logger.error("OANDA_ACCOUNT_ID environment variable not set")
            sys.exit(1)

        access_token = os.environ.get("OANDA_ACCESS_TOKEN") or os.environ.get("OANDA_API_KEY")
        if not access_token:
            logger.error("Neither OANDA_ACCESS_TOKEN nor OANDA_API_KEY environment variable is set")
            sys.exit(1)
            
        # Create credentials from environment variables
        try:
            credentials = OandaCredentials.from_env()
            server = OandaProxyServer(
                port=args.port,
                access_token=credentials.access_token,
                account_id=credentials.default_account_id
            )
        except ValueError as e:
            logger.error(f"Error creating credentials: {str(e)}")
            sys.exit(1)
    else:
        logger.error("No credentials provided. Use --user-id, --access-token/--account-id, or --dev")
        sys.exit(1)

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
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple OANDA WebSocket connection test.

This script is for DEVELOPMENT USE ONLY. It provides a minimal test of the OANDA 
WebSocket connection, focusing only on establishing the connection and receiving price data.

Required environment variables:
- OANDA_ACCESS_TOKEN or OANDA_API_KEY
- OANDA_ACCOUNT_ID

WARNING: This script should never be used in production. In production,
credentials should always come from the database/API.
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from forex_ai.utils.logging import setup_logging, get_logger

# Set up logging
setup_logging()
logger = get_logger(__name__)

# OANDA WebSocket URL (practice environment)
OANDA_WS_URL = "wss://stream-fxpractice.oanda.com/v3/accounts/{}/pricing/stream"
INSTRUMENTS = ["EUR_USD", "GBP_USD", "USD_JPY"]
TEST_DURATION = 30  # seconds


def get_oanda_credentials():
    """
    Get OANDA credentials from environment variables.
    
    Returns:
        tuple: (access_token, account_id) or (None, None) if not found
        
    Raises:
        SystemExit: If required environment variables are not set
    """
    logger.warning("DEVELOPMENT MODE: Using environment variables for OANDA credentials")
    
    # Check for required environment variables
    access_token = os.environ.get("OANDA_ACCESS_TOKEN") or os.environ.get("OANDA_API_KEY")
    account_id = os.environ.get("OANDA_ACCOUNT_ID")
    
    if not access_token:
        logger.error("Neither OANDA_ACCESS_TOKEN nor OANDA_API_KEY environment variable is set")
        print("Error: OANDA_ACCESS_TOKEN or OANDA_API_KEY environment variable must be set")
        return None, None
        
    if not account_id:
        logger.error("OANDA_ACCOUNT_ID environment variable not set")
        print("Error: OANDA_ACCOUNT_ID environment variable must be set")
        return None, None
        
    return access_token, account_id


async def connect_to_oanda(access_token, account_id):
    """
    Connect to OANDA WebSocket and stream price data.
    
    Args:
        access_token: OANDA API access token
        account_id: OANDA account ID
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Build connection URL
        url = OANDA_WS_URL.format(account_id)
        logger.info(f"Constructed WebSocket URL: {url}")
        headers = {"Authorization": f"Bearer {access_token}"}

        logger.info(f"Connecting to OANDA WebSocket at {url}")
        logger.info(f"Monitoring instruments: {', '.join(INSTRUMENTS)}")

        # Create connection with proper headers
        async with websockets.connect(
            url,
            additional_headers={"Authorization": f"Bearer {access_token}"}
        ) as websocket:
            logger.info("Successfully connected to OANDA WebSocket")

            # Subscribe to instruments
            subscribe_msg = {
                "instruments": INSTRUMENTS
            }
            await websocket.send(json.dumps(subscribe_msg))
            logger.info("Sent subscription request")

            # Start time for test duration
            start_time = datetime.now()
            message_count = 0

            # Process messages
            while True:
                try:
                    # Check if we've exceeded test duration
                    if (datetime.now() - start_time).total_seconds() > TEST_DURATION:
                        logger.info(f"Test duration ({TEST_DURATION}s) completed")
                        break

                    # Receive and process message
                    message = await websocket.recv()
                    message_count += 1

                    # Parse message
                    data = json.loads(message)
                    message_type = data.get("type", "UNKNOWN")

                    if message_type == "PRICE":
                        instrument = data.get("instrument", "UNKNOWN")
                        bid = data.get("bids", [{}])[0].get("price", "N/A")
                        ask = data.get("asks", [{}])[0].get("price", "N/A")
                        time = data.get("time", "N/A")
                        
                        logger.info(f"Price update - {instrument}: Bid={bid}, Ask={ask}, Time={time}")
                    elif message_type == "HEARTBEAT":
                        if message_count % 10 == 0:  # Log every 10th heartbeat
                            logger.debug("Received heartbeat")
                    else:
                        logger.debug(f"Received message type: {message_type}")

                except ConnectionClosed:
                    logger.error("Connection to OANDA closed unexpectedly")
                    break
                except json.JSONDecodeError:
                    logger.error("Failed to parse message as JSON")
                    continue
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
                    continue

            # Print summary
            logger.info("\n===== Test Summary =====")
            logger.info(f"Total messages received: {message_count}")
            logger.info(f"Test duration: {TEST_DURATION} seconds")
            logger.info("=======================\n")

            return True

    except Exception as e:
        logger.error(f"Failed to connect to OANDA WebSocket: {str(e)}")
        return False


async def main():
    """Main entry point."""
    logger.info("Starting simple OANDA WebSocket test (DEVELOPMENT USE ONLY)")
    
    # Get OANDA credentials
    access_token, account_id = get_oanda_credentials()
    if not access_token or not account_id:
        sys.exit(1)
    
    logger.info(f"Using access token: {access_token[:5]}...{access_token[-5:]}")
    logger.info(f"Using account ID: {account_id}")
    
    try:
        success = await connect_to_oanda(access_token, account_id)
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 
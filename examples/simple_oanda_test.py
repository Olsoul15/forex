#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple OANDA WebSocket connection test.

This script provides a minimal test of the OANDA WebSocket connection,
focusing only on establishing the connection and receiving price data.
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

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
OANDA_ACCOUNT_ID = os.environ.get("OANDA_ACCOUNT_ID", "")
OANDA_ACCESS_TOKEN = "f1086a5e2c1718a39aa9c8dd0c38f5c9-0329c78b1d1cc274a149ff4151365df0"  # Using the key from basic_oanda_test.py
OANDA_WS_URL = "wss://stream-fxpractice.oanda.com/v3/accounts/{}/pricing/stream"  # Using practice environment
INSTRUMENTS = ["EUR_USD", "GBP_USD", "USD_JPY"]
TEST_DURATION = 30  # seconds


async def connect_to_oanda():
    """Connect to OANDA WebSocket and stream price data."""
    if not OANDA_ACCOUNT_ID or not OANDA_ACCESS_TOKEN:
        logger.error("OANDA credentials not configured")
        logger.error("Please set OANDA_ACCOUNT_ID and OANDA_ACCESS_TOKEN environment variables")
        return False

    try:
        # Build connection URL
        url = OANDA_WS_URL.format(OANDA_ACCOUNT_ID)
        logger.info(f"Constructed WebSocket URL: {url}")
        headers = {"Authorization": f"Bearer {OANDA_ACCESS_TOKEN}"}

        logger.info(f"Connecting to OANDA WebSocket at {url}")
        logger.info(f"Monitoring instruments: {', '.join(INSTRUMENTS)}")

        # Create connection with proper headers
        async with websockets.connect(
            url,
            additional_headers={"Authorization": f"Bearer {OANDA_ACCESS_TOKEN}"}
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
    logger.info("Starting simple OANDA WebSocket test")
    
    try:
        success = await connect_to_oanda()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 
"""
Test script for the OANDA WebSocket connector.

This script is for DEVELOPMENT USE ONLY. It tests the real-time price data streaming 
from OANDA via WebSocket, the processing of this data through the DataRouter, 
and the integration with technical analysis modules.

Required environment variables:
- OANDA_ACCESS_TOKEN or OANDA_API_KEY
- OANDA_ACCOUNT_ID

WARNING: This script should never be used in production. In production,
credentials should always come from the database/API.
"""

import asyncio
import logging
import os
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from forex_ai.utils.logging import setup_logging, get_logger
from forex_ai.core.data_router import DataRouter, DataType
from forex_ai.integration.connectors.oanda_websocket_connector import (
    OandaWebSocketConnector,
)
from forex_ai.agents.technical import TechnicalAnalysisAgent
from forex_ai.integration.patterns.enhanced_pattern_recognition import (
    EnhancedPatternRecognition,
)

# Set up logging
setup_logging()
logger = get_logger(__name__)

# Configuration
INSTRUMENTS = ["EUR_USD", "GBP_USD", "USD_JPY"]
USE_PROXY = False  # Direct connection to OANDA
PROXY_URL = "ws://localhost:8080/stream"
MAX_TEST_DURATION = 120  # seconds

# Price data storage (for technical analysis)
price_cache = {}


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


class OandaWebSocketTest:
    """Test harness for OANDA WebSocket integration."""

    def __init__(self, access_token, account_id):
        """
        Initialize the test harness.
        
        Args:
            access_token: OANDA API access token
            account_id: OANDA account ID
        """
        # Setup router
        self.data_router = DataRouter()

        # Setup websocket connector
        ws_config = {
            "use_proxy": USE_PROXY,
            "proxy_url": PROXY_URL,
            "instruments": INSTRUMENTS,
            "max_reconnect_attempts": 3,
            "account_id": account_id,
            "access_token": access_token,
        }
        self.oanda_connector = OandaWebSocketConnector(
            config=ws_config, data_router=self.data_router
        )

        # Setup technical analysis
        self.tech_agent = TechnicalAnalysisAgent()
        self.pattern_recognizer = EnhancedPatternRecognition()

        # Setup metrics
        self.message_count = 0
        self.processing_times = []
        self.last_analysis_time = {}

    async def test_websocket_connection(self):
        """Test WebSocket connection and data streaming."""
        logger.info("Starting OANDA WebSocket connection test")

        # Register handlers for different data types
        self.data_router.register_direct_handler(
            DataType.PRICE_TICK, self.handle_price_tick
        )

        # Connect to WebSocket
        success = await self.oanda_connector.connect()
        if not success:
            logger.error("Failed to connect to OANDA WebSocket")
            return False

        # Start processing messages
        await self.oanda_connector.start_processing()

        # Let it run for a while
        logger.info(f"Running test for {MAX_TEST_DURATION} seconds...")
        try:
            await asyncio.sleep(MAX_TEST_DURATION)
        except asyncio.CancelledError:
            pass

        # Print results
        self.print_test_results()

        # Stop everything
        await self.oanda_connector.stop()

        return True

    async def handle_price_tick(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle price tick data.

        Args:
            data: Price tick data

        Returns:
            Processing result
        """
        start_time = datetime.now()
        self.message_count += 1

        try:
            # Extract data
            instrument = data.get("instrument")
            if not instrument:
                return {"success": False, "message": "Missing instrument"}

            # Store in cache
            if instrument not in price_cache:
                price_cache[instrument] = []

            # Add to cache with timestamp
            tick_data = {
                "timestamp": data.get("timestamp", datetime.now().isoformat()),
                "bid": data.get("bid", 0),
                "ask": data.get("ask", 0),
                "mid": (data.get("bid", 0) + data.get("ask", 0)) / 2,
            }
            price_cache[instrument].append(tick_data)

            # Keep cache at reasonable size
            max_cache_size = 500
            if len(price_cache[instrument]) > max_cache_size:
                price_cache[instrument] = price_cache[instrument][-max_cache_size:]

            # Log every 10th message
            if self.message_count % 10 == 0:
                logger.info(f"Received {self.message_count} price ticks")

            # Trigger technical analysis occasionally (every 30 seconds)
            last_analysis = self.last_analysis_time.get(instrument)
            now = datetime.now()
            if not last_analysis or (now - last_analysis).total_seconds() > 30:
                self.last_analysis_time[instrument] = now
                asyncio.create_task(self.analyze_price_data(instrument))

            # Calculate processing time
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds() * 1000
            self.processing_times.append(processing_time)

            return {"success": True, "message": f"Processed {instrument} price tick"}

        except Exception as e:
            logger.error(f"Error processing price tick: {str(e)}")
            return {"success": False, "message": f"Error: {str(e)}"}

    async def analyze_price_data(self, instrument: str) -> None:
        """
        Perform technical analysis on cached price data.

        Args:
            instrument: Instrument to analyze
        """
        try:
            if instrument not in price_cache or len(price_cache[instrument]) < 50:
                logger.info(f"Not enough data for {instrument} analysis")
                return

            logger.info(f"Analyzing {instrument} price data...")

            # Convert to pandas DataFrame for analysis
            import pandas as pd

            df = pd.DataFrame(price_cache[instrument])

            # Convert to OHLC format (using bid for simplicity)
            # In a real implementation, would use proper OHLC data
            ohlc = df.copy()
            ohlc["open"] = df["mid"].iloc[0]
            ohlc["high"] = df["mid"].max()
            ohlc["low"] = df["mid"].min()
            ohlc["close"] = df["mid"].iloc[-1]

            # Detect patterns
            patterns = await self.pattern_recognizer.detect_patterns(
                ohlc, pair=instrument.replace("_", "/"), timeframe="realtime"
            )

            if patterns and patterns.get("patterns"):
                logger.info(
                    f"Detected {len(patterns['patterns'])} patterns for {instrument}"
                )
                logger.info(
                    f"Strongest pattern: {patterns.get('strongest_pattern', {}).get('pattern_type')}"
                )
                logger.info(
                    f"Overall bias: {patterns.get('overall_bias')} (strength: {patterns.get('bias_strength', 0):.2f})"
                )
            else:
                logger.info(f"No patterns detected for {instrument}")

        except Exception as e:
            logger.error(f"Error analyzing price data for {instrument}: {str(e)}")

    def print_test_results(self) -> None:
        """Print test results."""
        logger.info("\n===== OANDA WebSocket Test Results =====")
        logger.info(f"Total messages received: {self.message_count}")

        if self.processing_times:
            avg_processing_time = sum(self.processing_times) / len(
                self.processing_times
            )
            logger.info(f"Average processing time: {avg_processing_time:.2f} ms")
            logger.info(f"Min processing time: {min(self.processing_times):.2f} ms")
            logger.info(f"Max processing time: {max(self.processing_times):.2f} ms")

        logger.info(f"Connection metrics: {self.oanda_connector.get_status()}")
        logger.info(f"Routing metrics: {self.data_router.get_metrics()}")
        logger.info("==========================================\n")


async def main():
    """Main function."""
    logger.info("OANDA WebSocket Test (DEVELOPMENT USE ONLY)")
    logger.info("==========================================")
    
    # Get OANDA credentials
    access_token, account_id = get_oanda_credentials()
    if not access_token or not account_id:
        sys.exit(1)
    
    logger.info(f"Using access token: {access_token[:5]}...{access_token[-5:]}")
    logger.info(f"Using account ID: {account_id}")
    
    try:
        # Initialize and run the test
        test = OandaWebSocketTest(access_token, account_id)
        success = await test.test_websocket_connection()
        
        if not success:
            logger.error("Test failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

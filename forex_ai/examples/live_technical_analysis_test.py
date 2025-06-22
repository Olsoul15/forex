"""
Live technical analysis test with OANDA data and AutoAgent integration.

This script is for DEVELOPMENT USE ONLY. It demonstrates:
1. Connecting to OANDA's API to stream real-time price data
2. Processing the data through the DataRouter
3. Initializing the AutoAgent orchestrator for AI-powered analysis
4. Performing real-time technical analysis with pattern recognition
5. Displaying the analysis results and generating trading signals

Required environment variables:
- OANDA_ACCESS_TOKEN or OANDA_API_KEY
- OANDA_ACCOUNT_ID

WARNING: This script should never be used in production. In production,
credentials should always come from the database/API.
"""

import asyncio
import os
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
from forex_ai.utils.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

# Import core components
from forex_ai.core.data_router import DataRouter, DataType, Priority
from forex_ai.integration.connectors.oanda_websocket_connector import (
    OandaWebSocketConnector,
)
from forex_ai.integration import create_orchestrator
from forex_ai.agents.technical import TechnicalAnalysisAgent
from forex_ai.integration.patterns.enhanced_pattern_recognition import (
    EnhancedPatternRecognition,
)

# Configuration
INSTRUMENTS = ["EUR_USD", "GBP_USD", "USD_JPY"]
USE_PROXY = False  # Direct connection to OANDA
MAX_TEST_DURATION = 300  # 5 minutes

# Data storage for analysis
price_cache = {}
analysis_results = {}


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


class LiveTechnicalAnalysisTest:
    """Test harness for live technical analysis with AutoAgent integration."""

    def __init__(self, access_token, account_id):
        """
        Initialize the test harness.
        
        Args:
            access_token: OANDA API access token
            account_id: OANDA account ID
        """
        logger.info("Initializing live technical analysis test")

        # Setup router
        self.data_router = DataRouter()

        # Setup websocket connector
        ws_config = {
            "use_proxy": USE_PROXY,
            "instruments": INSTRUMENTS,
            "max_reconnect_attempts": 3,
            "account_id": account_id,
            "access_token": access_token,
        }
        self.oanda_connector = OandaWebSocketConnector(
            config=ws_config, data_router=self.data_router
        )

        # Setup technical analysis components
        self.tech_agent = TechnicalAnalysisAgent()
        self.pattern_recognizer = EnhancedPatternRecognition()

        # AutoAgent orchestrator will be initialized in run()
        self.orchestrator = None

        # Setup metrics
        self.message_count = 0
        self.analysis_count = 0
        self.processing_times = []
        self.last_analysis_time = {}

    async def run(self):
        """Run the live technical analysis test."""
        logger.info("Starting live technical analysis test")

        # Initialize AutoAgent orchestrator
        logger.info("Initializing AutoAgent orchestrator")
        orchestrator_config = {
            "memory_config": {"schema_prefix": "test_", "cache_size": 50},
            "model": "gpt-4",
            "temperature": 0.3,
            "confidence_threshold": 0.65,
        }
        self.orchestrator = await create_orchestrator(orchestrator_config)
        logger.info("AutoAgent orchestrator initialized")

        # Register handlers for different data types
        self.data_router.register_direct_handler(
            DataType.PRICE_TICK, self.handle_price_tick
        )

        # Connect to WebSocket
        logger.info("Connecting to OANDA WebSocket")
        success = await self.oanda_connector.connect()
        if not success:
            logger.error("Failed to connect to OANDA WebSocket")
            return False

        # Start processing messages
        await self.oanda_connector.start_processing()
        logger.info("WebSocket processing started")

        # Let it run for the configured duration
        logger.info(f"Running test for {MAX_TEST_DURATION} seconds...")
        try:
            await asyncio.sleep(MAX_TEST_DURATION)
        except asyncio.CancelledError:
            logger.info("Test was cancelled")

        # Print results
        self.print_test_results()

        # Stop everything
        logger.info("Stopping WebSocket connector")
        await self.oanda_connector.stop()

        logger.info("Stopping orchestrator")
        await self.orchestrator.stop()

        logger.info("Test completed")
        return True

    async def handle_price_tick(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle price tick data from OANDA.

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

            # Log periodically
            if self.message_count % 20 == 0:
                logger.info(f"Received {self.message_count} price ticks")

            # Trigger technical analysis occasionally (every 30 seconds)
            last_analysis = self.last_analysis_time.get(instrument)
            now = datetime.now()
            if not last_analysis or (now - last_analysis).total_seconds() > 30:
                self.last_analysis_time[instrument] = now
                asyncio.create_task(self.analyze_instrument(instrument))

            # Calculate processing time
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds() * 1000  # ms
            self.processing_times.append(processing_time)

            return {"success": True, "instrument": instrument}

        except Exception as e:
            logger.error(f"Error processing price tick: {str(e)}")
            return {"success": False, "error": str(e)}

    async def analyze_instrument(self, instrument: str):
        """
        Perform technical analysis for an instrument.

        Args:
            instrument: Instrument to analyze
        """
        logger.info(f"Analyzing {instrument}")
        self.analysis_count += 1

        try:
            # Need at least 20 data points for analysis
            if instrument not in price_cache or len(price_cache[instrument]) < 20:
                logger.info(f"Not enough data for {instrument} analysis")
                return

            # Extract data from cache
            price_data = price_cache[instrument]
            timestamps = [item["timestamp"] for item in price_data]
            prices = [item["mid"] for item in price_data]

            # Convert to OHLC (simple approximation for testing)
            timeframes = ["1h", "4h", "daily"]

            for timeframe in timeframes:
                # This is a simplified approach - normally you'd want proper OHLC aggregation
                tf_str = "hourly" if "h" in timeframe else timeframe

                # Prepare simulated OHLC data from the tick data
                # In a real implementation, you would use proper resampling, but this is simplified for testing
                ohlc_data = {
                    "dates": timestamps,
                    "opens": [prices[0]] + prices[:-1],
                    "highs": [
                        (
                            max(prices[i : i + 5])
                            if i + 5 < len(prices)
                            else max(prices[i:])
                        )
                        for i in range(0, len(prices), 5)
                    ],
                    "lows": [
                        (
                            min(prices[i : i + 5])
                            if i + 5 < len(prices)
                            else min(prices[i:])
                        )
                        for i in range(0, len(prices), 5)
                    ],
                    "closes": prices,
                    "volumes": [1000] * len(prices),  # Dummy volume data
                }

                # Technical analysis using the agent
                analysis = await self.tech_agent.analyze_data(
                    instrument=instrument, timeframe=tf_str, ohlc_data=ohlc_data
                )

                if analysis:
                    logger.info(f"Technical analysis for {instrument} ({tf_str}):")
                    logger.info(f"Direction: {analysis.get('direction', 'unknown')}")
                    logger.info(f"Confidence: {analysis.get('confidence', 0):.2f}")

                    # Process with AutoAgent for deep analysis
                    await self.process_with_autoagent(instrument, timeframe, analysis)

        except Exception as e:
            logger.error(f"Error analyzing {instrument}: {str(e)}")

    async def process_with_autoagent(
        self, instrument: str, timeframe: str, analysis: Dict[str, Any]
    ):
        """
        Process technical analysis with AutoAgent.

        Args:
            instrument: Instrument being analyzed
            timeframe: Timeframe of the analysis
            analysis: Technical analysis results
        """
        logger.info(f"Processing {instrument} ({timeframe}) with AutoAgent")

        if not self.orchestrator:
            logger.error("AutoAgent orchestrator not initialized")
            return

        try:
            # Change format of instrument for AutoAgent (EUR_USD -> EUR/USD)
            pair = instrument.replace("_", "/")

            # Process the analysis result
            result = await self.orchestrator.process_analysis_result(
                analysis_result=analysis,
                analysis_type="technical",
                pair=pair,
                timeframe=timeframe,
            )

            if result.get("success"):
                logger.info(f"Analysis processed successfully")

                # Store the result
                if instrument not in analysis_results:
                    analysis_results[instrument] = []
                analysis_results[instrument].append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "timeframe": timeframe,
                        "analysis": analysis,
                        "autoagent_result": result,
                    }
                )

                # Check for trading signals
                signals = result.get("signals", [])
                if signals:
                    logger.info(f"Generated {len(signals)} trading signals")
                    for i, signal in enumerate(signals):
                        logger.info(f"Signal {i+1}:")
                        logger.info(f"Type: {signal.get('signal_type')}")
                        logger.info(f"Direction: {signal.get('direction')}")
                        logger.info(f"Strength: {signal.get('strength')}")
                        logger.info(f"Confidence: {signal.get('confidence')}")
                else:
                    logger.info("No signals generated")
            else:
                logger.error(f"Processing failed: {result.get('message')}")

        except Exception as e:
            logger.error(f"Error processing with AutoAgent: {str(e)}")

    def print_test_results(self):
        """Print test results summary."""
        logger.info("\n===== Test Results =====")
        logger.info(f"Total messages processed: {self.message_count}")
        logger.info(f"Total analyses performed: {self.analysis_count}")

        # Calculate processing time statistics
        if self.processing_times:
            avg_time = sum(self.processing_times) / len(self.processing_times)
            max_time = max(self.processing_times)
            min_time = min(self.processing_times)
            logger.info(f"Average processing time: {avg_time:.2f} ms")
            logger.info(f"Max processing time: {max_time:.2f} ms")
            logger.info(f"Min processing time: {min_time:.2f} ms")

        # Print analysis results
        logger.info("\nAnalysis Results:")
        for instrument, results in analysis_results.items():
            logger.info(f"\n{instrument}:")
            for timeframe, result in results.items():
                logger.info(f"  {timeframe}:")
                for key, value in result.items():
                    if isinstance(value, dict):
                        logger.info(f"    {key}:")
                        for k, v in value.items():
                            logger.info(f"      {k}: {v}")
                    else:
                        logger.info(f"    {key}: {value}")

        logger.info("=======================\n")


async def main():
    """Main function."""
    logger.info("Live Technical Analysis Test (DEVELOPMENT USE ONLY)")
    logger.info("==================================================")
    
    # Get OANDA credentials
    access_token, account_id = get_oanda_credentials()
    if not access_token or not account_id:
        sys.exit(1)
    
    logger.info(f"Using access token: {access_token[:5]}...{access_token[-5:]}")
    logger.info(f"Using account ID: {account_id}")
    
    try:
        # Initialize and run the test
        test = LiveTechnicalAnalysisTest(access_token, account_id)
        success = await test.run()
        
        if not success:
            logger.error("Test failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

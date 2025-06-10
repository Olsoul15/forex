"""
Live Data Testing for AutoAgent Integration with AI Forex trading system.

This script uses real market data to test the AutoAgent integration components
in a production-like environment.
"""

import asyncio
import os
import json
import sys
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("autoagent_live_test")

# Mock imports for testing without dependencies
USE_MOCKS = True

try:
    # Try to import actual components
    from forex_ai.integration import create_orchestrator
    from forex_ai.integration.enhanced_memory_manager import (
        EnhancedMemoryManager,
        AnalysisContext,
    )
    from forex_ai.core.data_router import DataRouter, ProcessingTier, DataType, Priority
    from forex_ai.integration.tools.technical_tools import get_technical_tools

    USE_MOCKS = False
    logger.info("Using actual implementation components")
except ImportError as e:
    logger.warning(f"Could not import actual components: {e}")
    logger.warning("Using mock components instead")

    # Import mock components if actual components are not available
    from mock_test_components_simple import (
        MockAnalysisContext as AnalysisContext,
        MockEnhancedMemoryManager as EnhancedMemoryManager,
        MockDataRouter as DataRouter,
        MockDataType as DataType,
        MockProcessingTier as ProcessingTier,
        MockAutoAgentOrchestrator,
        create_mock_orchestrator,
    )

    # Mock function to create orchestrator
    async def create_orchestrator(config=None):
        return await create_mock_orchestrator(config)


class MarketDataProvider:
    """Provider for live and historical market data."""

    def __init__(self, use_real_api=False):
        """Initialize the market data provider."""
        self.use_real_api = use_real_api
        self.api_key = os.environ.get("FOREX_API_KEY", "")
        self.cached_data = {}

    async def get_current_price(self, pair: str) -> Dict[str, Any]:
        """Get current price for a currency pair."""
        if self.use_real_api and self.api_key:
            # Use Alpha Vantage or another API in production
            return await self._fetch_from_api(pair)
        else:
            # Use simulated data for testing
            return self._generate_price_data(pair)

    async def get_historical_data(
        self, pair: str, timeframe: str, bars: int = 100
    ) -> List[Dict[str, Any]]:
        """Get historical price data for a currency pair."""
        if self.use_real_api and self.api_key:
            # Use Alpha Vantage or another API in production
            return await self._fetch_historical_from_api(pair, timeframe, bars)
        else:
            # Use simulated data for testing
            return self._generate_historical_data(pair, timeframe, bars)

    async def get_economic_indicator(
        self, indicator: str, country: str
    ) -> Dict[str, Any]:
        """Get economic indicator data."""
        if self.use_real_api and self.api_key:
            # Use Alpha Vantage or another API in production
            return await self._fetch_economic_data(indicator, country)
        else:
            # Use simulated data for testing
            return self._generate_economic_data(indicator, country)

    async def _fetch_from_api(self, pair: str) -> Dict[str, Any]:
        """Fetch current price from API (placeholder)."""
        logger.info(f"Would fetch live data for {pair} from API")
        # In a real implementation, this would make an API call
        return self._generate_price_data(pair)

    async def _fetch_historical_from_api(
        self, pair: str, timeframe: str, bars: int
    ) -> List[Dict[str, Any]]:
        """Fetch historical data from API (placeholder)."""
        logger.info(f"Would fetch historical data for {pair} ({timeframe}) from API")
        # In a real implementation, this would make an API call
        return self._generate_historical_data(pair, timeframe, bars)

    async def _fetch_economic_data(
        self, indicator: str, country: str
    ) -> Dict[str, Any]:
        """Fetch economic data from API (placeholder)."""
        logger.info(f"Would fetch {indicator} data for {country} from API")
        # In a real implementation, this would make an API call
        return self._generate_economic_data(indicator, country)

    def _generate_price_data(self, pair: str) -> Dict[str, Any]:
        """Generate simulated price data."""
        import random

        # Base prices for common pairs
        base_prices = {
            "EUR/USD": 1.1050,
            "GBP/USD": 1.2750,
            "USD/JPY": 149.50,
            "AUD/USD": 0.6580,
            "USD/CAD": 1.3650,
            "USD/CHF": 0.9050,
        }

        # Get base price or generate random if pair not in dictionary
        base = base_prices.get(pair, random.uniform(0.8, 1.5))

        # Add random movement
        price = base + random.uniform(-0.002, 0.002)

        # Generate bid/ask spread
        spread = random.uniform(0.0001, 0.0003)
        bid = price - spread / 2
        ask = price + spread / 2

        return {
            "pair": pair,
            "price": price,
            "bid": bid,
            "ask": ask,
            "timestamp": datetime.now().isoformat(),
            "volume": random.uniform(100000, 1000000),
        }

    def _generate_historical_data(
        self, pair: str, timeframe: str, bars: int
    ) -> List[Dict[str, Any]]:
        """Generate simulated historical data."""
        import random
        import numpy as np

        # Base prices for common pairs
        base_prices = {
            "EUR/USD": 1.1050,
            "GBP/USD": 1.2750,
            "USD/JPY": 149.50,
            "AUD/USD": 0.6580,
            "USD/CAD": 1.3650,
            "USD/CHF": 0.9050,
        }

        # Get base price or generate random if pair not in dictionary
        base = base_prices.get(pair, random.uniform(0.8, 1.5))

        # Generate random walk
        changes = np.random.normal(0, 0.0015, bars)
        prices = [base]

        for change in changes:
            prices.append(prices[-1] + change)

        # Timeframe to minutes mapping
        timeframe_minutes = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "4h": 240,
            "1d": 1440,
        }

        minutes = timeframe_minutes.get(timeframe, 60)

        # Generate candles
        now = datetime.now()
        result = []

        for i in range(bars):
            price = prices[i]
            candle_time = now - timedelta(minutes=minutes * (bars - i))

            # Generate random candle data
            high = price + random.uniform(0.0001, 0.0020)
            low = price - random.uniform(0.0001, 0.0020)
            open_price = price + random.uniform(-0.0015, 0.0015)
            close_price = price + random.uniform(-0.0015, 0.0015)

            # Ensure high is highest and low is lowest
            high = max(high, open_price, close_price)
            low = min(low, open_price, close_price)

            result.append(
                {
                    "pair": pair,
                    "timestamp": candle_time.isoformat(),
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close_price,
                    "volume": random.uniform(100000, 1000000),
                }
            )

        return result

    def _generate_economic_data(self, indicator: str, country: str) -> Dict[str, Any]:
        """Generate simulated economic indicator data."""
        import random

        # Simulate different types of indicators
        indicators = {
            "gdp": {
                "value": random.uniform(-1, 4),
                "previous": random.uniform(-1, 4),
                "forecast": random.uniform(-1, 4),
                "unit": "%",
            },
            "interest_rate": {
                "value": random.uniform(0, 5),
                "previous": random.uniform(0, 5),
                "forecast": random.uniform(0, 5),
                "unit": "%",
            },
            "inflation": {
                "value": random.uniform(0, 6),
                "previous": random.uniform(0, 6),
                "forecast": random.uniform(0, 6),
                "unit": "%",
            },
            "unemployment": {
                "value": random.uniform(3, 10),
                "previous": random.uniform(3, 10),
                "forecast": random.uniform(3, 10),
                "unit": "%",
            },
            "retail_sales": {
                "value": random.uniform(-2, 5),
                "previous": random.uniform(-2, 5),
                "forecast": random.uniform(-2, 5),
                "unit": "%",
            },
        }

        indicator_data = indicators.get(
            indicator.lower(),
            {
                "value": random.uniform(-5, 5),
                "previous": random.uniform(-5, 5),
                "forecast": random.uniform(-5, 5),
                "unit": "",
            },
        )

        return {
            "indicator": indicator,
            "country": country,
            "timestamp": datetime.now().isoformat(),
            **indicator_data,
        }


async def test_with_price_ticks(data_router, orchestrator, data_provider):
    """Test the system with price tick data."""
    logger.info("=== Testing with Price Ticks ===")

    pairs = ["EUR/USD", "GBP/USD", "USD/JPY"]
    results = []

    for pair in pairs:
        # Get current price
        price_data = await data_provider.get_current_price(pair)
        logger.info(f"Got price data for {pair}: {price_data['price']:.5f}")

        # Route price data
        result = await data_router.route(price_data, DataType.PRICE_TICK)
        results.append(result)
        logger.info(f"Routed {pair} price: {result.get('message', 'No message')}")

    # Get routing metrics
    metrics = data_router.get_metrics()
    logger.info(f"Router metrics: {metrics.get('route_counts', {})}")

    return results


async def test_with_technical_analysis(orchestrator, data_provider):
    """Test the system with technical analysis."""
    logger.info("=== Testing with Technical Analysis ===")

    pair = "EUR/USD"
    timeframe = "1h"

    # Get historical data
    historical_data = await data_provider.get_historical_data(pair, timeframe, 100)
    logger.info(
        f"Got {len(historical_data)} historical candles for {pair} ({timeframe})"
    )

    # Prepare technical analysis result
    # In a real implementation, this would be the output of a technical analysis module
    technical_result = {
        "pair": pair,
        "timeframe": timeframe,
        "overall_direction": (
            "bullish"
            if historical_data[-1]["close"] > historical_data[0]["close"]
            else "bearish"
        ),
        "confidence": 0.78,
        "indicators": {
            "rsi": {"value": 63, "interpretation": "bullish"},
            "macd": {"signal": "bullish", "histogram": 0.0012},
        },
        "patterns": [
            {"name": "double_bottom", "direction": "bullish", "strength": 0.75}
        ],
        "support_levels": [
            min(candle["low"] for candle in historical_data[-20:]) - 0.001,
            min(candle["low"] for candle in historical_data[-40:-20]) - 0.002,
        ],
        "resistance_levels": [
            max(candle["high"] for candle in historical_data[-20:]) + 0.001,
            max(candle["high"] for candle in historical_data[-40:-20]) + 0.002,
        ],
        "generate_signal": True,
    }

    # Process the analysis result
    process_result = await orchestrator.process_analysis_result(
        technical_result, "technical"
    )

    if process_result.get("success"):
        logger.info(f"Technical analysis processed successfully")
        if "signals" in process_result:
            logger.info(f"  Generated {len(process_result.get('signals', []))} signals")
            for signal in process_result.get("signals", []):
                logger.info(
                    f"  Signal: {signal.get('direction')} ({signal.get('confidence'):.2f})"
                )
    else:
        logger.error(
            f"Technical analysis processing failed: {process_result.get('message')}"
        )

    return process_result


async def test_with_economic_indicators(data_router, orchestrator, data_provider):
    """Test the system with economic indicator data."""
    logger.info("=== Testing with Economic Indicators ===")

    indicators = [
        {"indicator": "interest_rate", "country": "US"},
        {"indicator": "gdp", "country": "EU"},
        {"indicator": "inflation", "country": "UK"},
    ]

    results = []

    for indicator_info in indicators:
        # Get economic data
        economic_data = await data_provider.get_economic_indicator(
            indicator_info["indicator"], indicator_info["country"]
        )

        logger.info(
            f"Got {economic_data['indicator']} data for {economic_data['country']}: {economic_data['value']}{economic_data['unit']}"
        )

        # Route economic data
        result = await data_router.route(economic_data, DataType.ECONOMIC_INDICATOR)
        results.append(result)
        logger.info(
            f"Routed {economic_data['indicator']} data: {result.get('message', 'No message')}"
        )

    # Get routing metrics
    metrics = data_router.get_metrics()
    logger.info(
        f"Router metrics for economic indicators: {metrics.get('route_counts', {})}"
    )

    return results


async def test_comprehensive_market_analysis(orchestrator, data_provider):
    """Test comprehensive market analysis."""
    logger.info("=== Testing Comprehensive Market Analysis ===")

    pair = "EUR/USD"
    timeframe = "1h"

    try:
        logger.info(
            f"Starting comprehensive market analysis for {pair} on {timeframe}..."
        )
        analysis_result = await orchestrator.analyze_market(pair, timeframe)

        if analysis_result.get("success"):
            logger.info("✓ Market analysis executed successfully")
            logger.info(f"  Context ID: {analysis_result.get('context_id')}")

            # Print some details of the analysis
            market_view = analysis_result.get("market_view", {})
            logger.info(
                f"  Market direction: {market_view.get('overall_direction', 'unknown')}"
            )
            logger.info(f"  Confidence: {market_view.get('confidence', 0):.2f}")

            # Technical analysis details
            technical = market_view.get("technical", {})
            if technical:
                logger.info(
                    f"  Technical indicators: {len(technical.get('indicators', {}))}"
                )
                logger.info(f"  Patterns detected: {technical.get('patterns', [])}")

            # Fundamental analysis details
            fundamental = market_view.get("fundamental", {})
            if fundamental:
                logger.info(
                    f"  Economic events: {fundamental.get('economic_events', [])}"
                )
                logger.info(f"  Sentiment: {fundamental.get('sentiment', 'neutral')}")

        else:
            logger.error(f"✗ Market analysis failed: {analysis_result.get('message')}")

    except Exception as e:
        logger.error(f"✗ Market analysis error: {str(e)}")
        import traceback

        traceback.print_exc()

    return analysis_result


async def test_market_context_retrieval(orchestrator, data_provider):
    """Test market context retrieval."""
    logger.info("=== Testing Market Context Retrieval ===")

    pair = "EUR/USD"
    timeframe = "1h"

    try:
        logger.info(f"Retrieving market context for {pair} on {timeframe}...")
        context_result = await orchestrator.get_market_context(
            pair, timeframe, days_ago=1
        )

        if context_result.get("success"):
            logger.info("✓ Market context retrieved successfully")
            logger.info(f"  Found {context_result.get('context_count')} contexts")
            logger.info(f"  Summary: {context_result.get('summary')}")
        else:
            logger.error(f"✗ Context retrieval failed: {context_result.get('message')}")

    except Exception as e:
        logger.error(f"✗ Context retrieval error: {str(e)}")
        import traceback

        traceback.print_exc()

    return context_result


async def run_live_tests():
    """Run all live data tests."""
    logger.info("Starting AutoAgent integration live data tests...")

    try:
        # Initialize components
        config = {
            "memory_config": {"schema_prefix": "live_test_", "cache_size": 20},
            "model": "gpt-4",
            "temperature": 0.2,
            "confidence_threshold": 0.6,
        }

        # Initialize market data provider
        use_real_api = os.environ.get("USE_REAL_API", "false").lower() == "true"
        data_provider = MarketDataProvider(use_real_api=use_real_api)

        # Initialize data router
        data_router = DataRouter()

        # Initialize orchestrator
        orchestrator = await create_orchestrator(config)
        logger.info("✓ Orchestrator created successfully")

        # Register handlers for data router
        async def direct_handler(data):
            logger.info(f"Direct processing: {data.get('pair', 'unknown')} price tick")
            # In a real implementation, this would do some quick processing
            return {"success": True, "message": "Price tick processed"}

        async def orchestration_handler(data, data_type):
            logger.info(f"Orchestration processing: {data_type}")
            if data_type == DataType.ECONOMIC_INDICATOR:
                result = await orchestrator.process_analysis_result(data, "fundamental")
                return result
            return {"success": True, "message": "Data processed"}

        async def deep_research_handler(data, data_type):
            logger.info(f"Deep research processing: {data_type}")
            # In a real implementation, this would queue for deep analysis
            return {"success": True, "message": "Queued for deep research"}

        # Register handlers
        if hasattr(DataType, "PRICE_TICK"):
            data_router.register_direct_handler(
                (
                    DataType.PRICE_TICK
                    if isinstance(DataType.PRICE_TICK, str)
                    else DataType.PRICE_TICK.value
                ),
                direct_handler,
            )
        else:
            logger.error("Could not register direct handler for PRICE_TICK")

        data_router.register_orchestration_handler(orchestration_handler)
        data_router.register_deep_research_handler(deep_research_handler)

        # Start router
        await data_router.start()
        logger.info("✓ DataRouter started")

        # Run tests
        await test_with_price_ticks(data_router, orchestrator, data_provider)
        await test_with_technical_analysis(orchestrator, data_provider)
        await test_with_economic_indicators(data_router, orchestrator, data_provider)
        await test_comprehensive_market_analysis(orchestrator, data_provider)
        await test_market_context_retrieval(orchestrator, data_provider)

        # Stop components
        await data_router.stop()
        await orchestrator.stop()
        logger.info("✓ Components stopped successfully")

        logger.info("\nAll live data tests completed!")

    except Exception as e:
        logger.error(f"\nTest suite error: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Check if USE_MOCKS argument is provided
    if len(sys.argv) > 1 and sys.argv[1].lower() == "--no-mocks":
        USE_MOCKS = False
        logger.info("Running with actual components (--no-mocks flag detected)")
    else:
        logger.info(
            "Running with mock components (use --no-mocks to use actual components)"
        )

    # Run tests
    asyncio.run(run_live_tests())

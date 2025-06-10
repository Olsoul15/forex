"""
Live Data Testing for AutoAgent Integration with AI Forex trading system.

This script includes all necessary mock components and tests the AutoAgent
integration with simulated live data.
"""

import asyncio
import os
import json
import sys
import random
import numpy as np
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


#################################################
# MOCK COMPONENTS
#################################################


class AnalysisContext:
    """Mock of the AnalysisContext class."""

    def __init__(
        self,
        context_id: Optional[str] = None,
        pair: Optional[str] = None,
        timeframe: Optional[str] = None,
        analysis_type: Optional[str] = None,
        findings: Optional[Dict[str, Any]] = None,
        confidence: float = 0.0,
        timestamp: Optional[datetime] = None,
        related_contexts: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the context with the provided parameters."""
        self.context_id = context_id or f"ctx-{datetime.now().timestamp()}"
        self.pair = pair
        self.timeframe = timeframe
        self.analysis_type = analysis_type
        self.findings = findings or {}
        self.confidence = confidence
        self.timestamp = timestamp or datetime.now()
        self.related_contexts = related_contexts or []
        self.tags = tags or []
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for storage."""
        return {
            "context_id": self.context_id,
            "pair": self.pair,
            "timeframe": self.timeframe,
            "analysis_type": self.analysis_type,
            "findings": json.dumps(self.findings),
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "related_contexts": json.dumps(self.related_contexts),
            "tags": json.dumps(self.tags),
            "metadata": json.dumps(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisContext":
        """Create context from dictionary."""
        return cls(
            context_id=data.get("context_id"),
            pair=data.get("pair"),
            timeframe=data.get("timeframe"),
            analysis_type=data.get("analysis_type"),
            findings=json.loads(data.get("findings")) if data.get("findings") else {},
            confidence=data.get("confidence", 0.0),
            timestamp=(
                datetime.fromisoformat(data.get("timestamp"))
                if data.get("timestamp")
                else datetime.now()
            ),
            related_contexts=(
                json.loads(data.get("related_contexts"))
                if data.get("related_contexts")
                else []
            ),
            tags=json.loads(data.get("tags")) if data.get("tags") else [],
            metadata=json.loads(data.get("metadata")) if data.get("metadata") else {},
        )


class EnhancedMemoryManager:
    """Mock of the EnhancedMemoryManager class."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the memory manager with the provided configuration."""
        self.config = config or {}
        self.schema_prefix = self.config.get("schema_prefix", "mock_")
        self.cache_size = self.config.get("cache_size", 100)
        self.memory_cache = {}
        self.table_names = self._initialize_schema()

    def _initialize_schema(self) -> Dict[str, str]:
        """Initialize schema with table names."""
        return {
            "analysis_contexts": f"{self.schema_prefix}analysis_contexts",
            "context_embeddings": f"{self.schema_prefix}context_embeddings",
            "context_relationships": f"{self.schema_prefix}context_relationships",
        }

    async def ensure_tables_exist(self) -> None:
        """Ensure all required tables exist in the database."""
        logger.info(f"Would create tables: {list(self.table_names.values())}")

    async def store_analysis_result(
        self, analysis_result: Dict[str, Any], analysis_type: str
    ) -> str:
        """Store an analysis result in memory."""
        context_id = f"ctx-{datetime.now().timestamp()}"
        context = AnalysisContext(
            context_id=context_id,
            pair=analysis_result.get("pair"),
            timeframe=analysis_result.get("timeframe"),
            analysis_type=analysis_type,
            findings=analysis_result,
            confidence=analysis_result.get("confidence", 0.5),
            tags=[analysis_type, analysis_result.get("overall_direction", "neutral")],
        )

        self.memory_cache[context_id] = context.to_dict()
        return context_id

    async def retrieve_context_by_id(self, context_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific context by ID."""
        return self.memory_cache.get(context_id)

    async def retrieve_context(
        self,
        pair: Optional[str] = None,
        timeframe: Optional[str] = None,
        analysis_type: Optional[str] = None,
        days_ago: int = 30,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Retrieve contexts matching filters."""
        results = []
        for ctx_id, ctx_data in self.memory_cache.items():
            matches_pair = pair is None or ctx_data.get("pair") == pair
            matches_timeframe = (
                timeframe is None or ctx_data.get("timeframe") == timeframe
            )
            matches_type = (
                analysis_type is None or ctx_data.get("analysis_type") == analysis_type
            )

            if matches_pair and matches_timeframe and matches_type:
                results.append(ctx_data)

            if len(results) >= limit:
                break

        return results

    async def summarize_contexts(self, contexts: List[Dict[str, Any]]) -> str:
        """Generate a summary of multiple contexts."""
        if not contexts:
            return "No contexts to summarize."

        pairs = set(ctx.get("pair") for ctx in contexts if ctx.get("pair"))
        timeframes = set(
            ctx.get("timeframe") for ctx in contexts if ctx.get("timeframe")
        )
        types = set(
            ctx.get("analysis_type") for ctx in contexts if ctx.get("analysis_type")
        )

        return f"Summary of {len(contexts)} contexts for {', '.join(pairs)} on {', '.join(timeframes)} timeframes. Analysis types: {', '.join(types)}."


class DataType:
    """Mock enum for data types."""

    PRICE_TICK = "PRICE_TICK"
    ECONOMIC_INDICATOR = "ECONOMIC_INDICATOR"
    PATTERN_CONFIRMATION = "PATTERN_CONFIRMATION"
    UNKNOWN = "UNKNOWN"


class ProcessingTier:
    """Mock enum for processing tiers."""

    DIRECT = "direct"
    ORCHESTRATION = "orchestration"
    DEEP_RESEARCH = "deep_research"


class Priority:
    """Mock enum for priority levels."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RoutingMetrics:
    """Class for tracking routing metrics."""

    def __init__(self):
        """Initialize routing metrics."""
        self.route_counts = {}
        self.latencies = {}
        self.errors = {}

    def record_routing(self, data_type, tier):
        """Record a routing decision."""
        if data_type not in self.route_counts:
            self.route_counts[data_type] = {}

        if tier not in self.route_counts[data_type]:
            self.route_counts[data_type][tier] = 0

        self.route_counts[data_type][tier] += 1

    def record_latency(self, data_type, latency_ms):
        """Record processing latency."""
        if data_type not in self.latencies:
            self.latencies[data_type] = []

        self.latencies[data_type].append(latency_ms)

    def record_error(self, data_type, error):
        """Record an error."""
        if data_type not in self.errors:
            self.errors[data_type] = []

        self.errors[data_type].append(
            {"timestamp": datetime.now().isoformat(), "error": str(error)}
        )

    def get_summary(self):
        """Get a summary of metrics."""
        return {
            "route_counts": self.route_counts,
            "latencies": {
                k: {
                    "avg": sum(v) / len(v) if v else 0,
                    "min": min(v) if v else 0,
                    "max": max(v) if v else 0,
                    "count": len(v),
                }
                for k, v in self.latencies.items()
            },
            "errors": {k: len(v) for k, v in self.errors.items()},
        }


class DataRouter:
    """Mock of the DataRouter class."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the data router with the provided configuration."""
        self.config = config or {}
        self.direct_handlers = {}
        self.orchestration_handler = None
        self.deep_research_handler = None
        self.metrics = RoutingMetrics()
        self.running = False

    def register_direct_handler(self, data_type: str, handler) -> None:
        """Register a handler for direct processing."""
        self.direct_handlers[data_type] = handler

    def register_orchestration_handler(self, handler) -> None:
        """Register a handler for orchestration processing."""
        self.orchestration_handler = handler

    def register_deep_research_handler(self, handler) -> None:
        """Register a handler for deep research processing."""
        self.deep_research_handler = handler

    async def start(self) -> None:
        """Start the router background workers."""
        self.running = True
        logger.info("Router started")

    async def stop(self) -> None:
        """Stop the router background workers."""
        self.running = False
        logger.info("Router stopped")

    async def route(
        self, data: Dict[str, Any], data_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Route data to appropriate processing tier."""
        data_type = data_type or self._infer_data_type(data)

        # Determine tier based on data type
        if data_type == DataType.PRICE_TICK:
            tier = ProcessingTier.DIRECT
        elif data_type == DataType.ECONOMIC_INDICATOR:
            tier = ProcessingTier.ORCHESTRATION
        else:
            tier = ProcessingTier.DEEP_RESEARCH

        # Record routing
        self.metrics.record_routing(data_type, tier)

        # Simulate processing time
        start_time = datetime.now()

        try:
            # Route based on tier
            if tier == ProcessingTier.DIRECT:
                result = await self._route_to_direct(data, data_type)
            elif tier == ProcessingTier.ORCHESTRATION:
                result = await self._route_to_orchestration(data, data_type)
            else:  # DEEP_RESEARCH
                result = await self._route_to_deep_research(data, data_type)

            # Record latency
            end_time = datetime.now()
            latency_ms = (end_time - start_time).total_seconds() * 1000
            self.metrics.record_latency(data_type, latency_ms)

            return result

        except Exception as e:
            # Record error
            self.metrics.record_error(data_type, e)
            logger.error(f"Error routing {data_type}: {str(e)}")

            return {"success": False, "message": f"Error routing data: {str(e)}"}

    async def _route_to_direct(
        self, data: Dict[str, Any], data_type: str
    ) -> Dict[str, Any]:
        """Route to direct processing tier."""
        handler = self.direct_handlers.get(data_type)
        if handler:
            return await handler(data)

        return {
            "success": False,
            "message": f"No direct handler registered for {data_type}",
        }

    async def _route_to_orchestration(
        self, data: Dict[str, Any], data_type: str
    ) -> Dict[str, Any]:
        """Route to orchestration processing tier."""
        if self.orchestration_handler:
            return await self.orchestration_handler(data, data_type)

        return {"success": False, "message": "No orchestration handler registered"}

    async def _route_to_deep_research(
        self, data: Dict[str, Any], data_type: str
    ) -> Dict[str, Any]:
        """Route to deep research processing tier."""
        if self.deep_research_handler:
            return await self.deep_research_handler(data, data_type)

        return {"success": False, "message": "No deep research handler registered"}

    def _infer_data_type(self, data: Dict[str, Any]) -> str:
        """Infer data type from data characteristics."""
        if "price" in data:
            return DataType.PRICE_TICK
        if "indicator" in data:
            return DataType.ECONOMIC_INDICATOR
        if "pattern" in data:
            return DataType.PATTERN_CONFIRMATION
        return DataType.UNKNOWN

    def get_metrics(self) -> Dict[str, Any]:
        """Get routing metrics."""
        return self.metrics.get_summary()


class AutoAgentOrchestrator:
    """Mock of the AutoAgentOrchestrator class."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the orchestrator with the provided configuration."""
        self.config = config or {}
        self.memory = EnhancedMemoryManager(self.config.get("memory_config"))
        self.running = False

    async def initialize(self) -> None:
        """Initialize the orchestrator."""
        await self.memory.ensure_tables_exist()
        logger.info("Orchestrator initialized")

    async def process_analysis_result(
        self, analysis_result: Dict[str, Any], analysis_type: str
    ) -> Dict[str, Any]:
        """Process an analysis result."""
        context_id = await self.memory.store_analysis_result(
            analysis_result, analysis_type
        )

        # Check if signal generation is needed
        generate_signal = analysis_result.get("generate_signal", False)
        confidence = analysis_result.get("confidence", 0.0)
        confidence_threshold = self.config.get("confidence_threshold", 0.7)

        if generate_signal or confidence >= confidence_threshold:
            return {
                "success": True,
                "message": "Signal generation completed",
                "context_id": context_id,
                "signals": [
                    {
                        "signal_type": "entry",
                        "direction": analysis_result.get(
                            "overall_direction", "neutral"
                        ),
                        "strength": "medium",
                        "confidence": confidence,
                    }
                ],
            }

        return {
            "success": True,
            "message": "Analysis stored without signal generation",
            "context_id": context_id,
        }

    async def analyze_market(self, pair: str, timeframe: str) -> Dict[str, Any]:
        """Perform comprehensive market analysis."""
        # Simulate a market analysis result
        analysis_result = {
            "pair": pair,
            "timeframe": timeframe,
            "overall_direction": "bullish",
            "confidence": 0.75,
            "technical": {
                "indicators": {"rsi": 65, "macd": "bullish"},
                "patterns": ["double_bottom"],
            },
            "fundamental": {
                "economic_events": ["positive_gdp", "rate_hold"],
                "sentiment": "positive",
            },
        }

        context_id = await self.memory.store_analysis_result(
            analysis_result, "comprehensive"
        )

        return {
            "success": True,
            "message": "Market analysis completed",
            "context_id": context_id,
            "market_view": analysis_result,
        }

    async def get_market_context(
        self, pair: str, timeframe: str, days_ago: int = 30
    ) -> Dict[str, Any]:
        """Get market context."""
        contexts = await self.memory.retrieve_context(
            pair=pair, timeframe=timeframe, days_ago=days_ago
        )
        summary = await self.memory.summarize_contexts(contexts)

        return {
            "success": True,
            "message": "Market context retrieved",
            "context_count": len(contexts),
            "summary": summary,
            "contexts": contexts,
        }

    async def start(self) -> None:
        """Start the orchestrator."""
        if not self.running:
            await self.initialize()
            self.running = True
            logger.info("Orchestrator started")

    async def stop(self) -> None:
        """Stop the orchestrator."""
        if self.running:
            self.running = False
            logger.info("Orchestrator stopped")


async def create_orchestrator(config=None):
    """Create and initialize a mock orchestrator."""
    orchestrator = AutoAgentOrchestrator(config)
    await orchestrator.start()
    return orchestrator


#################################################
# MARKET DATA PROVIDER
#################################################


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


#################################################
# TEST FUNCTIONS
#################################################


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
        data_router.register_direct_handler(DataType.PRICE_TICK, direct_handler)
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
    # Run tests
    asyncio.run(run_live_tests())

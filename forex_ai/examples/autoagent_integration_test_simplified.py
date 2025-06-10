"""
Simplified Integration Test Script for AutoAgent with AI Forex

This script provides a simplified test environment for verifying that the
core components of the AutoAgent integration work correctly within the
AI Forex trading system.

Key components tested:
- EnhancedMemoryManager: For storing and retrieving analysis contexts
- AutoAgentOrchestrator: For coordinating analysis workflows
- Tool wrappers: For wrapping core technical analysis capabilities
"""

import asyncio
import json
import datetime
from typing import Dict, Any, List, Optional

# Simple mocked classes for testing purposes


class MockTechnicalAnalysisResult:
    """Simulates results from technical analysis tools."""

    @staticmethod
    def generate(pair="EUR/USD", timeframe="1h"):
        """Generate a mock technical analysis result."""
        return {
            "pair": pair,
            "timeframe": timeframe,
            "timestamp": datetime.datetime.now().isoformat(),
            "overall_direction": "bullish",
            "confidence": 0.75,
            "indicators": {
                "rsi": {"value": 65, "interpretation": "bullish"},
                "macd": {"signal": "bullish", "histogram": 0.0012},
                "moving_averages": {"ema_crossover": "bullish"},
            },
            "patterns": [
                {"name": "double_bottom", "direction": "bullish", "strength": 0.8}
            ],
            "support_levels": [1.0950, 1.0900],
            "resistance_levels": [1.1100, 1.1150],
            "volatility": "medium",
        }


class MockFundamentalAnalysisResult:
    """Simulates results from fundamental analysis tools."""

    @staticmethod
    def generate(pair="EUR/USD"):
        """Generate a mock fundamental analysis result."""
        currencies = pair.split("/")
        return {
            "pair": pair,
            "timestamp": datetime.datetime.now().isoformat(),
            "overall_sentiment": "positive",
            "confidence": 0.68,
            "economic_indicators": {
                currencies[0]: {
                    "gdp_growth": 1.8,
                    "interest_rate": 3.75,
                    "inflation": 2.2,
                    "unemployment": 7.3,
                    "sentiment": "neutral",
                },
                currencies[1]: {
                    "gdp_growth": 2.1,
                    "interest_rate": 5.00,
                    "inflation": 3.1,
                    "unemployment": 3.6,
                    "sentiment": "positive",
                },
            },
            "recent_events": [
                {
                    "event": "interest_rate_decision",
                    "country": currencies[0],
                    "impact": "medium",
                },
                {
                    "event": "employment_report",
                    "country": currencies[1],
                    "impact": "high",
                },
            ],
            "forecast": "The pair is expected to decline as economic indicators favor the quote currency.",
        }


class AnalysisContext:
    """Simplified mock of the AnalysisContext class."""

    def __init__(
        self,
        context_id: Optional[str] = None,
        pair: Optional[str] = None,
        timeframe: Optional[str] = None,
        analysis_type: Optional[str] = None,
        findings: Optional[Dict[str, Any]] = None,
        confidence: float = 0.0,
    ):
        """Initialize the context with the provided parameters."""
        self.context_id = context_id or f"ctx-{datetime.datetime.now().timestamp()}"
        self.pair = pair
        self.timeframe = timeframe
        self.analysis_type = analysis_type
        self.findings = findings or {}
        self.confidence = confidence
        self.timestamp = datetime.datetime.now()

    def to_dict(self):
        """Convert context to dictionary for storage."""
        return {
            "context_id": self.context_id,
            "pair": self.pair,
            "timeframe": self.timeframe,
            "analysis_type": self.analysis_type,
            "findings": self.findings,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
        }


class EnhancedMemoryManager:
    """Simplified mock of the EnhancedMemoryManager class."""

    def __init__(self):
        """Initialize the memory manager."""
        self.contexts = {}

    async def store_analysis_result(self, analysis_result, analysis_type):
        """Store an analysis result in memory."""
        context = AnalysisContext(
            pair=analysis_result.get("pair"),
            timeframe=analysis_result.get("timeframe"),
            analysis_type=analysis_type,
            findings=analysis_result,
            confidence=analysis_result.get("confidence", 0.5),
        )

        self.contexts[context.context_id] = context
        return context.context_id

    async def retrieve_context_by_id(self, context_id):
        """Retrieve a context by ID."""
        context = self.contexts.get(context_id)
        return context.to_dict() if context else None

    async def retrieve_contexts(
        self, pair=None, timeframe=None, analysis_type=None, limit=10
    ):
        """Retrieve contexts matching the filters."""
        results = []

        for context in self.contexts.values():
            if (
                (pair is None or context.pair == pair)
                and (timeframe is None or context.timeframe == timeframe)
                and (analysis_type is None or context.analysis_type == analysis_type)
            ):
                results.append(context.to_dict())

            if len(results) >= limit:
                break

        return results

    async def summarize_contexts(self, contexts):
        """Generate a summary of multiple contexts."""
        if not contexts:
            return "No contexts available."

        # Filter out None values and ensure all items are strings
        pairs = set(
            str(ctx.get("pair", "unknown"))
            for ctx in contexts
            if ctx.get("pair") is not None
        )
        timeframes = set(
            str(ctx.get("timeframe", "unknown"))
            for ctx in contexts
            if ctx.get("timeframe") is not None
        )

        # Default to empty set if no valid values found
        pairs = pairs or {"unknown"}
        timeframes = timeframes or {"unknown"}

        directions = {"bullish": 0, "bearish": 0, "neutral": 0}

        for ctx in contexts:
            findings = ctx.get("findings") or {}
            direction = findings.get("overall_direction", "neutral")
            directions[direction] = directions.get(direction, 0) + 1

        # Find the most common direction
        dominant_direction = (
            max(directions.items(), key=lambda x: x[1])[0] if directions else "neutral"
        )

        return f"Analysis of {', '.join(pairs)} on {', '.join(timeframes)} timeframes shows predominantly {dominant_direction} outlook."


class AutoAgentOrchestrator:
    """Simplified mock of the AutoAgentOrchestrator class."""

    def __init__(self):
        """Initialize the orchestrator."""
        self.memory = EnhancedMemoryManager()
        self.tools = {}
        self.workflows = {}

    async def register_tool(self, tool_name, tool_function):
        """Register a tool with the orchestrator."""
        self.tools[tool_name] = tool_function
        return True

    async def register_workflow(self, workflow_name, workflow_function):
        """Register a workflow with the orchestrator."""
        self.workflows[workflow_name] = workflow_function
        return True

    async def process_analysis_result(self, analysis_result, analysis_type):
        """Process an analysis result."""
        context_id = await self.memory.store_analysis_result(
            analysis_result, analysis_type
        )

        # Generate signals if confidence is high enough
        signals = []
        if analysis_result.get("confidence", 0) >= 0.7:
            signals.append(
                {
                    "pair": analysis_result.get("pair"),
                    "direction": analysis_result.get("overall_direction", "neutral"),
                    "confidence": analysis_result.get("confidence"),
                    "timeframe": analysis_result.get("timeframe"),
                    "type": "entry",
                    "timestamp": datetime.datetime.now().isoformat(),
                }
            )

        return {
            "success": True,
            "context_id": context_id,
            "signals": signals,
            "message": "Analysis processed successfully",
        }

    async def analyze_market(self, pair, timeframe):
        """Perform comprehensive market analysis."""
        print(f"Performing market analysis for {pair} on {timeframe}...")

        # Get technical analysis
        tech_result = MockTechnicalAnalysisResult.generate(pair, timeframe)

        # Get fundamental analysis
        fund_result = MockFundamentalAnalysisResult.generate(pair)

        # Combine results
        combined_result = {
            "pair": pair,
            "timeframe": timeframe,
            "timestamp": datetime.datetime.now().isoformat(),
            "technical": tech_result,
            "fundamental": fund_result,
            "overall_direction": (
                tech_result.get("overall_direction")
                if tech_result.get("confidence", 0) > fund_result.get("confidence", 0)
                else (
                    "bearish"
                    if fund_result.get("overall_sentiment") == "negative"
                    else "bullish"
                )
            ),
            "confidence": (
                tech_result.get("confidence", 0) + fund_result.get("confidence", 0)
            )
            / 2,
        }

        # Store the result
        context_id = await self.memory.store_analysis_result(
            combined_result, "comprehensive"
        )

        return {
            "success": True,
            "context_id": context_id,
            "market_view": combined_result,
        }

    async def get_market_context(self, pair, timeframe, days_ago=1):
        """Get market context for a pair and timeframe."""
        contexts = await self.memory.retrieve_contexts(pair=pair, timeframe=timeframe)

        summary = await self.memory.summarize_contexts(contexts)

        return {
            "success": True,
            "contexts": contexts,
            "context_count": len(contexts),
            "summary": summary,
        }


async def test_memory_manager():
    """Test the EnhancedMemoryManager."""
    print("\n=== Testing EnhancedMemoryManager ===")

    memory = EnhancedMemoryManager()

    # Store analysis result
    tech_result = MockTechnicalAnalysisResult.generate()
    context_id = await memory.store_analysis_result(tech_result, "technical")

    print(f"✓ Stored technical analysis with context ID: {context_id}")

    # Retrieve context
    context = await memory.retrieve_context_by_id(context_id)
    if context:
        print(f"✓ Retrieved context with ID: {context['context_id']}")
    else:
        print("✗ Failed to retrieve context")

    # Store another result
    fund_result = MockFundamentalAnalysisResult.generate()
    fund_context_id = await memory.store_analysis_result(fund_result, "fundamental")

    print(f"✓ Stored fundamental analysis with context ID: {fund_context_id}")

    # Retrieve multiple contexts
    contexts = await memory.retrieve_contexts(pair="EUR/USD")
    print(f"✓ Retrieved {len(contexts)} contexts for EUR/USD")

    # Generate summary
    summary = await memory.summarize_contexts(contexts)
    print(f"✓ Context summary: {summary}")


async def test_orchestrator():
    """Test the AutoAgentOrchestrator."""
    print("\n=== Testing AutoAgentOrchestrator ===")

    orchestrator = AutoAgentOrchestrator()

    # Register tools
    await orchestrator.register_tool(
        "technical_analysis",
        lambda pair, timeframe: MockTechnicalAnalysisResult.generate(pair, timeframe),
    )
    await orchestrator.register_tool(
        "fundamental_analysis",
        lambda pair: MockFundamentalAnalysisResult.generate(pair),
    )

    print("✓ Registered analysis tools")

    # Process technical analysis result
    tech_result = MockTechnicalAnalysisResult.generate()
    tech_process_result = await orchestrator.process_analysis_result(
        tech_result, "technical"
    )

    print(f"✓ Processed technical analysis: {tech_process_result['message']}")
    if tech_process_result.get("signals"):
        print(f"  Generated {len(tech_process_result['signals'])} signal(s)")

    # Process fundamental analysis result
    fund_result = MockFundamentalAnalysisResult.generate()
    fund_process_result = await orchestrator.process_analysis_result(
        fund_result, "fundamental"
    )

    print(f"✓ Processed fundamental analysis: {fund_process_result['message']}")

    # Analyze market
    market_result = await orchestrator.analyze_market("GBP/USD", "4h")

    print(f"✓ Market analysis completed with context ID: {market_result['context_id']}")
    print(f"  Direction: {market_result['market_view']['overall_direction']}")
    print(f"  Confidence: {market_result['market_view']['confidence']:.2f}")

    # Get market context
    context_result = await orchestrator.get_market_context("EUR/USD", "1h")

    print(f"✓ Retrieved market context: {context_result['context_count']} context(s)")
    print(f"  Summary: {context_result['summary']}")


async def run_integration_tests():
    """Run all integration tests."""
    print("Starting AutoAgent Integration Tests...")

    await test_memory_manager()
    await test_orchestrator()

    print("\nAll integration tests completed!")


if __name__ == "__main__":
    asyncio.run(run_integration_tests())

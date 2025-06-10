"""
Comprehensive tests for AutoAgent integration with AI Forex trading system.

This script tests the functionality of the AutoAgent integration components,
including memory management, orchestration, tool wrappers, and workflows.
"""

import asyncio
import os
import json
from datetime import datetime, timedelta

from forex_ai.integration import create_orchestrator
from forex_ai.integration.enhanced_memory_manager import EnhancedMemoryManager
from forex_ai.integration.tools.technical_tools import get_technical_tools
from forex_ai.integration.tools.fundamental_tools import get_fundamental_tools
from forex_ai.integration.tools.correlation_tools import get_correlation_tools
from forex_ai.integration.tools.signal_tools import get_signal_tools


async def test_memory_manager():
    """Test EnhancedMemoryManager functionality."""
    print("\n=== Testing EnhancedMemoryManager ===")

    # Initialize memory manager
    memory = EnhancedMemoryManager({"schema_prefix": "test_", "cache_size": 10})

    # Ensure tables exist
    await memory.ensure_tables_exist()
    print("✓ Tables initialized successfully")

    # Store test analysis result
    test_result = {
        "pair": "EUR/USD",
        "timeframe": "1h",
        "overall_direction": "bullish",
        "confidence": 0.82,
        "indicators": {
            "rsi": {"value": 65, "interpretation": "bullish"},
            "macd": {"signal": "bullish", "histogram": 0.0031},
        },
        "patterns": [
            {"name": "double_bottom", "direction": "bullish", "strength": 0.75}
        ],
        "support_levels": [1.0920, 1.0890],
        "resistance_levels": [1.1050, 1.1080],
    }

    context_id = await memory.store_analysis_result(test_result, "technical")
    print(f"✓ Analysis stored with context ID: {context_id}")

    # Retrieve context by ID
    retrieved_context = await memory.retrieve_context_by_id(context_id)
    if retrieved_context:
        print(
            f"✓ Context retrieved successfully: {retrieved_context['pair']} - {retrieved_context['timeframe']}"
        )
    else:
        print("✗ Failed to retrieve context")

    # Store a related analysis
    related_result = {
        "pair": "EUR/USD",
        "timeframe": "4h",
        "overall_direction": "bullish",
        "confidence": 0.75,
        "indicators": {
            "rsi": {"value": 62, "interpretation": "bullish"},
        },
        "patterns": [],
    }

    related_id = await memory.store_analysis_result(related_result, "technical")
    print(f"✓ Related analysis stored with context ID: {related_id}")

    # Test context retrieval with filters
    contexts = await memory.retrieve_context(pair="EUR/USD", days_ago=1, limit=5)
    print(f"✓ Retrieved {len(contexts)} contexts with filters")

    # Test context summarization
    if contexts:
        summary = await memory.summarize_contexts(contexts)
        print(f"✓ Generated context summary: {summary[:100]}...")

    print("EnhancedMemoryManager tests completed")


async def test_technical_tools():
    """Test technical analysis tool wrappers."""
    print("\n=== Testing Technical Analysis Tools ===")

    # Get technical tools
    tools = get_technical_tools()
    print(f"✓ Loaded {len(tools)} technical analysis tools")

    # Test tool execution methods exist
    for tool in tools:
        print(f"✓ Tool available: {tool.description.name}")
        assert hasattr(
            tool, "execute"
        ), f"Tool {tool.description.name} missing execute method"

    print("Technical tool tests completed")


async def test_fundamental_tools():
    """Test fundamental analysis tool wrappers."""
    print("\n=== Testing Fundamental Analysis Tools ===")

    # Get fundamental tools
    tools = get_fundamental_tools()
    print(f"✓ Loaded {len(tools)} fundamental analysis tools")

    # Test tool execution methods exist
    for tool in tools:
        print(f"✓ Tool available: {tool.description.name}")
        assert hasattr(
            tool, "execute"
        ), f"Tool {tool.description.name} missing execute method"

    print("Fundamental tool tests completed")


async def test_correlation_tools():
    """Test correlation analysis tool wrappers."""
    print("\n=== Testing Correlation Analysis Tools ===")

    # Get correlation tools
    tools = get_correlation_tools()
    print(f"✓ Loaded {len(tools)} correlation analysis tools")

    # Test tool execution methods exist
    for tool in tools:
        print(f"✓ Tool available: {tool.description.name}")
        assert hasattr(
            tool, "execute"
        ), f"Tool {tool.description.name} missing execute method"

    print("Correlation tool tests completed")


async def test_signal_tools():
    """Test signal generation tool wrappers."""
    print("\n=== Testing Signal Generation Tools ===")

    # Get signal tools
    tools = get_signal_tools()
    print(f"✓ Loaded {len(tools)} signal generation tools")

    # Test tool execution methods exist
    for tool in tools:
        print(f"✓ Tool available: {tool.description.name}")
        assert hasattr(
            tool, "execute"
        ), f"Tool {tool.description.name} missing execute method"

    print("Signal tool tests completed")


async def test_orchestrator():
    """Test AutoAgent orchestrator functionality."""
    print("\n=== Testing AutoAgent Orchestrator ===")

    # Create orchestrator
    config = {
        "memory_config": {"schema_prefix": "test_", "cache_size": 20},
        "model": "gpt-4",
        "temperature": 0.2,
        "confidence_threshold": 0.6,
    }

    orchestrator = await create_orchestrator(config)
    print("✓ Orchestrator created successfully")

    # Test market analysis
    pair = "EUR/USD"
    timeframe = "1h"

    try:
        print(f"Testing market analysis for {pair} on {timeframe}...")
        analysis_result = await orchestrator.analyze_market(pair, timeframe)

        if analysis_result.get("success"):
            print("✓ Market analysis executed successfully")
            print(f"  Context ID: {analysis_result.get('context_id')}")
        else:
            print(f"✗ Market analysis failed: {analysis_result.get('message')}")
    except Exception as e:
        print(f"✗ Market analysis error: {str(e)}")

    # Test process_analysis_result
    try:
        print("Testing analysis result processing...")

        # Simulate a technical analysis result
        technical_result = {
            "pair": "EUR/USD",
            "timeframe": "1h",
            "overall_direction": "bullish",
            "confidence": 0.78,
            "indicators": {"rsi": {"value": 63, "interpretation": "bullish"}},
            "generate_signal": True,
        }

        process_result = await orchestrator.process_analysis_result(
            technical_result, "technical"
        )

        if process_result.get("success"):
            print("✓ Analysis result processed successfully")
            if "signals" in process_result:
                print(f"  Generated {len(process_result.get('signals', []))} signals")
        else:
            print(f"✗ Analysis processing failed: {process_result.get('message')}")
    except Exception as e:
        print(f"✗ Analysis processing error: {str(e)}")

    # Test market context retrieval
    try:
        print("Testing market context retrieval...")
        context_result = await orchestrator.get_market_context(
            pair, timeframe, days_ago=1
        )

        if context_result.get("success"):
            print("✓ Market context retrieved successfully")
            print(f"  Found {context_result.get('context_count')} contexts")
        else:
            print(f"✗ Context retrieval failed: {context_result.get('message')}")
    except Exception as e:
        print(f"✗ Context retrieval error: {str(e)}")

    # Stop the orchestrator
    await orchestrator.stop()
    print("✓ Orchestrator stopped successfully")

    print("Orchestrator tests completed")


async def test_data_routing_integration():
    """Test integration with DataRouter."""
    print("\n=== Testing DataRouter Integration ===")

    try:
        from forex_ai.core.data_router import DataRouter, ProcessingTier, DataType

        # Initialize DataRouter
        router = DataRouter()

        # Initialize orchestrator
        orchestrator = await create_orchestrator()

        # Register handlers
        async def orchestration_handler(data, data_type):
            print(f"✓ Orchestration handler called with {data_type}")
            if data_type == DataType.ECONOMIC_INDICATOR.value:
                result = await orchestrator.process_analysis_result(data, "fundamental")
                return result
            return {"success": True, "message": "Data processed"}

        router.register_orchestration_handler(orchestration_handler)

        # Start router
        await router.start()
        print("✓ DataRouter started")

        # Route test data
        data = {
            "pair": "EUR/USD",
            "indicator": "interest_rate",
            "value": 4.5,
            "previous": 4.25,
            "timestamp": datetime.now().isoformat(),
        }

        result = await router.route(data, DataType.ECONOMIC_INDICATOR.value)
        print(f"✓ Data routed successfully: {result.get('message')}")

        # Get metrics
        metrics = router.get_metrics()
        print(f"✓ Router metrics recorded: {metrics.get('route_counts')}")

        # Stop router
        await router.stop()
        await orchestrator.stop()
        print("✓ DataRouter stopped")

        print("DataRouter integration tests completed")
    except ImportError:
        print("✗ Could not import DataRouter for testing")


async def run_tests():
    """Run all tests."""
    print("Starting AutoAgent integration tests...")

    try:
        # Core components
        await test_memory_manager()
        await test_technical_tools()
        await test_fundamental_tools()
        await test_correlation_tools()
        await test_signal_tools()
        await test_orchestrator()
        await test_data_routing_integration()

        print("\nAll tests completed!")
    except Exception as e:
        print(f"\nTest suite error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(run_tests())

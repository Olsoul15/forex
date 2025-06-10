"""
Example usage of AutoAgent integration with AI Forex trading system.

This example demonstrates how to initialize the AutoAgent orchestrator,
perform a comprehensive market analysis, and generate trading signals.
"""

import asyncio
import os
import json
from datetime import datetime

from forex_ai.integration import create_orchestrator


async def main():
    """Run the AutoAgent integration example."""
    print("Starting AutoAgent integration example...")

    # Create and initialize orchestrator with custom configuration
    config = {
        "memory_config": {"schema_prefix": "example_", "cache_size": 50},
        "model": "gpt-4",
        "temperature": 0.3,
        "confidence_threshold": 0.65,
    }

    print("Initializing AutoAgent orchestrator...")
    orchestrator = await create_orchestrator(config)
    print("Orchestrator initialized successfully")

    # Example 1: Perform comprehensive market analysis
    print("\n--- Example 1: Comprehensive Market Analysis ---")

    pair = "EUR/USD"
    timeframe = "4h"

    print(f"Analyzing {pair} on {timeframe} timeframe...")
    analysis_result = await orchestrator.analyze_market(pair=pair, timeframe=timeframe)

    if analysis_result.get("success"):
        print("Analysis completed successfully")
        print(f"Context ID: {analysis_result.get('context_id')}")

        market_view = analysis_result.get("market_view", {})
        print(f"Market Direction: {market_view.get('overall_direction', 'unknown')}")
        print(f"Confidence: {market_view.get('confidence', 0)}")

        # Print key findings if available
        if "key_findings" in market_view:
            print("\nKey Findings:")
            for finding in market_view.get("key_findings", [])[
                :3
            ]:  # Show top 3 findings
                print(f"- {finding}")
    else:
        print(f"Analysis failed: {analysis_result.get('message')}")

    # Example 2: Process individual analysis results
    print("\n--- Example 2: Processing Individual Analysis Results ---")

    # Simulate a technical analysis result
    technical_result = {
        "pair": "EUR/USD",
        "timeframe": "4h",
        "overall_direction": "bullish",
        "confidence": 0.78,
        "indicators": {
            "rsi": {"value": 62, "interpretation": "bullish"},
            "macd": {"signal": "bullish", "histogram": 0.0025},
            "moving_averages": {"alignment": "bullish", "crossovers": True},
        },
        "patterns": [{"name": "engulfing", "direction": "bullish", "strength": 0.8}],
        "support_levels": [1.0950, 1.0920],
        "resistance_levels": [1.1020, 1.1050],
        "generate_signal": True,
    }

    print("Processing technical analysis result...")
    process_result = await orchestrator.process_analysis_result(
        analysis_result=technical_result, analysis_type="technical"
    )

    if process_result.get("success"):
        print("Analysis processed successfully")

        if "signals" in process_result:
            signals = process_result.get("signals", [])
            print(f"Generated {len(signals)} trading signals")

            for i, signal in enumerate(signals):
                print(f"\nSignal {i+1}:")
                print(f"Type: {signal.get('signal_type')}")
                print(f"Direction: {signal.get('direction')}")
                print(f"Strength: {signal.get('strength')}")
                print(f"Confidence: {signal.get('confidence')}")

                # Print price levels if available
                if "entry_price" in signal:
                    print(f"Entry: {signal.get('entry_price')}")
                    print(f"Target: {signal.get('target_price')}")
                    print(f"Stop Loss: {signal.get('stop_loss')}")
        else:
            print("No signals generated")
    else:
        print(f"Processing failed: {process_result.get('message')}")

    # Example 3: Get historical context
    print("\n--- Example 3: Getting Historical Context ---")

    print(f"Retrieving recent context for {pair} on {timeframe} timeframe...")
    context_result = await orchestrator.get_market_context(
        pair=pair, timeframe=timeframe, days_ago=7  # Look back 7 days
    )

    if context_result.get("success"):
        print("Context retrieved successfully")
        print(f"Found {context_result.get('context_count')} context entries")

        # Print summary if available
        summary = context_result.get("summary")
        if summary:
            print("\nContext Summary:")
            print(summary)
    else:
        print(f"Context retrieval failed: {context_result.get('message')}")

    # Example 4: Strategy validation
    print("\n--- Example 4: Strategy Validation ---")

    strategy_id = "trend_following"
    start_date = (
        datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    )[:10]
    end_date = start_date  # Same day, use historical data

    print(f"Validating {strategy_id} strategy for {pair}...")
    validation_result = await orchestrator.validate_strategy(
        strategy_id=strategy_id,
        pair=pair,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
    )

    if validation_result.get("success"):
        print("Strategy validation completed successfully")

        performance = validation_result.get("performance_metrics", {})
        print("\nPerformance Metrics:")
        print(f"Win Rate: {performance.get('win_rate', 0):.2f}")
        print(f"Profit Factor: {performance.get('profit_factor', 0):.2f}")
        print(f"Sharpe Ratio: {performance.get('sharpe_ratio', 0):.2f}")
    else:
        print(f"Strategy validation failed: {validation_result.get('message')}")

    # Cleanup
    print("\nStopping orchestrator...")
    await orchestrator.stop()
    print("Orchestrator stopped")

    print("\nAutoAgent integration example completed")


if __name__ == "__main__":
    asyncio.run(main())

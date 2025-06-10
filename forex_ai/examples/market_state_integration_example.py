"""
Market State Integration Example

This module demonstrates the various approaches for integrating market state detection
into the forex_ai system, including:

1. Structured Workflow Approach (StrategyOrchestrator)
2. Agent-Based Approach (AdvancedOrchestrator)
3. Template-Based Strategies
4. AutoAgentOrchestrator Integration

Run this example to see how market state detection can be used in different contexts.
"""

import os
import asyncio
import logging
from typing import Dict, List, Any, Optional
import json
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


async def example_strategy_orchestrator():
    """
    Example of using market state detection with the StrategyOrchestrator.

    This demonstrates the Structured Workflow Approach where MarketStateStrategy
    is registered with the StrategyOrchestrator like any other strategy.
    """
    try:
        from auto_agent.core.orchestrator import get_orchestrator
        from auto_agent.strategies import MarketStateStrategy
        from auto_agent.core.data_manager import get_data_manager

        logger.info("=== Example 1: StrategyOrchestrator with MarketStateStrategy ===")

        # Get the orchestrator instance
        orchestrator = get_orchestrator()

        # Create MarketStateStrategy
        market_state_strategy = MarketStateStrategy(
            strategy_id="market_state_example",
            min_confidence_threshold=0.6,
            reversal_confidence_threshold=0.7,
        )

        # Register with orchestrator
        orchestrator.register_strategy(
            market_state_strategy.strategy_id, market_state_strategy
        )

        # Set up data manager for the orchestrator
        data_manager = get_data_manager()
        orchestrator.register_data_manager(data_manager)

        logger.info("MarketStateStrategy registered with StrategyOrchestrator")
        logger.info("In a real application, you would also register:")
        logger.info("- feature_aggregator")
        logger.info("- risk_manager")
        logger.info("- execution_agent")
        logger.info("- state_manager")
        logger.info("Then start the orchestrator with: await orchestrator.start()")

    except ImportError as e:
        logger.warning(f"Could not run StrategyOrchestrator example: {e}")
        logger.info("This example requires the auto_agent package to be installed.")


async def example_advanced_orchestrator():
    """
    Example of using market state detection with the AdvancedOrchestrator.

    This demonstrates the Agent-Based Approach where MarketStateAnalysisAgent
    is registered with the AdvancedOrchestrator and used as part of a workflow.
    """
    try:
        from auto_agent.agent.manager.advanced_orchestrator import (
            create_advanced_orchestrator,
            TaskPriority,
        )
        from forex_ai.agents.workflows.market_state_workflow import (
            create_market_state_workflow,
        )

        logger.info(
            "\n=== Example 2: AdvancedOrchestrator with MarketStateAnalysisAgent ==="
        )

        # Create advanced orchestrator
        orchestrator = await create_advanced_orchestrator()

        # Create market state workflow
        workflow = await create_market_state_workflow(orchestrator)

        # Example of creating market state analysis tasks
        pairs = ["EUR_USD", "GBP_USD"]
        timeframes = ["H1", "H4"]

        # Mock features dictionary
        features = {
            "EUR_USD": {
                "H1": {
                    "close": [1.1001, 1.1005, 1.1010, 1.1015, 1.1020],
                    "high": [1.1015, 1.1010, 1.1025, 1.1025, 1.1030],
                    "low": [1.0995, 1.1000, 1.1005, 1.1010, 1.1015],
                    "open": [1.1000, 1.1001, 1.1005, 1.1010, 1.1015],
                    "adx": [25.5, 26.2, 27.0, 28.3, 29.1],
                    "rsi": [55.2, 57.5, 60.3, 63.8, 67.2],
                    "bollinger_bands": {
                        "upper": [1.1050, 1.1055, 1.1060, 1.1065, 1.1070],
                        "middle": [1.1000, 1.1005, 1.1010, 1.1015, 1.1020],
                        "lower": [1.0950, 1.0955, 1.0960, 1.0965, 1.0970],
                    },
                    "atr": [0.0020, 0.0021, 0.0022, 0.0023, 0.0024],
                },
                "H4": {},  # Empty for demonstration of error handling
            },
            "GBP_USD": {
                "H1": {
                    "close": [1.3001, 1.2995, 1.2990, 1.2985, 1.2980],
                    "high": [1.3015, 1.3010, 1.3000, 1.2995, 1.2990],
                    "low": [1.2995, 1.2990, 1.2985, 1.2980, 1.2975],
                    "open": [1.3000, 1.3001, 1.2995, 1.2990, 1.2985],
                    "adx": [18.2, 17.5, 16.8, 16.1, 15.5],
                    "rsi": [45.2, 43.5, 42.1, 40.8, 39.5],
                    "bollinger_bands": {
                        "upper": [1.3050, 1.3045, 1.3040, 1.3035, 1.3030],
                        "middle": [1.3000, 1.2995, 1.2990, 1.2985, 1.2980],
                        "lower": [1.2950, 1.2945, 1.2940, 1.2935, 1.2930],
                    },
                    "atr": [0.0022, 0.0021, 0.0020, 0.0019, 0.0018],
                },
                "H4": {},
            },
        }

        # Create dependent task generator function
        async def trading_task_generator(market_state_task_ids):
            """Create trading tasks that depend on market state analysis."""
            logger.info(
                f"Creating trading tasks dependent on {len(market_state_task_ids)} market state tasks"
            )

            # In a real application, this would create trading tasks that
            # leverage the market state information from the prerequisites
            trading_task_ids = []

            for i, market_state_task_id in enumerate(market_state_task_ids):
                task_id = await orchestrator.add_task(
                    title=f"Trading Decision {i+1}",
                    description=f"Make trading decision based on market state analysis",
                    priority=TaskPriority.MEDIUM,
                    prerequisites=[market_state_task_id],
                    required_capabilities=["trading_decision"],
                    context={"market_state_task_id": market_state_task_id},
                )
                trading_task_ids.append(task_id)

            return trading_task_ids

        # Create market state dependency chain
        task_ids = await workflow.create_market_state_dependency_chain(
            pairs=pairs,
            timeframes=timeframes,
            features=features,
            dependent_task_generator=trading_task_generator,
            priority=TaskPriority.HIGH,
        )

        logger.info(f"Created {len(task_ids)} tasks in the dependency chain")
        logger.info("In a real application, the orchestrator would process these tasks")
        logger.info("and use the market state information to guide trading decisions.")

        # Clean up
        await orchestrator.shutdown()

    except ImportError as e:
        logger.warning(f"Could not run AdvancedOrchestrator example: {e}")
        logger.info("This example requires the auto_agent package to be installed.")


async def example_template_based_strategy():
    """
    Example of using market state detection with template-based strategies.

    This demonstrates how existing strategies can be enhanced with market state
    awareness using the MarketStateAwareMixin.
    """
    try:
        from forex_ai.analysis.technical.strategies.trend_following_state_aware import (
            create_trend_following_state_aware_strategy,
        )

        logger.info(
            "\n=== Example 3: Template-Based Strategy with MarketStateAwareMixin ==="
        )

        # Create the strategy
        strategy = create_trend_following_state_aware_strategy(
            fast_ma_period=15,
            slow_ma_period=30,
            market_state_params={
                "min_confidence_threshold": 0.5,
                "trend_bias_multiplier": 1.5,
                "range_bias_multiplier": 0.7,
            },
        )

        # Mock data for testing
        data = {
            "open": [1.1001, 1.1005, 1.1010, 1.1015, 1.1020, 1.1025, 1.1030],
            "high": [1.1015, 1.1010, 1.1025, 1.1025, 1.1030, 1.1035, 1.1040],
            "low": [1.0995, 1.1000, 1.1005, 1.1010, 1.1015, 1.1020, 1.1025],
            "close": [1.1005, 1.1010, 1.1015, 1.1020, 1.1025, 1.1030, 1.1035],
            "volume": [1000, 1200, 1300, 1100, 1400, 1500, 1600],
            # Pre-calculated indicators
            "ema_fast": [1.1003, 1.1007, 1.1012, 1.1017, 1.1022, 1.1027, 1.1032],
            "ema_slow": [1.1002, 1.1004, 1.1008, 1.1012, 1.1017, 1.1022, 1.1027],
            "adx": [22.5, 23.2, 24.0, 25.3, 26.1, 27.0, 28.2],
            "rsi": [52.2, 54.5, 56.3, 58.8, 60.2, 62.5, 64.8],
            "bollinger_bands": {
                "upper": [1.1050, 1.1055, 1.1060, 1.1065, 1.1070, 1.1075, 1.1080],
                "middle": [1.1000, 1.1005, 1.1010, 1.1015, 1.1020, 1.1025, 1.1030],
                "lower": [1.0950, 1.0955, 1.0960, 1.0965, 1.0970, 1.0975, 1.0980],
            },
            "atr": [0.0020, 0.0021, 0.0022, 0.0023, 0.0024, 0.0025, 0.0026],
        }

        # Generate signals
        pair = "EUR_USD"
        timeframe = "H1"

        signals = strategy.generate_signals(pair, timeframe, data)

        logger.info(f"Generated {len(signals)} signals")
        for signal in signals:
            logger.info(f"Signal: {signal.direction} {pair} at {signal.entry_price}")
            logger.info(f"  Strength: {signal.strength:.4f}")
            logger.info(f"  Stop Loss: {signal.stop_loss_pips:.2f} pips")
            logger.info(f"  Take Profit: {signal.take_profit_pips:.2f} pips")

            if "market_state" in signal.metadata:
                market_state = signal.metadata["market_state"]
                logger.info(f"  Market State: {market_state.get('state_type')}")
                logger.info(
                    f"  Market State Confidence: {market_state.get('confidence', 0):.2f}"
                )

    except ImportError as e:
        logger.warning(f"Could not run Template-Based Strategy example: {e}")
        logger.info("Make sure the required modules are installed.")


async def example_autoagent_orchestrator():
    """
    Example of integrating market state detection with the AutoAgentOrchestrator.

    This demonstrates how market state awareness can be added to the
    comprehensive market analysis performed by the AutoAgentOrchestrator.
    """
    try:
        from forex_ai.integration.autoagent_orchestrator import AutoAgentOrchestrator
        from forex_ai.integration.market_state_autoagent_integration import (
            enhance_autoagent_with_market_state,
        )

        logger.info(
            "\n=== Example 4: AutoAgentOrchestrator with Market State Integration ==="
        )

        # Create the basic orchestrator
        orchestrator = AutoAgentOrchestrator(config={"confidence_threshold": 0.6})

        # Enhance it with market state capabilities
        integrator = enhance_autoagent_with_market_state(orchestrator)

        # Mock a sample market analysis call
        pair = "EUR_USD"
        timeframe = "H1"

        # Mock the original analyze_market method
        original_analyze_market = orchestrator.analyze_market

        async def mock_analyze_market(instrument, timeframe=None, *args, **kwargs):
            """Mock implementation of analyze_market for demonstration."""
            logger.info(f"Mocking analysis for {instrument} on {timeframe}")

            # Return a mock result
            return {
                "success": True,
                "market_view": {
                    "pair": instrument,
                    "timeframe": timeframe,
                    "timestamp": datetime.now().isoformat(),
                    "overall_direction": "neutral",
                    "confidence": 0.5,
                    "technical": {
                        "indicators": {
                            "adx": 22.5,
                            "rsi": 52.8,
                            "ema": {"fast": 1.1032, "slow": 1.1027},
                            "bollinger_bands": {
                                "upper": 1.1080,
                                "middle": 1.1030,
                                "lower": 1.0980,
                            },
                            "atr": 0.0026,
                        },
                        "patterns": ["no_clear_pattern"],
                    },
                    "candles": [
                        {
                            "open": 1.1020,
                            "high": 1.1035,
                            "low": 1.1015,
                            "close": 1.1030,
                            "volume": 1500,
                        },
                        {
                            "open": 1.1030,
                            "high": 1.1040,
                            "low": 1.1025,
                            "close": 1.1035,
                            "volume": 1600,
                        },
                    ],
                },
            }

        # Temporarily replace the method with our mock
        orchestrator.analyze_market = mock_analyze_market

        # Call analyze_market (which will use our enhanced version)
        result = await orchestrator.analyze_market(pair, timeframe)

        # Restore original method for cleanup
        orchestrator.analyze_market = original_analyze_market

        # Display the results
        if result.get("success", False):
            market_view = result.get("market_view", {})

            logger.info(f"Analysis result for {pair} on {timeframe}:")
            logger.info(f"  Overall Direction: {market_view.get('overall_direction')}")
            logger.info(f"  Confidence: {market_view.get('confidence', 0):.2f}")

            # Check if market state was added
            if "market_state" in market_view:
                market_state = market_view.get("market_state", {})
                logger.info(f"  Market State: {market_state.get('state_type')}")
                logger.info(
                    f"  Market State Confidence: {market_state.get('confidence', 0):.2f}"
                )
                logger.info(
                    f"  Market State Summary: {market_view.get('market_state_summary', '')}"
                )

                # Display trading implications
                if (
                    "implications" in market_view
                    and "market_state" in market_view["implications"]
                ):
                    implications = market_view["implications"]["market_state"]
                    logger.info(
                        f"  Trading Bias: {implications.get('bias', 'neutral')}"
                    )
                    logger.info(
                        f"  Position Sizing: {implications.get('position_sizing', 'normal')}"
                    )
                    logger.info(
                        f"  Preferred Strategies: {', '.join(implications.get('preferred_strategies', []))}"
                    )
            else:
                logger.info(
                    "  Market state information not available - may be missing required indicators"
                )
        else:
            logger.warning(f"Analysis failed: {result.get('error', 'Unknown error')}")

    except ImportError as e:
        logger.warning(f"Could not run AutoAgentOrchestrator example: {e}")
        logger.info("Make sure the required modules are installed.")


async def run_all_examples():
    """Run all examples."""
    await example_strategy_orchestrator()
    await example_advanced_orchestrator()
    await example_template_based_strategy()
    await example_autoagent_orchestrator()


if __name__ == "__main__":
    # Run the examples
    asyncio.run(run_all_examples())

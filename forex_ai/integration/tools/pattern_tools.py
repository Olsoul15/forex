"""
Candlestick pattern recognition tools for AutoAgent integration.

This module provides tool wrappers for candlestick pattern recognition
to be used with the AutoAgent system.
"""

import logging
from typing import Dict, List, Any, Optional, Union

from AutoAgent.app_auto_agent.tool.base import (
    Tool,
    ToolDescription,
    ToolParameter,
    ToolContext,
    ToolResult,
)
from forex_ai.integration.patterns.enhanced_pattern_recognition import (
    EnhancedPatternRecognition,
)
from forex_ai.utils.logging import get_logger

logger = get_logger(__name__)


class CandlestickPatternTool(Tool):
    """
    Tool for comprehensive candlestick pattern recognition.

    This tool provides pattern recognition capabilities using TA-Lib's
    extensive candlestick pattern functions through the forex_ai system.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the candlestick pattern tool.

        Args:
            config: Configuration dictionary
        """
        super().__init__()
        self.config = config or {}

        # Initialize pattern recognizers
        self.enhanced_recognizer = EnhancedPatternRecognition(config)

        # Import the new comprehensive pattern recognizer
        from AutoAgent.app_auto_agent.forex_ai.features.candlestick_patterns import (
            get_pattern_recognizer,
        )

        self.comprehensive_recognizer = get_pattern_recognizer()

        # Set up tool description
        self.description = ToolDescription(
            name="candlestick_patterns",
            description="Detect candlestick patterns in forex market data using TA-Lib",
            parameters=[
                ToolParameter(
                    name="pair",
                    type="string",
                    description="Currency pair (e.g., 'EUR/USD')",
                    required=True,
                ),
                ToolParameter(
                    name="timeframe",
                    type="string",
                    description="Timeframe (e.g., '1h', '4h', '1d')",
                    required=True,
                ),
                ToolParameter(
                    name="patterns",
                    type="array",
                    description="List of specific patterns to detect (default: all)",
                    required=False,
                ),
                ToolParameter(
                    name="categories",
                    type="array",
                    description="Categories of patterns to detect (reversal, continuation, indecision)",
                    required=False,
                ),
                ToolParameter(
                    name="confidence_threshold",
                    type="number",
                    description="Minimum confidence threshold (0.0-1.0)",
                    required=False,
                ),
                ToolParameter(
                    name="use_enhanced",
                    type="boolean",
                    description="Whether to use enhanced pattern recognition with confidence scoring",
                    required=False,
                ),
                ToolParameter(
                    name="count",
                    type="integer",
                    description="Number of candles to analyze",
                    required=False,
                ),
            ],
        )

    async def execute(self, context: ToolContext) -> ToolResult:
        """
        Execute candlestick pattern recognition.

        Args:
            context: Tool execution context

        Returns:
            Pattern recognition results
        """
        try:
            # Extract parameters
            pair = context.parameters.get("pair")
            timeframe = context.parameters.get("timeframe")
            patterns = context.parameters.get("patterns")
            categories = context.parameters.get("categories")
            confidence_threshold = float(
                context.parameters.get("confidence_threshold", 0.0)
            )
            use_enhanced = context.parameters.get("use_enhanced", True)
            count = int(context.parameters.get("count", 100))

            # Validate required parameters
            if not pair or not timeframe:
                return ToolResult(
                    success=False, message="Pair and timeframe are required", data=None
                )

            # Expand patterns by category if categories specified
            if categories and not patterns:
                patterns = []
                for category in categories:
                    patterns.extend(
                        self.comprehensive_recognizer.get_available_patterns(category)
                    )

            # Fetch market data
            market_data = await self._fetch_market_data(pair, timeframe, count)
            if not market_data:
                return ToolResult(
                    success=False,
                    message=f"Failed to fetch market data for {pair} on {timeframe}",
                    data=None,
                )

            # Choose recognizer based on parameters
            if use_enhanced:
                # Use both recognizers and combine results
                enhanced_results = await self.enhanced_recognizer.detect_patterns(
                    market_data, pair=pair, timeframe=timeframe
                )

                comprehensive_results = (
                    await self.comprehensive_recognizer.recognize_patterns(
                        market_data,
                        patterns=patterns,
                        confidence_threshold=confidence_threshold,
                    )
                )

                # Combine results
                results = self._combine_recognition_results(
                    enhanced_results, comprehensive_results
                )
            else:
                # Use only the comprehensive recognizer
                results = await self.comprehensive_recognizer.recognize_patterns(
                    market_data,
                    patterns=patterns,
                    confidence_threshold=confidence_threshold,
                )

            return ToolResult(
                success=True,
                message="Candlestick pattern recognition completed successfully",
                data=results,
            )

        except Exception as e:
            logger.error(f"Error executing candlestick pattern recognition: {str(e)}")
            return ToolResult(
                success=False,
                message=f"Error executing candlestick pattern recognition: {str(e)}",
                data=None,
            )

    async def _fetch_market_data(
        self, pair: str, timeframe: str, count: int
    ) -> Dict[str, Any]:
        """
        Fetch market data for the specified pair and timeframe.

        Args:
            pair: Currency pair
            timeframe: Timeframe for analysis
            count: Number of candles to fetch

        Returns:
            Dictionary with OHLC data
        """
        try:
            # Try to get data fetcher from config
            data_fetcher = self.config.get("data_fetcher")

            if data_fetcher:
                # Use the provided data fetcher
                return await data_fetcher.get_candles(
                    instrument=pair, timeframe=timeframe, count=count
                )
            else:
                # No data fetcher, attempt to use the market data service
                from forex_ai.data.market_data import get_market_data_service

                market_data_service = get_market_data_service()

                return await market_data_service.get_candles(
                    instrument=pair, timeframe=timeframe, count=count
                )
        except Exception as e:
            logger.error(f"Error fetching market data: {str(e)}")
            return None

    def _combine_recognition_results(
        self, enhanced_results: Dict[str, Any], comprehensive_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Combine results from both pattern recognizers.

        Args:
            enhanced_results: Results from enhanced recognizer
            comprehensive_results: Results from comprehensive recognizer

        Returns:
            Combined results
        """
        # Start with comprehensive results as base
        combined = comprehensive_results.copy()

        # Add enhanced patterns that aren't in comprehensive
        enhanced_patterns = enhanced_results.get("patterns", [])
        comprehensive_patterns = comprehensive_results.get("patterns", [])

        # Create a set of pattern names in comprehensive results
        comp_pattern_names = {p.get("pattern") for p in comprehensive_patterns}

        # Add enhanced patterns that aren't already included
        for pattern in enhanced_patterns:
            if pattern.get("pattern_type") not in comp_pattern_names:
                # Convert format to match comprehensive
                converted = {
                    "pattern": pattern.get("pattern_type"),
                    "direction": pattern.get("direction"),
                    "confidence": pattern.get("confidence", 0.6),
                    "category": pattern.get("category", "unknown"),
                    "completion_index": pattern.get("end_idx", 0),
                    "start_index": pattern.get("start_idx", 0),
                    "completion_price": pattern.get("price", 0.0),
                }
                combined["patterns"].append(converted)

        # Update counts
        combined["detected_count"] = len(combined["patterns"])

        # Recalculate strongest pattern
        if combined["patterns"]:
            combined["strongest_pattern"] = max(
                combined["patterns"], key=lambda x: x.get("confidence", 0)
            )

        return combined


def get_pattern_tools(config: Optional[Dict[str, Any]] = None) -> List[Tool]:
    """
    Get all pattern recognition tools.

    Args:
        config: Configuration dictionary

    Returns:
        List of pattern recognition tools
    """
    return [CandlestickPatternTool(config)]

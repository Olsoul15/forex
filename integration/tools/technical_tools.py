"""
Technical analysis tool wrappers for AutoAgent integration.

This module provides AutoAgent tool wrappers around the existing technical
analysis modules in the AI Forex system.
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
from forex_ai.agents.technical_analysis import TechnicalAnalysisAgent
from forex_ai.analysis.technical.patterns import PatternRecognition
from forex_ai.analysis.technical.indicators import IndicatorCalculator
from forex_ai.utils.logging import get_logger

logger = get_logger(__name__)


class TechnicalAnalysisTool(Tool):
    """
    Tool wrapper for the technical analysis agent.

    This tool provides comprehensive technical analysis capabilities
    by wrapping the existing TechnicalAnalysisAgent.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the technical analysis tool.

        Args:
            config: Configuration dictionary
        """
        super().__init__()
        self.config = config or {}
        self.agent = TechnicalAnalysisAgent(
            name="TechnicalForAutoAgent", config=self.config.get("agent_config", {})
        )

        # Set up tool description
        self.description = ToolDescription(
            name="technical_analysis",
            description="Perform technical analysis on forex market data",
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
                    name="indicators",
                    type="array",
                    description="List of indicators to calculate (e.g., ['rsi', 'macd', 'bollinger'])",
                    required=False,
                ),
                ToolParameter(
                    name="patterns",
                    type="array",
                    description="List of patterns to detect (e.g., ['engulfing', 'doji', 'hammer'])",
                    required=False,
                ),
                ToolParameter(
                    name="start_date",
                    type="string",
                    description="Start date for analysis (ISO format, e.g., '2023-01-01')",
                    required=False,
                ),
                ToolParameter(
                    name="end_date",
                    type="string",
                    description="End date for analysis (ISO format, e.g., '2023-01-31')",
                    required=False,
                ),
                ToolParameter(
                    name="use_ml",
                    type="boolean",
                    description="Whether to use machine learning predictions",
                    required=False,
                ),
            ],
        )

    async def execute(self, context: ToolContext) -> ToolResult:
        """
        Execute technical analysis.

        Args:
            context: Tool execution context

        Returns:
            Analysis results
        """
        try:
            # Extract parameters
            pair = context.parameters.get("pair")
            timeframe = context.parameters.get("timeframe")
            indicators = context.parameters.get("indicators")
            patterns = context.parameters.get("patterns")
            start_date = context.parameters.get("start_date")
            end_date = context.parameters.get("end_date")
            use_ml = context.parameters.get("use_ml", False)

            # Validate required parameters
            if not pair or not timeframe:
                return ToolResult(
                    success=False, message="Pair and timeframe are required", data=None
                )

            # Prepare input for the agent
            input_data = {
                "pair": pair,
                "timeframe": timeframe,
            }

            # Add optional parameters
            if indicators:
                input_data["indicators"] = indicators
            if patterns:
                input_data["patterns"] = patterns
            if start_date:
                input_data["start_date"] = start_date
            if end_date:
                input_data["end_date"] = end_date
            if use_ml:
                input_data["use_ml"] = use_ml

            # Execute analysis
            logger.info(
                f"Executing technical analysis for {pair} on {timeframe} timeframe"
            )
            result = self.agent.analyze(**input_data)

            # Process result
            if not result.success:
                return ToolResult(
                    success=False,
                    message=f"Technical analysis failed: {result.message}",
                    data=None,
                )

            return ToolResult(
                success=True,
                message="Technical analysis completed successfully",
                data=result.data,
            )

        except Exception as e:
            logger.error(f"Error executing technical analysis: {str(e)}")
            return ToolResult(
                success=False,
                message=f"Error executing technical analysis: {str(e)}",
                data=None,
            )


class PatternRecognitionTool(Tool):
    """
    Tool wrapper for pattern recognition.

    This tool provides pattern recognition capabilities by wrapping
    the existing PatternRecognition class.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the pattern recognition tool.

        Args:
            config: Configuration dictionary
        """
        super().__init__()
        self.config = config or {}
        self.pattern_recognizer = PatternRecognition()

        # Set up tool description
        self.description = ToolDescription(
            name="pattern_recognition",
            description="Detect technical patterns in forex market data",
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
                    description="List of patterns to detect, or 'all' for all patterns",
                    required=False,
                ),
                ToolParameter(
                    name="lookback",
                    type="integer",
                    description="Number of candles to look back",
                    required=False,
                ),
            ],
        )

    async def execute(self, context: ToolContext) -> ToolResult:
        """
        Execute pattern recognition.

        Args:
            context: Tool execution context

        Returns:
            Pattern recognition results
        """
        try:
            # Extract parameters
            pair = context.parameters.get("pair")
            timeframe = context.parameters.get("timeframe")
            patterns = context.parameters.get("patterns", ["all"])
            lookback = context.parameters.get("lookback", 100)

            # Validate required parameters
            if not pair or not timeframe:
                return ToolResult(
                    success=False, message="Pair and timeframe are required", data=None
                )

            # Execute pattern recognition
            logger.info(f"Detecting patterns for {pair} on {timeframe} timeframe")
            result = await self.pattern_recognizer.detect_patterns(
                pair=pair, timeframe=timeframe, patterns=patterns, lookback=lookback
            )

            return ToolResult(
                success=True,
                message=f"Detected {len(result.get('patterns', []))} patterns",
                data=result,
            )

        except Exception as e:
            logger.error(f"Error detecting patterns: {str(e)}")
            return ToolResult(
                success=False, message=f"Error detecting patterns: {str(e)}", data=None
            )


class IndicatorCalculationTool(Tool):
    """
    Tool wrapper for technical indicator calculation.

    This tool provides technical indicator calculation capabilities by wrapping
    the existing IndicatorCalculator class.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the indicator calculation tool.

        Args:
            config: Configuration dictionary
        """
        super().__init__()
        self.config = config or {}
        self.indicator_calculator = IndicatorCalculator()

        # Set up tool description
        self.description = ToolDescription(
            name="indicator_calculation",
            description="Calculate technical indicators for forex market data",
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
                    name="indicators",
                    type="array",
                    description="List of indicators to calculate, or 'all' for all indicators",
                    required=False,
                ),
                ToolParameter(
                    name="lookback",
                    type="integer",
                    description="Number of candles to look back",
                    required=False,
                ),
            ],
        )

    async def execute(self, context: ToolContext) -> ToolResult:
        """
        Execute indicator calculation.

        Args:
            context: Tool execution context

        Returns:
            Indicator calculation results
        """
        try:
            # Extract parameters
            pair = context.parameters.get("pair")
            timeframe = context.parameters.get("timeframe")
            indicators = context.parameters.get("indicators", ["all"])
            lookback = context.parameters.get("lookback", 100)

            # Validate required parameters
            if not pair or not timeframe:
                return ToolResult(
                    success=False, message="Pair and timeframe are required", data=None
                )

            # Execute indicator calculation
            logger.info(f"Calculating indicators for {pair} on {timeframe} timeframe")
            result = await self.indicator_calculator.calculate_indicators(
                pair=pair, timeframe=timeframe, indicators=indicators, lookback=lookback
            )

            return ToolResult(
                success=True,
                message=f"Calculated {len(result.get('indicators', {}))} indicators",
                data=result,
            )

        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            return ToolResult(
                success=False,
                message=f"Error calculating indicators: {str(e)}",
                data=None,
            )


def get_technical_tools(config: Optional[Dict[str, Any]] = None) -> List[Tool]:
    """
    Get all technical analysis tools.

    Args:
        config: Configuration dictionary

    Returns:
        List of technical analysis tools
    """
    tools = [
        TechnicalAnalysisTool(config),
        PatternRecognitionTool(config),
        IndicatorCalculationTool(config),
    ]

    # Add pattern tools from pattern_tools module
    from forex_ai.integration.tools.pattern_tools import get_pattern_tools

    tools.extend(get_pattern_tools(config))

    return tools

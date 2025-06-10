"""
AutoAgent tool wrappers for signal generation in the AI Forex system.

This module provides AutoAgent tool wrappers for signal generation capabilities,
allowing the system to generate actionable trading signals based on combined
analysis results from various sources.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import json
import logging
import uuid

from AutoAgent.app_auto_agent.tool.base import (
    Tool,
    ToolDescription,
    ToolParameter,
    ToolContext,
    ToolResult,
)

from forex_ai.utils.logging import get_logger
from forex_ai.integration.enhanced_memory_manager import EnhancedMemoryManager

logger = get_logger(__name__)


class SignalGeneratorTool(Tool):
    """Tool for generating trading signals from analysis results."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the signal generator tool.

        Args:
            config: Optional configuration dictionary
        """
        super().__init__()
        self.config = config or {}
        self.memory = EnhancedMemoryManager(self.config.get("memory_config"))

        # Define tool metadata
        self.description = ToolDescription(
            name="signal_generation",
            description="Generates trading signals based on analysis results",
            version="1.0.0",
        )

        # Define parameters
        self.parameters = [
            ToolParameter(
                name="pair",
                description="Currency pair (e.g., 'EUR/USD')",
                required=True,
                type="string",
            ),
            ToolParameter(
                name="timeframe",
                description="Timeframe for signal (e.g., '1h', '4h', 'daily')",
                required=True,
                type="string",
            ),
            ToolParameter(
                name="analysis_results",
                description="Analysis results to base signal on",
                required=False,
                type="object",
                default=None,
            ),
            ToolParameter(
                name="context_ids",
                description="List of context IDs for analysis results",
                required=False,
                type="array",
                default=None,
            ),
            ToolParameter(
                name="confidence_threshold",
                description="Minimum confidence threshold for generating a signal",
                required=False,
                type="number",
                default=0.65,
            ),
            ToolParameter(
                name="risk_profile",
                description="Risk profile to use for signal generation (conservative, moderate, aggressive)",
                required=False,
                type="string",
                default="moderate",
            ),
        ]

    async def execute(self, context: ToolContext) -> ToolResult:
        """
        Execute signal generation.

        Args:
            context: Tool execution context with parameters

        Returns:
            Generated trading signal
        """
        # Extract parameters
        pair = context.get_parameter("pair")
        timeframe = context.get_parameter("timeframe")
        analysis_results = context.get_parameter("analysis_results")
        context_ids = context.get_parameter("context_ids")
        confidence_threshold = context.get_parameter("confidence_threshold", 0.65)
        risk_profile = context.get_parameter("risk_profile", "moderate")

        try:
            logger.info(
                f"Generating trading signal for {pair} on {timeframe} timeframe"
            )

            analysis_contexts = []

            # Retrieve analysis results from context IDs if provided
            if context_ids:
                for context_id in context_ids:
                    context_data = await self.memory.retrieve_context_by_id(context_id)
                    if context_data:
                        analysis_contexts.append(context_data)

            # If no context IDs provided or retrieval failed, but analysis_results provided
            if not analysis_contexts and analysis_results:
                # Convert direct analysis results into a list of contexts
                analysis_contexts = [analysis_results]

            # If still no contexts, retrieve recent analyses for this pair and timeframe
            if not analysis_contexts:
                # Get most recent analyses from memory (last 2 days)
                analysis_contexts = await self.memory.retrieve_context(
                    pair=pair, timeframe=timeframe, days_ago=2
                )

            # Generate signal based on available analyses
            if analysis_contexts:
                signal = await self._generate_signal(
                    pair=pair,
                    timeframe=timeframe,
                    analysis_contexts=analysis_contexts,
                    confidence_threshold=confidence_threshold,
                    risk_profile=risk_profile,
                )

                # Store the signal in memory with references to source analyses
                signal_id = str(uuid.uuid4())

                signal_context = {
                    "context_id": signal_id,
                    "pair": pair,
                    "timeframe": timeframe,
                    "analysis_type": "trading_signal",
                    "timestamp": datetime.now().isoformat(),
                    "findings": signal,
                    "confidence": signal.get("confidence", 0),
                    "related_contexts": [
                        ctx.get("context_id")
                        for ctx in analysis_contexts
                        if ctx.get("context_id")
                    ],
                    "tags": ["signal", signal.get("signal_type", ""), risk_profile],
                    "metadata": {
                        "risk_profile": risk_profile,
                        "source_analyses_count": len(analysis_contexts),
                    },
                }

                # Store in memory
                await self.memory._store_context(signal_context)

                # Prepare response
                return ToolResult(
                    success=True,
                    result={
                        "signal_id": signal_id,
                        "pair": pair,
                        "timeframe": timeframe,
                        "signal": signal,
                        "generated_at": datetime.now().isoformat(),
                        "source_analyses": len(analysis_contexts),
                        "risk_profile": risk_profile,
                    },
                )
            else:
                return ToolResult(
                    success=False,
                    error=f"No analysis results available for {pair} on {timeframe} timeframe",
                )

        except Exception as e:
            logger.error(f"Error generating trading signal: {str(e)}")

            return ToolResult(
                success=False, error=f"Error generating trading signal: {str(e)}"
            )

    async def _generate_signal(
        self,
        pair: str,
        timeframe: str,
        analysis_contexts: List[Dict[str, Any]],
        confidence_threshold: float,
        risk_profile: str,
    ) -> Dict[str, Any]:
        """
        Generate a trading signal based on analysis contexts.

        Args:
            pair: Currency pair
            timeframe: Timeframe
            analysis_contexts: List of analysis contexts
            confidence_threshold: Minimum confidence threshold
            risk_profile: Risk profile

        Returns:
            Generated trading signal
        """
        # Extract directions and confidences from analyses
        directions = {"bullish": 0, "bearish": 0, "neutral": 0}
        confidences = []
        technical_indicators = {}
        patterns = []
        support_levels = []
        resistance_levels = []
        fundamental_factors = []
        sentiment_factors = []

        # Process each analysis context
        for ctx in analysis_contexts:
            findings = ctx.get("findings", {})
            analysis_type = ctx.get("analysis_type", "unknown")

            # Extract direction
            direction = findings.get("overall_direction", "neutral")
            if direction in directions:
                directions[direction] += 1

            # Collect confidence
            confidence = ctx.get("confidence", 0.5)
            confidences.append(confidence)

            # Collect technical indicators
            if analysis_type == "technical":
                indicators = findings.get("indicators", {})
                for indicator_name, indicator_data in indicators.items():
                    technical_indicators[indicator_name] = indicator_data

                # Collect patterns
                ctx_patterns = findings.get("patterns", [])
                patterns.extend(ctx_patterns)

                # Collect support/resistance levels
                supports = findings.get("support_levels", [])
                resistances = findings.get("resistance_levels", [])

                support_levels.extend(supports)
                resistance_levels.extend(resistances)

            # Collect fundamental factors
            elif analysis_type == "fundamental":
                factors = findings.get("key_factors", [])
                fundamental_factors.extend(factors)

            # Collect sentiment factors
            elif analysis_type == "sentiment":
                factors = findings.get("key_factors", [])
                sentiment_factors.extend(factors)

        # Determine overall direction
        total_analyses = sum(directions.values())

        if total_analyses == 0:
            return {
                "signal_type": "no_signal",
                "direction": "neutral",
                "confidence": 0,
                "reason": "No valid analysis data available",
            }

        # Calculate direction percentages
        direction_percentages = {
            direction: count / total_analyses for direction, count in directions.items()
        }

        # Calculate average confidence
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5

        # Determine signal direction
        signal_direction = "neutral"

        # Adjust threshold based on risk profile
        adjusted_threshold = confidence_threshold

        if risk_profile == "conservative":
            # Conservative requires stronger consensus
            adjusted_threshold = confidence_threshold + 0.1
        elif risk_profile == "aggressive":
            # Aggressive allows weaker consensus
            adjusted_threshold = confidence_threshold - 0.1

        if direction_percentages["bullish"] >= adjusted_threshold:
            signal_direction = "bullish"
        elif direction_percentages["bearish"] >= adjusted_threshold:
            signal_direction = "bearish"

        # Determine confidence based on consensus and average confidence
        signal_confidence = direction_percentages[signal_direction] * avg_confidence

        # Generate signal strength based on confidence and risk profile
        signal_strength = self._calculate_signal_strength(
            signal_confidence, direction_percentages[signal_direction], risk_profile
        )

        # Determine entry, target and stop loss prices
        price_levels = self._determine_price_levels(
            signal_direction,
            technical_indicators,
            support_levels,
            resistance_levels,
            risk_profile,
        )

        # Generate trading signal
        trading_signal = {
            "signal_type": "entry" if signal_direction != "neutral" else "no_signal",
            "direction": signal_direction,
            "confidence": signal_confidence,
            "strength": signal_strength,
            "entry_price": price_levels.get("entry"),
            "target_price": price_levels.get("target"),
            "stop_loss": price_levels.get("stop_loss"),
            "risk_reward_ratio": price_levels.get("risk_reward"),
            "timeframe": timeframe,
            "generated_at": datetime.now().isoformat(),
            "expiry": self._calculate_expiry(timeframe),
            "supporting_factors": {
                "technical_indicators": [
                    {
                        "name": name,
                        "value": data.get("value"),
                        "interpretation": data.get("interpretation"),
                    }
                    for name, data in technical_indicators.items()
                    if data.get("interpretation", "").lower() == signal_direction
                ],
                "patterns": [
                    pattern
                    for pattern in patterns
                    if pattern.get("direction", "").lower() == signal_direction
                ],
                "fundamental_factors": fundamental_factors,
                "sentiment_factors": sentiment_factors,
            },
            "opposing_factors": {
                "technical_indicators": [
                    {
                        "name": name,
                        "value": data.get("value"),
                        "interpretation": data.get("interpretation"),
                    }
                    for name, data in technical_indicators.items()
                    if data.get("interpretation", "").lower() != signal_direction
                    and data.get("interpretation", "").lower() != "neutral"
                ],
                "patterns": [
                    pattern
                    for pattern in patterns
                    if pattern.get("direction", "").lower() != signal_direction
                    and pattern.get("direction", "").lower() != "neutral"
                ],
            },
            "analysis_summary": {
                "total_analyses": total_analyses,
                "direction_distribution": direction_percentages,
                "bullish_count": directions["bullish"],
                "bearish_count": directions["bearish"],
                "neutral_count": directions["neutral"],
                "avg_confidence": avg_confidence,
            },
        }

        return trading_signal

    def _calculate_signal_strength(
        self, confidence: float, consensus: float, risk_profile: str
    ) -> str:
        """
        Calculate signal strength based on confidence and consensus.

        Args:
            confidence: Signal confidence
            consensus: Direction consensus
            risk_profile: Risk profile

        Returns:
            Signal strength ("weak", "moderate", "strong")
        """
        # Base score from confidence and consensus
        strength_score = (confidence * 0.7) + (consensus * 0.3)

        # Adjust based on risk profile
        if risk_profile == "conservative":
            # Conservative requires higher scores for each category
            if strength_score >= 0.8:
                return "strong"
            elif strength_score >= 0.65:
                return "moderate"
            else:
                return "weak"
        elif risk_profile == "aggressive":
            # Aggressive has lower thresholds
            if strength_score >= 0.6:
                return "strong"
            elif strength_score >= 0.45:
                return "moderate"
            else:
                return "weak"
        else:
            # Moderate (default)
            if strength_score >= 0.7:
                return "strong"
            elif strength_score >= 0.55:
                return "moderate"
            else:
                return "weak"

    def _determine_price_levels(
        self,
        direction: str,
        indicators: Dict[str, Any],
        support_levels: List[float],
        resistance_levels: List[float],
        risk_profile: str,
    ) -> Dict[str, Any]:
        """
        Determine entry, target and stop loss price levels.

        Args:
            direction: Signal direction
            indicators: Technical indicators
            support_levels: Support levels
            resistance_levels: Resistance levels
            risk_profile: Risk profile

        Returns:
            Dictionary with entry, target, and stop loss prices
        """
        # Mock implementation - in a real system this would use actual price data
        # and support/resistance levels to calculate appropriate entry, target and stop
        entry_price = 1.1000  # Mock price

        # Set target and stop loss based on direction and risk profile
        if direction == "bullish":
            target_price = entry_price * 1.02  # 2% target

            if risk_profile == "conservative":
                stop_loss = entry_price * 0.995  # 0.5% stop
            elif risk_profile == "aggressive":
                stop_loss = entry_price * 0.99  # 1% stop
            else:
                stop_loss = entry_price * 0.993  # 0.7% stop
        elif direction == "bearish":
            target_price = entry_price * 0.98  # 2% target

            if risk_profile == "conservative":
                stop_loss = entry_price * 1.005  # 0.5% stop
            elif risk_profile == "aggressive":
                stop_loss = entry_price * 1.01  # 1% stop
            else:
                stop_loss = entry_price * 1.007  # 0.7% stop
        else:
            return {
                "entry": None,
                "target": None,
                "stop_loss": None,
                "risk_reward": None,
            }

        # Calculate risk/reward ratio
        risk = abs(entry_price - stop_loss)
        reward = abs(entry_price - target_price)
        risk_reward = reward / risk if risk > 0 else 0

        return {
            "entry": entry_price,
            "target": target_price,
            "stop_loss": stop_loss,
            "risk_reward": risk_reward,
        }

    def _calculate_expiry(self, timeframe: str) -> str:
        """
        Calculate signal expiry date based on timeframe.

        Args:
            timeframe: Analysis timeframe

        Returns:
            Expiry date as ISO string
        """
        now = datetime.now()

        if timeframe in ["1m", "5m", "15m"]:
            expiry = now + timedelta(hours=4)
        elif timeframe in ["30m", "1h"]:
            expiry = now + timedelta(hours=24)
        elif timeframe in ["4h", "8h"]:
            expiry = now + timedelta(days=3)
        elif timeframe == "daily":
            expiry = now + timedelta(days=7)
        elif timeframe == "weekly":
            expiry = now + timedelta(days=30)
        elif timeframe == "monthly":
            expiry = now + timedelta(days=90)
        else:
            # Default to 3 days
            expiry = now + timedelta(days=3)

        return expiry.isoformat()


class StrategySignalTool(Tool):
    """Tool for generating trading signals based on pre-defined strategies."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the strategy signal tool.

        Args:
            config: Optional configuration dictionary
        """
        super().__init__()
        self.config = config or {}
        self.memory = EnhancedMemoryManager(self.config.get("memory_config"))

        # Define tool metadata
        self.description = ToolDescription(
            name="strategy_signal",
            description="Generates trading signals based on pre-defined strategies",
            version="1.0.0",
        )

        # Define parameters
        self.parameters = [
            ToolParameter(
                name="pair",
                description="Currency pair (e.g., 'EUR/USD')",
                required=True,
                type="string",
            ),
            ToolParameter(
                name="timeframe",
                description="Timeframe for signal (e.g., '1h', '4h', 'daily')",
                required=True,
                type="string",
            ),
            ToolParameter(
                name="strategy_id",
                description="ID of the strategy to use",
                required=True,
                type="string",
            ),
            ToolParameter(
                name="strategy_params",
                description="Additional parameters for the strategy",
                required=False,
                type="object",
                default={},
            ),
        ]

    async def execute(self, context: ToolContext) -> ToolResult:
        """
        Execute strategy-based signal generation.

        Args:
            context: Tool execution context with parameters

        Returns:
            Generated trading signal
        """
        # Extract parameters
        pair = context.get_parameter("pair")
        timeframe = context.get_parameter("timeframe")
        strategy_id = context.get_parameter("strategy_id")
        strategy_params = context.get_parameter("strategy_params", {})

        try:
            logger.info(f"Generating signal for {pair} using strategy {strategy_id}")

            # Apply the strategy to generate a signal
            # In a real implementation, this would:
            # 1. Retrieve the strategy definition
            # 2. Apply the strategy to current market data
            # 3. Generate a signal according to the strategy rules

            # Mock strategy application
            signal = await self._apply_strategy(
                pair=pair,
                timeframe=timeframe,
                strategy_id=strategy_id,
                strategy_params=strategy_params,
            )

            # Store the signal in memory
            signal_id = str(uuid.uuid4())

            signal_context = {
                "context_id": signal_id,
                "pair": pair,
                "timeframe": timeframe,
                "analysis_type": "strategy_signal",
                "timestamp": datetime.now().isoformat(),
                "findings": signal,
                "confidence": signal.get("confidence", 0),
                "tags": [
                    "signal",
                    "strategy",
                    strategy_id,
                    signal.get("signal_type", ""),
                ],
                "metadata": {
                    "strategy_id": strategy_id,
                    "strategy_params": strategy_params,
                },
            }

            # Store in memory
            await self.memory._store_context(signal_context)

            # Prepare response
            return ToolResult(
                success=True,
                result={
                    "signal_id": signal_id,
                    "pair": pair,
                    "timeframe": timeframe,
                    "strategy_id": strategy_id,
                    "signal": signal,
                    "generated_at": datetime.now().isoformat(),
                },
            )

        except Exception as e:
            logger.error(f"Error generating strategy signal: {str(e)}")

            return ToolResult(
                success=False, error=f"Error generating strategy signal: {str(e)}"
            )

    async def _apply_strategy(
        self,
        pair: str,
        timeframe: str,
        strategy_id: str,
        strategy_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Apply a strategy to generate a signal.

        Args:
            pair: Currency pair
            timeframe: Timeframe
            strategy_id: Strategy ID
            strategy_params: Strategy parameters

        Returns:
            Generated signal
        """
        # Mock strategy signals - in a real implementation, this would use actual strategy definitions
        # and apply them to market data

        # Predefined mock strategies
        mock_strategies = {
            "trend_following": {
                "signal_type": "entry",
                "direction": "bullish",
                "confidence": 0.75,
                "strength": "moderate",
                "entry_price": 1.1020,
                "target_price": 1.1080,
                "stop_loss": 1.0980,
                "risk_reward_ratio": 1.5,
                "description": "Trend following strategy based on moving averages alignment",
            },
            "breakout": {
                "signal_type": "entry",
                "direction": "bullish",
                "confidence": 0.82,
                "strength": "strong",
                "entry_price": 1.1040,
                "target_price": 1.1120,
                "stop_loss": 1.0990,
                "risk_reward_ratio": 1.6,
                "description": "Breakout strategy based on resistance level breach with volume confirmation",
            },
            "mean_reversion": {
                "signal_type": "entry",
                "direction": "bearish",
                "confidence": 0.71,
                "strength": "moderate",
                "entry_price": 1.1060,
                "target_price": 1.0990,
                "stop_loss": 1.1100,
                "risk_reward_ratio": 1.75,
                "description": "Mean reversion strategy based on overbought conditions and divergence",
            },
            "support_resistance": {
                "signal_type": "entry",
                "direction": "bullish",
                "confidence": 0.68,
                "strength": "moderate",
                "entry_price": 1.1010,
                "target_price": 1.1060,
                "stop_loss": 1.0980,
                "risk_reward_ratio": 1.67,
                "description": "Support-resistance strategy based on bounce from key support level",
            },
        }

        # Return a strategy signal if found
        if strategy_id in mock_strategies:
            signal = mock_strategies[strategy_id].copy()

            # Add standard fields
            signal.update(
                {
                    "timeframe": timeframe,
                    "generated_at": datetime.now().isoformat(),
                    "expiry": self._calculate_expiry(timeframe),
                    "strategy_id": strategy_id,
                    "strategy_params": strategy_params,
                }
            )

            return signal
        else:
            return {
                "signal_type": "no_signal",
                "direction": "neutral",
                "confidence": 0,
                "reason": f"Strategy with ID '{strategy_id}' not found",
            }

    def _calculate_expiry(self, timeframe: str) -> str:
        """
        Calculate signal expiry date based on timeframe.

        Args:
            timeframe: Analysis timeframe

        Returns:
            Expiry date as ISO string
        """
        now = datetime.now()

        if timeframe in ["1m", "5m", "15m"]:
            expiry = now + timedelta(hours=4)
        elif timeframe in ["30m", "1h"]:
            expiry = now + timedelta(hours=24)
        elif timeframe in ["4h", "8h"]:
            expiry = now + timedelta(days=3)
        elif timeframe == "daily":
            expiry = now + timedelta(days=7)
        elif timeframe == "weekly":
            expiry = now + timedelta(days=30)
        elif timeframe == "monthly":
            expiry = now + timedelta(days=90)
        else:
            # Default to 3 days
            expiry = now + timedelta(days=3)

        return expiry.isoformat()


def get_signal_tools(config: Optional[Dict[str, Any]] = None) -> List[Tool]:
    """
    Get all signal generation tools.

    Args:
        config: Optional configuration dictionary

    Returns:
        List of signal generation tools
    """
    return [SignalGeneratorTool(config), StrategySignalTool(config)]

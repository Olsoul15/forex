"""
Market State Integration with AutoAgentOrchestrator

This module provides integration between the MarketStateDetector and AutoAgentOrchestrator,
enhancing the orchestrator's market analysis capabilities with market state awareness.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import asyncio

# Import from the new features module
from forex_ai.features.market_states import (
    MarketState,
    MarketStateAnalyzer,
    get_market_state,
    VolatilityRegime,
)

try:
    from forex_ai.integration.autoagent_orchestrator import AutoAgentOrchestrator
except ImportError:
    # Minimal implementation for testing
    class AutoAgentOrchestrator:
        """Mock AutoAgentOrchestrator for testing."""

        def __init__(self, config=None):
            self.config = config or {}


logger = logging.getLogger(__name__)


class MarketStateAutoAgentIntegrator:
    """
    Integrates market state detection with the AutoAgentOrchestrator.

    This class enhances the AutoAgentOrchestrator with market state awareness,
    providing additional context for market analysis and decision-making.
    """

    def __init__(self, orchestrator: AutoAgentOrchestrator):
        """
        Initialize the integrator.

        Args:
            orchestrator: AutoAgentOrchestrator instance to enhance
        """
        self.orchestrator = orchestrator
        self.state_detector = MarketStateAnalyzer()

        # Cache of recent market states
        self.market_state_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_duration = 300  # Cache duration in seconds

        # Apply monkey patching to enhance the orchestrator
        self._enhance_orchestrator()

        logger.info("Market State AutoAgent Integrator initialized")

    def _enhance_orchestrator(self):
        """
        Enhance the AutoAgentOrchestrator with market state capabilities.

        This method monkey patches the orchestrator to add market state awareness
        to its analyze_market method.
        """
        # Store original method
        original_analyze_market = self.orchestrator.analyze_market

        # Define the enhanced method
        async def enhanced_analyze_market(instrument, timeframe=None, *args, **kwargs):
            """Enhanced analyze_market method with market state detection."""
            # First, call the original method
            result = await original_analyze_market(
                instrument, timeframe, *args, **kwargs
            )

            # Check if the analysis was successful
            if not result.get("success", False):
                return result

            # If we have a market view, enhance it with market state information
            if "market_view" in result:
                # Get the market state for this instrument and timeframe
                market_state_info = await self.analyze_market_state(
                    instrument, timeframe, result.get("market_view", {})
                )

                # Add market state to the result
                if market_state_info.get("success", False):
                    market_state = market_state_info.get("market_state", {})
                    implications = market_state_info.get("trading_implications", {})

                    # Add to the market view
                    result["market_view"]["market_state"] = market_state
                    result["market_view"]["market_state_summary"] = (
                        market_state_info.get("market_state_summary", "")
                    )

                    # Add market state bias to overall direction if confidence is high enough
                    if market_state.get("confidence", 0) > 0.7:
                        state_type = market_state.get("state_type", "")
                        if "UPTREND" in state_type or "BREAKOUT_UP" in state_type:
                            # Boost bullish bias
                            result["market_view"]["overall_direction"] = "bullish"
                            result["market_view"]["confidence"] = max(
                                result["market_view"].get("confidence", 0.5),
                                market_state.get("confidence", 0.5),
                            )
                        elif "DOWNTREND" in state_type or "BREAKOUT_DOWN" in state_type:
                            # Boost bearish bias
                            result["market_view"]["overall_direction"] = "bearish"
                            result["market_view"]["confidence"] = max(
                                result["market_view"].get("confidence", 0.5),
                                market_state.get("confidence", 0.5),
                            )

                    # Add trading implications
                    if "implications" not in result["market_view"]:
                        result["market_view"]["implications"] = {}

                    result["market_view"]["implications"]["market_state"] = implications

            return result

        # Replace the original method with the enhanced one
        self.orchestrator.analyze_market = enhanced_analyze_market

        # Also enhance other methods if needed
        if hasattr(self.orchestrator, "get_market_context"):
            original_get_context = self.orchestrator.get_market_context

            async def enhanced_get_context(*args, **kwargs):
                """Enhanced get_market_context method with market state history."""
                result = await original_get_context(*args, **kwargs)

                # Add market state history if available
                if result.get("success", False) and "contexts" in result:
                    # Add market state information to each context
                    for context in result["contexts"]:
                        if (
                            "market_state" not in context
                            and "pair" in context
                            and "timeframe" in context
                        ):
                            # Try to find market state from cache
                            cache_key = f"{context['pair']}_{context['timeframe']}"
                            if cache_key in self.market_state_cache:
                                context["market_state"] = self.market_state_cache[
                                    cache_key
                                ]["market_state"]

                return result

            self.orchestrator.get_market_context = enhanced_get_context

        logger.info("Enhanced AutoAgentOrchestrator with market state capabilities")

    async def analyze_market_state(
        self, pair: str, timeframe: Optional[str], market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze the market state for a specific pair and timeframe.

        Args:
            pair: Currency pair to analyze
            timeframe: Timeframe to analyze (if None, will try to extract from market_data)
            market_data: Market data containing features and indicators

        Returns:
            Dictionary containing market state analysis results
        """
        if not timeframe and "timeframe" in market_data:
            timeframe = market_data["timeframe"]

        if not timeframe:
            timeframe = "H1"  # Default to H1 if not specified

        # Check cache first
        cache_key = f"{pair}_{timeframe}"
        if cache_key in self.market_state_cache:
            cache_entry = self.market_state_cache[cache_key]
            cache_age = asyncio.get_event_loop().time() - cache_entry["timestamp"]

            if cache_age < self.cache_duration:
                logger.debug(f"Using cached market state for {pair}/{timeframe}")
                return cache_entry["result"]

        # Extract technical features from market data
        features = self._extract_features_from_market_data(market_data)

        # Ensure we have minimal required features
        if not self._validate_features(features):
            logger.warning(
                f"Missing required features for market state detection: {pair}/{timeframe}"
            )
            return {
                "success": False,
                "error": "Missing required features for market state detection",
            }

        # Detect market state
        try:
            market_state = self.state_detector.detect_market_state(
                pair, timeframe, features
            )

            # Generate human-readable summary
            summary = self._generate_state_summary(market_state)

            # Generate trading implications
            implications = self._generate_trading_implications(market_state)

            # Prepare result
            result = {
                "success": True,
                "market_state": market_state.to_dict(),
                "market_state_summary": summary,
                "trading_implications": implications,
            }

            # Cache the result
            self.market_state_cache[cache_key] = {
                "timestamp": asyncio.get_event_loop().time(),
                "result": result,
            }

            logger.info(
                f"Market state for {pair}/{timeframe}: {market_state.state_type.name} "
                f"with confidence {market_state.confidence:.2f}"
            )

            return result

        except Exception as e:
            logger.error(f"Error detecting market state for {pair}/{timeframe}: {e}")
            return {
                "success": False,
                "error": f"Error detecting market state: {str(e)}",
            }

    def _extract_features_from_market_data(
        self, market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract technical features from market data.

        Args:
            market_data: Market data from AutoAgentOrchestrator

        Returns:
            Dictionary of features suitable for market state detection
        """
        features = {}

        # Extract OHLCV data if available
        if "candles" in market_data:
            candles = market_data["candles"]
            features.update(
                {
                    "open": (
                        [c["open"] for c in candles]
                        if isinstance(candles, list)
                        else []
                    ),
                    "high": (
                        [c["high"] for c in candles]
                        if isinstance(candles, list)
                        else []
                    ),
                    "low": (
                        [c["low"] for c in candles] if isinstance(candles, list) else []
                    ),
                    "close": (
                        [c["close"] for c in candles]
                        if isinstance(candles, list)
                        else []
                    ),
                    "volume": (
                        [c["volume"] for c in candles]
                        if isinstance(candles, list)
                        else []
                    ),
                }
            )

        # Extract technical indicators if available
        if "technical" in market_data and isinstance(market_data["technical"], dict):
            tech = market_data["technical"]

            # Extract indicators
            indicators = tech.get("indicators", {})

            # Map indicator names to expected feature names
            if "adx" in indicators:
                features["adx"] = (
                    indicators["adx"]
                    if isinstance(indicators["adx"], list)
                    else [indicators["adx"]]
                )

            if "ema" in indicators:
                if (
                    isinstance(indicators["ema"], dict)
                    and "fast" in indicators["ema"]
                    and "slow" in indicators["ema"]
                ):
                    features["ema_fast"] = (
                        indicators["ema"]["fast"]
                        if isinstance(indicators["ema"]["fast"], list)
                        else [indicators["ema"]["fast"]]
                    )
                    features["ema_slow"] = (
                        indicators["ema"]["slow"]
                        if isinstance(indicators["ema"]["slow"], list)
                        else [indicators["ema"]["slow"]]
                    )

            if "bollinger_bands" in indicators:
                if isinstance(indicators["bollinger_bands"], dict):
                    features["bollinger_bands"] = indicators["bollinger_bands"]

            if "atr" in indicators:
                features["atr"] = (
                    indicators["atr"]
                    if isinstance(indicators["atr"], list)
                    else [indicators["atr"]]
                )

            if "rsi" in indicators:
                features["rsi"] = (
                    indicators["rsi"]
                    if isinstance(indicators["rsi"], list)
                    else [indicators["rsi"]]
                )

        return features

    def _validate_features(self, features: Dict[str, Any]) -> bool:
        """
        Validate that the features dictionary contains required features.

        Args:
            features: Features dictionary

        Returns:
            True if required features are present, False otherwise
        """
        # Minimal required features
        required_keys = ["close", "high", "low"]

        # At least one of these indicator groups should be present
        indicator_groups = [
            ["adx"],  # For trend detection
            ["bollinger_bands"],  # For range detection
            ["atr"],  # For volatility detection
            ["rsi"],  # For reversal detection
        ]

        # Check basic price data
        if not all(key in features for key in required_keys):
            return False

        # Check that at least one indicator from each group is present
        indicator_count = 0
        for group in indicator_groups:
            if any(indicator in features for indicator in group):
                indicator_count += 1

        # Require at least 2 indicator groups for basic functionality
        return indicator_count >= 2

    def _generate_state_summary(self, market_state: MarketState) -> str:
        """
        Generate a human-readable summary of the market state.

        Args:
            market_state: Detected market state

        Returns:
            String summary of the market state
        """
        confidence_str = (
            "high"
            if market_state.confidence > 0.8
            else "medium" if market_state.confidence > 0.6 else "low"
        )

        duration_str = (
            "persistent"
            if market_state.duration > 5
            else "established" if market_state.duration > 2 else "recent"
        )

        state_type = market_state.state_type

        # Format summary based on state type
        if state_type == MarketStateType.STRONG_UPTREND:
            return f"Strong uptrend with {confidence_str} confidence, {duration_str} duration"
        elif state_type == MarketStateType.WEAK_UPTREND:
            return f"Weak uptrend with {confidence_str} confidence, {duration_str} duration"
        elif state_type == MarketStateType.STRONG_DOWNTREND:
            return f"Strong downtrend with {confidence_str} confidence, {duration_str} duration"
        elif state_type == MarketStateType.WEAK_DOWNTREND:
            return f"Weak downtrend with {confidence_str} confidence, {duration_str} duration"
        elif state_type == MarketStateType.RANGING:
            return f"Ranging market with {confidence_str} confidence, {duration_str} duration"
        elif state_type == MarketStateType.CONSOLIDATION:
            return f"Consolidation pattern with {confidence_str} confidence, {duration_str} duration"
        elif state_type == MarketStateType.VOLATILITY_EXPANSION:
            return f"Expanding volatility with {confidence_str} confidence, {duration_str} phase"
        elif state_type == MarketStateType.VOLATILITY_CONTRACTION:
            return f"Contracting volatility with {confidence_str} confidence, {duration_str} phase"
        elif state_type == MarketStateType.POTENTIAL_REVERSAL_UP:
            return f"Potential upward reversal with {confidence_str} confidence"
        elif state_type == MarketStateType.POTENTIAL_REVERSAL_DOWN:
            return f"Potential downward reversal with {confidence_str} confidence"
        elif state_type == MarketStateType.BREAKOUT_UP:
            return f"Upward breakout with {confidence_str} confidence"
        elif state_type == MarketStateType.BREAKOUT_DOWN:
            return f"Downward breakout with {confidence_str} confidence"
        else:
            return f"Undefined market state, insufficient data or confidence"

    def _generate_trading_implications(
        self, market_state: MarketState
    ) -> Dict[str, Any]:
        """
        Generate trading implications based on the detected market state.

        Args:
            market_state: Detected market state

        Returns:
            Dictionary containing trading implications
        """
        state_type = market_state.state_type
        confidence = market_state.confidence

        # Default implications
        implications = {
            "bias": "neutral",
            "position_sizing": "normal",
            "stop_loss": "normal",
            "take_profit": "normal",
            "preferred_strategies": [],
            "risk_level": "medium",
        }

        # Adjust implications based on market state
        if state_type in [MarketStateType.STRONG_UPTREND, MarketStateType.BREAKOUT_UP]:
            implications["bias"] = "bullish"
            implications["preferred_strategies"] = ["trend_following", "breakout"]
            implications["take_profit"] = "wide"

        elif state_type in [
            MarketStateType.STRONG_DOWNTREND,
            MarketStateType.BREAKOUT_DOWN,
        ]:
            implications["bias"] = "bearish"
            implications["preferred_strategies"] = ["trend_following", "breakout"]
            implications["take_profit"] = "wide"

        elif state_type == MarketStateType.WEAK_UPTREND:
            implications["bias"] = "slightly_bullish"
            implications["preferred_strategies"] = ["trend_following", "pullback"]
            implications["stop_loss"] = "tight"

        elif state_type == MarketStateType.WEAK_DOWNTREND:
            implications["bias"] = "slightly_bearish"
            implications["preferred_strategies"] = ["trend_following", "pullback"]
            implications["stop_loss"] = "tight"

        elif state_type in [MarketStateType.RANGING, MarketStateType.CONSOLIDATION]:
            implications["preferred_strategies"] = ["range_trading", "mean_reversion"]
            implications["take_profit"] = "tight"
            implications["position_sizing"] = "reduced"

        elif state_type == MarketStateType.VOLATILITY_EXPANSION:
            implications["risk_level"] = "high"
            implications["stop_loss"] = "wide"
            implications["position_sizing"] = "reduced"

        elif state_type == MarketStateType.VOLATILITY_CONTRACTION:
            implications["risk_level"] = "low"
            implications["preferred_strategies"] = ["breakout_anticipation"]
            implications["position_sizing"] = "reduced"

        elif state_type == MarketStateType.POTENTIAL_REVERSAL_UP:
            implications["bias"] = "potentially_bullish"
            implications["preferred_strategies"] = ["reversal", "counter_trend"]
            implications["stop_loss"] = "tight"
            implications["position_sizing"] = "reduced"

        elif state_type == MarketStateType.POTENTIAL_REVERSAL_DOWN:
            implications["bias"] = "potentially_bearish"
            implications["preferred_strategies"] = ["reversal", "counter_trend"]
            implications["stop_loss"] = "tight"
            implications["position_sizing"] = "reduced"

        # Adjust based on confidence
        if confidence < 0.5:
            implications["position_sizing"] = "minimal"
            implications["risk_level"] = "high"
        elif confidence > 0.8:
            if "reduced" not in implications["position_sizing"]:
                implications["position_sizing"] = "increased"
            implications["risk_level"] = "low"

        return implications


# Function to enhance an existing AutoAgentOrchestrator with market state capabilities
def enhance_autoagent_with_market_state(
    orchestrator: AutoAgentOrchestrator,
) -> MarketStateAutoAgentIntegrator:
    """
    Enhance an existing AutoAgentOrchestrator with market state capabilities.

    Args:
        orchestrator: AutoAgentOrchestrator instance to enhance

    Returns:
        MarketStateAutoAgentIntegrator instance
    """
    return MarketStateAutoAgentIntegrator(orchestrator)

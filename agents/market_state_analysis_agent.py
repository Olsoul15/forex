"""
Market State Analysis Agent

This agent is responsible for analyzing market conditions using the MarketStateDetector
to identify current market states, their confidence levels, and provide context for
decision-making by other components of the system.
"""

import logging
from typing import Dict, List, Any, Optional, Union
import asyncio

from forex_ai.agents.base import BaseAgent
from forex_ai.custom_types import AgentConfig, MarketData, SignalStrength

# Import the MarketStateDetector
try:
    # First try importing from auto_agent path (if running in that context)
    from auto_agent.app.forex_ai.features.market_states import (
        MarketStateDetector,
        MarketStateType,
        MarketState,
        get_market_state_detector,
    )
except ImportError:
    # Fall back to regular forex_ai path
    try:
        from forex_ai.features.market_states import (
            MarketStateDetector,
            MarketStateType,
            MarketState,
            get_market_state_detector,
        )
    except ImportError:
        # Define minimal versions for standalone operation if neither import works
        from enum import Enum, auto
        from dataclasses import dataclass

        class MarketStateType(Enum):
            """Enum representing different market state types."""

            STRONG_UPTREND = auto()
            WEAK_UPTREND = auto()
            STRONG_DOWNTREND = auto()
            WEAK_DOWNTREND = auto()
            RANGING = auto()
            CONSOLIDATION = auto()
            VOLATILITY_EXPANSION = auto()
            VOLATILITY_CONTRACTION = auto()
            POTENTIAL_REVERSAL_UP = auto()
            POTENTIAL_REVERSAL_DOWN = auto()
            BREAKOUT_UP = auto()
            BREAKOUT_DOWN = auto()
            UNDEFINED = auto()

        @dataclass
        class MarketState:
            """Class representing a detected market state."""

            state_type: MarketStateType
            confidence: float
            context: Dict[str, Any]
            duration: int

            def to_dict(self):
                return {
                    "state_type": self.state_type.name,
                    "confidence": self.confidence,
                    "duration": self.duration,
                    "context": self.context,
                }

            @property
            def is_trending(self) -> bool:
                """Check if the state is a trending state."""
                trending_states = [
                    MarketStateType.STRONG_UPTREND,
                    MarketStateType.WEAK_UPTREND,
                    MarketStateType.STRONG_DOWNTREND,
                    MarketStateType.WEAK_DOWNTREND,
                ]
                return self.state_type in trending_states

            @property
            def is_ranging(self) -> bool:
                """Check if the state is a ranging state."""
                ranging_states = [
                    MarketStateType.RANGING,
                    MarketStateType.CONSOLIDATION,
                ]
                return self.state_type in ranging_states

            @property
            def is_reversal(self) -> bool:
                """Check if the state is a reversal state."""
                reversal_states = [
                    MarketStateType.POTENTIAL_REVERSAL_UP,
                    MarketStateType.POTENTIAL_REVERSAL_DOWN,
                ]
                return self.state_type in reversal_states

            @property
            def is_breakout(self) -> bool:
                """Check if the state is a breakout state."""
                breakout_states = [
                    MarketStateType.BREAKOUT_UP,
                    MarketStateType.BREAKOUT_DOWN,
                ]
                return self.state_type in breakout_states

        class MarketStateDetector:
            """Minimal implementation of MarketStateDetector."""

            def detect_market_state(self, pair, timeframe, features):
                return MarketState(
                    state_type=MarketStateType.UNDEFINED,
                    confidence=0.0,
                    context={},
                    duration=0,
                )

        def get_market_state_detector():
            return MarketStateDetector()


logger = logging.getLogger(__name__)


class MarketStateAnalysisAgent(BaseAgent):
    """
    Agent for analyzing market states and providing context to other components.

    This agent leverages the MarketStateDetector to identify the current market state,
    including trend status, ranging conditions, volatility regimes, and potential
    reversal or breakout patterns. It provides this information to other agents
    and components to inform their decision-making.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        """
        Initialize the Market State Analysis Agent.

        Args:
            config: Optional configuration for the agent
        """
        super().__init__(config or AgentConfig(name="MarketStateAnalysisAgent"))

        # Initialize the market state detector
        self.state_detector = get_market_state_detector()

        # Cache for recently detected states to avoid redundant calculations
        self.state_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_duration = 300  # Cache duration in seconds

        # Default indicators required for state detection
        self.required_indicators = [
            {"name": "adx", "params": {"period": 14}},
            {"name": "ema", "params": {"period": 20, "price_type": "close"}},
            {"name": "ema", "params": {"period": 50, "price_type": "close"}},
            {"name": "bollinger_bands", "params": {"period": 20, "std_dev": 2.0}},
            {"name": "atr", "params": {"period": 14}},
            {"name": "rsi", "params": {"period": 14}},
        ]

        logger.info(f"Market State Analysis Agent initialized")

    async def analyze_market_state(
        self, pair: str, timeframe: str, features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze the current market state for a given pair and timeframe.

        Args:
            pair: Currency pair (e.g., 'EUR_USD')
            timeframe: Timeframe (e.g., 'H1')
            features: Dictionary containing calculated technical indicators

        Returns:
            Dictionary containing market state analysis results
        """
        logger.debug(f"Analyzing market state for {pair}/{timeframe}")

        # Check cache first to avoid redundant calculations
        cache_key = f"{pair}_{timeframe}"
        if cache_key in self.state_cache:
            cache_entry = self.state_cache[cache_key]
            cache_age = asyncio.get_event_loop().time() - cache_entry["timestamp"]

            if cache_age < self.cache_duration:
                logger.debug(f"Using cached market state for {pair}/{timeframe}")
                return cache_entry["result"]

        # Ensure we have all required features
        if not self._validate_features(features):
            logger.warning(
                f"Missing required features for market state detection: {pair}/{timeframe}"
            )
            return {
                "success": False,
                "error": "Missing required features for market state detection",
                "market_state": {
                    "state_type": MarketStateType.UNDEFINED.name,
                    "confidence": 0.0,
                    "duration": 0,
                },
            }

        # Detect market state
        market_state = self.state_detector.detect_market_state(
            pair, timeframe, features
        )

        # Prepare result
        result = {
            "success": True,
            "market_state": market_state.to_dict(),
            "market_state_summary": self._generate_state_summary(market_state),
            "trading_implications": self._generate_trading_implications(market_state),
        }

        # Cache the result
        self.state_cache[cache_key] = {
            "timestamp": asyncio.get_event_loop().time(),
            "result": result,
        }

        logger.info(
            f"Market state for {pair}/{timeframe}: {market_state.state_type.name} "
            f"with confidence {market_state.confidence:.2f}"
        )

        return result

    def _validate_features(self, features: Dict[str, Any]) -> bool:
        """
        Validate that all required features for market state detection are present.

        Args:
            features: Dictionary of calculated features

        Returns:
            True if all required features are present, False otherwise
        """
        # Minimal required features for market state detection
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
        for group in indicator_groups:
            if not any(indicator in features for indicator in group):
                logger.debug(f"Missing indicators from group: {group}")
                return False

        return True

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

    async def execute_task(
        self, task_type: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a task assigned to this agent.

        Args:
            task_type: Type of task to execute
            params: Parameters for the task

        Returns:
            Dictionary containing task execution results
        """
        if task_type == "analyze_market_state":
            required_params = ["pair", "timeframe", "features"]
            if not all(param in params for param in required_params):
                logger.error(f"Missing required parameters for market state analysis")
                return {
                    "success": False,
                    "error": f"Missing required parameters: {required_params}",
                }

            return await self.analyze_market_state(
                pair=params["pair"],
                timeframe=params["timeframe"],
                features=params["features"],
            )

        elif task_type == "get_required_indicators":
            return {"success": True, "required_indicators": self.required_indicators}

        else:
            logger.warning(f"Unknown task type: {task_type}")
            return {"success": False, "error": f"Unknown task type: {task_type}"}

    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a message received by this agent.

        Args:
            message: Message to process

        Returns:
            Response to the message
        """
        if "task" in message:
            return await self.execute_task(message["task"], message.get("params", {}))
        else:
            logger.warning(f"Message does not contain task: {message}")
            return {"success": False, "error": "Message does not contain task"}


# Function to create a new MarketStateAnalysisAgent instance
def create_market_state_analysis_agent(
    config: Optional[AgentConfig] = None,
) -> MarketStateAnalysisAgent:
    """
    Create a new MarketStateAnalysisAgent instance.

    Args:
        config: Optional configuration for the agent

    Returns:
        Initialized MarketStateAnalysisAgent
    """
    return MarketStateAnalysisAgent(config)

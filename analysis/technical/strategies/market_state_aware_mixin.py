"""
Market State Aware Mixin for Trading Strategies

This module provides a mixin class that can be used to add market state awareness to
existing trading strategies, allowing them to adjust their behavior based on detected
market conditions.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple

# Try to import MarketStateDetector from different possible locations
try:
    # First try importing from auto_agent path
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
        # Define minimal versions if neither import works
        from enum import Enum, auto

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

        class MarketState:
            """Class representing a detected market state."""

            def __init__(
                self,
                state_type=MarketStateType.UNDEFINED,
                confidence=0.0,
                context=None,
                duration=0,
            ):
                self.state_type = state_type
                self.confidence = confidence
                self.context = context or {}
                self.duration = duration

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

        class MarketStateDetector:
            """Minimal implementation of MarketStateDetector."""

            def detect_market_state(self, pair, timeframe, features):
                return MarketState()

        def get_market_state_detector():
            return MarketStateDetector()


logger = logging.getLogger(__name__)


class MarketStateAwareMixin:
    """
    Mixin class to add market state awareness to trading strategies.

    This mixin can be added to any strategy class to provide market state detection
    capabilities and helper methods for adjusting strategy behavior based on the
    detected market conditions.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the mixin.

        Note: This should be called by the parent class's __init__ method.
        """
        # Initialize the market state detector
        self.state_detector = get_market_state_detector()

        # Default market state parameters
        self.market_state_params = {
            "min_confidence_threshold": 0.5,  # Min confidence to consider a state
            "trend_bias_multiplier": 1.2,  # Increase signal strength in trends
            "range_bias_multiplier": 0.8,  # Decrease signal strength in ranges
            "volatility_expansion_sl_multiplier": 1.3,  # Widen stop loss in high volatility
            "volatility_contraction_tp_multiplier": 0.8,  # Tighten take profit in low volatility
            "reversal_confidence_boost": 0.1,  # Boost confidence for reversal signals that align with detected reversals
            "breakout_filter": True,  # Whether to filter signals based on breakout state
        }

        # Override with any provided kwargs
        if "market_state_params" in kwargs:
            self.market_state_params.update(kwargs.pop("market_state_params"))

        # Cache for recent market states to avoid redundant calculations
        self.market_state_cache = {}

        # Call parent init if this is used as a mixin
        super().__init__(*args, **kwargs)

        logger.info(
            "Market State Aware Mixin initialized with parameters: %s",
            self.market_state_params,
        )

    def detect_market_state(
        self, pair: str, timeframe: str, features: Dict[str, Any]
    ) -> MarketState:
        """
        Detect the current market state for a given pair and timeframe.

        Args:
            pair: Currency pair (e.g., 'EUR_USD')
            timeframe: Timeframe (e.g., 'H1')
            features: Dictionary of calculated features

        Returns:
            MarketState object representing the detected state
        """
        # Create cache key
        cache_key = f"{pair}_{timeframe}"

        # Check if we have a cached result
        if cache_key in self.market_state_cache:
            return self.market_state_cache[cache_key]

        # Detect market state
        market_state = self.state_detector.detect_market_state(
            pair, timeframe, features
        )

        # Cache the result
        self.market_state_cache[cache_key] = market_state

        logger.debug(
            f"Detected market state for {pair}/{timeframe}: "
            f"{market_state.state_type.name} with confidence {market_state.confidence:.2f}"
        )

        return market_state

    def filter_signal(
        self, signal: Dict[str, Any], market_state: MarketState
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Filter and adjust a trading signal based on the current market state.

        Args:
            signal: Trading signal dictionary
            market_state: Detected market state

        Returns:
            Tuple of (should_keep, adjusted_signal)
        """
        # Default to keeping the signal
        should_keep = True

        # Check confidence threshold
        if (
            market_state.confidence
            < self.market_state_params["min_confidence_threshold"]
        ):
            # Low confidence in market state, don't filter but don't adjust either
            return True, signal

        # Clone the signal to avoid modifying the original
        adjusted_signal = signal.copy()

        # Extract signal direction
        signal_direction = signal.get("direction", "").lower()
        if not signal_direction:
            return True, signal

        # Check if signal aligns with market state
        signal_matches_state = self._signal_matches_market_state(
            signal_direction, market_state
        )

        # Filter signals in certain conditions
        if self.market_state_params["breakout_filter"] and market_state.state_type in [
            MarketStateType.BREAKOUT_UP,
            MarketStateType.BREAKOUT_DOWN,
        ]:
            # Only keep breakout signals that match the breakout direction
            if not signal_matches_state:
                should_keep = False

        # Adjust signal strength based on market state
        if "strength" in signal and isinstance(signal["strength"], (int, float)):
            adjusted_strength = signal["strength"]

            if market_state.is_trending and signal_matches_state:
                # Increase strength for signals that match trend direction
                adjusted_strength *= self.market_state_params["trend_bias_multiplier"]
            elif market_state.is_ranging:
                # Decrease strength for all signals in ranging markets
                adjusted_strength *= self.market_state_params["range_bias_multiplier"]

            # If it's a reversal signal and matches detected reversal state, boost confidence
            if (
                market_state.state_type
                in [
                    MarketStateType.POTENTIAL_REVERSAL_UP,
                    MarketStateType.POTENTIAL_REVERSAL_DOWN,
                ]
                and signal_matches_state
            ):
                adjusted_strength += self.market_state_params[
                    "reversal_confidence_boost"
                ]

            # Cap at 1.0
            adjusted_strength = min(1.0, adjusted_strength)
            adjusted_signal["strength"] = adjusted_strength

        # Adjust stop loss and take profit based on volatility
        if market_state.state_type == MarketStateType.VOLATILITY_EXPANSION:
            # Widen stop loss in high volatility
            if "stop_loss_pips" in signal and isinstance(
                signal["stop_loss_pips"], (int, float)
            ):
                adjusted_signal["stop_loss_pips"] = (
                    signal["stop_loss_pips"]
                    * self.market_state_params["volatility_expansion_sl_multiplier"]
                )

        elif market_state.state_type == MarketStateType.VOLATILITY_CONTRACTION:
            # Tighten take profit in low volatility
            if "take_profit_pips" in signal and isinstance(
                signal["take_profit_pips"], (int, float)
            ):
                adjusted_signal["take_profit_pips"] = (
                    signal["take_profit_pips"]
                    * self.market_state_params["volatility_contraction_tp_multiplier"]
                )

        # Add market state info to signal metadata
        if "metadata" not in adjusted_signal:
            adjusted_signal["metadata"] = {}

        adjusted_signal["metadata"]["market_state"] = market_state.to_dict()

        return should_keep, adjusted_signal

    def reset_market_state_cache(self):
        """Reset the market state cache."""
        self.market_state_cache = {}

    def _signal_matches_market_state(
        self, signal_direction: str, market_state: MarketState
    ) -> bool:
        """
        Check if a signal direction matches the detected market state.

        Args:
            signal_direction: Signal direction ('buy' or 'sell')
            market_state: Detected market state

        Returns:
            True if the signal matches the market state, False otherwise
        """
        state_type = market_state.state_type

        # Check trend alignment
        if (
            state_type in [MarketStateType.STRONG_UPTREND, MarketStateType.WEAK_UPTREND]
            and signal_direction == "buy"
        ):
            return True
        elif (
            state_type
            in [MarketStateType.STRONG_DOWNTREND, MarketStateType.WEAK_DOWNTREND]
            and signal_direction == "sell"
        ):
            return True

        # Check reversal alignment
        elif (
            state_type == MarketStateType.POTENTIAL_REVERSAL_UP
            and signal_direction == "buy"
        ):
            return True
        elif (
            state_type == MarketStateType.POTENTIAL_REVERSAL_DOWN
            and signal_direction == "sell"
        ):
            return True

        # Check breakout alignment
        elif state_type == MarketStateType.BREAKOUT_UP and signal_direction == "buy":
            return True
        elif state_type == MarketStateType.BREAKOUT_DOWN and signal_direction == "sell":
            return True

        return False

    def apply_market_state_analysis(
        self,
        pair: str,
        timeframe: str,
        features: Dict[str, Any],
        signals: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Apply market state analysis to filter and adjust signals.

        Args:
            pair: Currency pair
            timeframe: Timeframe
            features: Dictionary of features
            signals: List of signals to filter and adjust

        Returns:
            Filtered and adjusted signals
        """
        # Detect market state
        market_state = self.detect_market_state(pair, timeframe, features)

        # Filter and adjust signals
        filtered_signals = []
        for signal in signals:
            should_keep, adjusted_signal = self.filter_signal(signal, market_state)
            if should_keep:
                filtered_signals.append(adjusted_signal)

        logger.debug(
            f"Applied market state analysis to {len(signals)} signals, "
            f"kept {len(filtered_signals)}"
        )

        return filtered_signals

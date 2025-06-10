"""
Trend Following Strategy with Market State Awareness

This module provides a trend-following strategy that leverages market state detection
to adjust its behavior based on the current market conditions, demonstrating how to
use the MarketStateAwareMixin.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np

from forex_ai.analysis.technical.strategies.market_state_aware_mixin import (
    MarketStateAwareMixin,
)
from forex_ai.custom_types import Signal, SignalStrength, OrderType

logger = logging.getLogger(__name__)


class TrendFollowingStateAwareStrategy(MarketStateAwareMixin):
    """
    A trend-following strategy that adapts to detected market states.

    This strategy uses moving average crossovers for its core trend detection logic,
    but enhances it with market state awareness to filter signals and adjust position
    sizing based on the current market conditions.
    """

    def __init__(self, **kwargs):
        """
        Initialize the trend following strategy with market state awareness.

        Args:
            **kwargs: Strategy parameters
        """
        # Default strategy parameters
        self.strategy_params = {
            "fast_ma_period": 20,  # Fast moving average period
            "slow_ma_period": 50,  # Slow moving average period
            "ma_type": "ema",  # Moving average type (ema, sma)
            "min_trend_strength": 0.3,  # Minimum trend strength to generate signals
            "stop_loss_atr_multiple": 1.5,  # ATR multiple for stop loss
            "take_profit_atr_multiple": 3.0,  # ATR multiple for take profit
            "atr_period": 14,  # ATR period
            # Market state specific parameters - passed to MarketStateAwareMixin
            "market_state_params": {
                "min_confidence_threshold": 0.5,  # Min confidence to consider a state
                "trend_bias_multiplier": 1.3,  # Increase signal strength in trends
                "range_bias_multiplier": 0.7,  # Decrease signal strength in ranges
                "breakout_filter": True,  # Whether to filter signals in breakout states
            },
        }

        # Override with any provided kwargs
        self.strategy_params.update(kwargs)

        # Initialize MarketStateAwareMixin with market state parameters
        super().__init__(
            market_state_params=self.strategy_params.get("market_state_params", {})
        )

        logger.info(
            "Trend Following State-Aware Strategy initialized with parameters: %s",
            self.strategy_params,
        )

    def generate_signals(
        self, pair: str, timeframe: str, data: Dict[str, Any]
    ) -> List[Signal]:
        """
        Generate trading signals based on trend following with market state awareness.

        Args:
            pair: Currency pair (e.g., 'EUR_USD')
            timeframe: Timeframe (e.g., 'H1')
            data: Dictionary containing OHLCV data and calculated indicators

        Returns:
            List of generated signals
        """
        logger.debug(f"Generating signals for {pair}/{timeframe}")

        # Ensure we have the required data
        if not self._validate_data(data):
            logger.warning(f"Missing required data for {pair}/{timeframe}")
            return []

        # Calculate moving averages if not already provided
        if self.strategy_params["ma_type"] == "ema":
            if "ema_fast" not in data:
                data["ema_fast"] = self._calculate_ema(
                    data["close"], self.strategy_params["fast_ma_period"]
                )
            if "ema_slow" not in data:
                data["ema_slow"] = self._calculate_ema(
                    data["close"], self.strategy_params["slow_ma_period"]
                )
            fast_ma = data["ema_fast"]
            slow_ma = data["ema_slow"]
        else:  # Default to SMA
            if "sma_fast" not in data:
                data["sma_fast"] = self._calculate_sma(
                    data["close"], self.strategy_params["fast_ma_period"]
                )
            if "sma_slow" not in data:
                data["sma_slow"] = self._calculate_sma(
                    data["close"], self.strategy_params["slow_ma_period"]
                )
            fast_ma = data["sma_fast"]
            slow_ma = data["sma_slow"]

        # Calculate ATR for position sizing if not provided
        if "atr" not in data:
            data["atr"] = self._calculate_atr(
                data["high"],
                data["low"],
                data["close"],
                self.strategy_params["atr_period"],
            )
        atr = data["atr"][-1] if isinstance(data["atr"], list) else data["atr"]

        # Check for moving average crossover
        if len(fast_ma) < 2 or len(slow_ma) < 2:
            logger.warning(f"Insufficient data for crossover detection")
            return []

        # Check current and previous crossover status
        current_crossover = fast_ma[-1] > slow_ma[-1]
        previous_crossover = fast_ma[-2] > slow_ma[-2]

        # Calculate trend strength
        trend_strength = self._calculate_trend_strength(fast_ma, slow_ma, data)

        # Generate raw signals based on moving average crossover
        signals = []

        # Bullish crossover (fast MA crosses above slow MA)
        if current_crossover and not previous_crossover:
            # Create buy signal
            stop_loss_pips = (
                atr * self.strategy_params["stop_loss_atr_multiple"] * 10000
            )  # Convert to pips
            take_profit_pips = (
                atr * self.strategy_params["take_profit_atr_multiple"] * 10000
            )  # Convert to pips

            signal = Signal(
                pair=pair,
                timeframe=timeframe,
                direction="buy",
                order_type=OrderType.MARKET,
                entry_price=data["close"][-1],
                stop_loss_pips=stop_loss_pips,
                take_profit_pips=take_profit_pips,
                strength=trend_strength,
                metadata={
                    "strategy": "trend_following_state_aware",
                    "reason": "bullish_ma_crossover",
                    "fast_ma": fast_ma[-1],
                    "slow_ma": slow_ma[-1],
                    "atr": atr,
                },
            )
            signals.append(signal)

        # Bearish crossover (fast MA crosses below slow MA)
        elif not current_crossover and previous_crossover:
            # Create sell signal
            stop_loss_pips = (
                atr * self.strategy_params["stop_loss_atr_multiple"] * 10000
            )  # Convert to pips
            take_profit_pips = (
                atr * self.strategy_params["take_profit_atr_multiple"] * 10000
            )  # Convert to pips

            signal = Signal(
                pair=pair,
                timeframe=timeframe,
                direction="sell",
                order_type=OrderType.MARKET,
                entry_price=data["close"][-1],
                stop_loss_pips=stop_loss_pips,
                take_profit_pips=take_profit_pips,
                strength=trend_strength,
                metadata={
                    "strategy": "trend_following_state_aware",
                    "reason": "bearish_ma_crossover",
                    "fast_ma": fast_ma[-1],
                    "slow_ma": slow_ma[-1],
                    "atr": atr,
                },
            )
            signals.append(signal)

        # Skip signals with insufficient trend strength
        if trend_strength < self.strategy_params["min_trend_strength"]:
            logger.debug(
                f"Signal skipped due to insufficient trend strength: {trend_strength:.2f}"
            )
            return []

        # Apply market state analysis to filter and adjust signals
        filtered_signals = self.apply_market_state_analysis(
            pair, timeframe, data, signals
        )

        if len(filtered_signals) != len(signals):
            logger.info(
                f"Filtered {len(signals) - len(filtered_signals)} signals based on market state"
            )

        return filtered_signals

    def _validate_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate that the data dictionary contains required data.

        Args:
            data: Data dictionary

        Returns:
            True if required data is present, False otherwise
        """
        required_keys = ["open", "high", "low", "close"]
        return all(key in data for key in required_keys) and len(data["close"]) >= max(
            self.strategy_params["fast_ma_period"],
            self.strategy_params["slow_ma_period"],
        )

    def _calculate_ema(self, prices: List[float], period: int) -> List[float]:
        """
        Calculate Exponential Moving Average.

        Args:
            prices: List of price values
            period: EMA period

        Returns:
            List of EMA values
        """
        if len(prices) < period:
            return []

        alpha = 2.0 / (period + 1)
        ema = [prices[0]]

        for i in range(1, len(prices)):
            ema.append(alpha * prices[i] + (1 - alpha) * ema[i - 1])

        return ema

    def _calculate_sma(self, prices: List[float], period: int) -> List[float]:
        """
        Calculate Simple Moving Average.

        Args:
            prices: List of price values
            period: SMA period

        Returns:
            List of SMA values
        """
        if len(prices) < period:
            return []

        sma = []
        for i in range(len(prices) - period + 1):
            sma.append(sum(prices[i : i + period]) / period)

        # Pad with NaN to match original length
        return [float("nan")] * (period - 1) + sma

    def _calculate_atr(
        self, high: List[float], low: List[float], close: List[float], period: int
    ) -> List[float]:
        """
        Calculate Average True Range.

        Args:
            high: List of high prices
            low: List of low prices
            close: List of close prices
            period: ATR period

        Returns:
            List of ATR values
        """
        if len(high) < period + 1 or len(low) < period + 1 or len(close) < period + 1:
            return []

        # Calculate True Range
        tr = []
        for i in range(1, len(close)):
            tr1 = high[i] - low[i]  # Current high - current low
            tr2 = abs(high[i] - close[i - 1])  # Current high - previous close
            tr3 = abs(low[i] - close[i - 1])  # Current low - previous close
            tr.append(max(tr1, tr2, tr3))

        # Calculate ATR
        atr = []
        atr.append(sum(tr[:period]) / period)

        for i in range(1, len(tr) - period + 1):
            atr.append((atr[i - 1] * (period - 1) + tr[i + period - 1]) / period)

        # Pad with NaN to match original length
        return [float("nan")] * period + atr

    def _calculate_trend_strength(
        self, fast_ma: List[float], slow_ma: List[float], data: Dict[str, Any]
    ) -> float:
        """
        Calculate trend strength based on various indicators.

        Args:
            fast_ma: Fast moving average values
            slow_ma: Slow moving average values
            data: Data dictionary with additional indicators

        Returns:
            Trend strength as a value between 0.0 and 1.0
        """
        # Base strength on MA separation relative to ATR
        ma_separation = abs(fast_ma[-1] - slow_ma[-1])
        atr = data["atr"][-1] if isinstance(data["atr"], list) else data["atr"]

        # Normalize based on ATR
        norm_separation = min(1.0, ma_separation / (atr * 2))

        # If ADX is available, factor it in
        adx_factor = 0.5
        if "adx" in data:
            adx = data["adx"][-1] if isinstance(data["adx"], list) else data["adx"]
            # Normalize ADX (typical range is 0-100, but effective range is often 0-50)
            norm_adx = min(1.0, adx / 50.0)
            adx_factor = norm_adx

        # If RSI is available, check for extreme values
        rsi_factor = 0.5
        if "rsi" in data:
            rsi = data["rsi"][-1] if isinstance(data["rsi"], list) else data["rsi"]
            # Stronger trend if RSI is aligned with direction (high for uptrend, low for downtrend)
            if fast_ma[-1] > slow_ma[-1]:  # Uptrend
                rsi_factor = min(1.0, rsi / 70.0)
            else:  # Downtrend
                rsi_factor = min(1.0, (100 - rsi) / 70.0)

        # Combine factors
        strength = 0.5 * norm_separation + 0.3 * adx_factor + 0.2 * rsi_factor

        # Cap at 1.0
        return min(1.0, strength)


# Function to create a new strategy instance
def create_trend_following_state_aware_strategy(
    **kwargs,
) -> TrendFollowingStateAwareStrategy:
    """
    Create a new TrendFollowingStateAwareStrategy instance.

    Args:
        **kwargs: Strategy parameters to override defaults

    Returns:
        Initialized TrendFollowingStateAwareStrategy
    """
    return TrendFollowingStateAwareStrategy(**kwargs)

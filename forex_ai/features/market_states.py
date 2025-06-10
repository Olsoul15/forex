"""
Market state analysis for the Forex AI Trading System.

This module provides functionality for analyzing and identifying
market states (trends, ranges, volatility regimes, etc.) for forex markets.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from enum import Enum, auto

from forex_ai.data.market_data import get_market_data_service
from forex_ai.utils.logging import get_logger

logger = get_logger(__name__)


class MarketState(Enum):
    """Enum defining different market states."""

    STRONG_UPTREND = auto()
    UPTREND = auto()
    WEAK_UPTREND = auto()
    CONSOLIDATION = auto()
    RANGE_BOUND = auto()
    WEAK_DOWNTREND = auto()
    DOWNTREND = auto()
    STRONG_DOWNTREND = auto()
    BREAKOUT = auto()
    REVERSAL = auto()
    HIGH_VOLATILITY = auto()
    LOW_VOLATILITY = auto()
    UNDEFINED = auto()


class VolatilityRegime(Enum):
    """Enum defining different volatility regimes."""

    VERY_LOW = auto()
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    VERY_HIGH = auto()
    INCREASING = auto()
    DECREASING = auto()


class MarketStateAnalyzer:
    """
    Analyzer for identifying forex market states.

    This class analyzes price data to identify the current market state,
    which can be used to adapt trading strategies to different market conditions.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the market state analyzer.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.data_service = get_market_data_service()

        # Configuration for state detection
        self.trend_lookback = self.config.get("trend_lookback", 20)
        self.volatility_lookback = self.config.get("volatility_lookback", 14)
        self.very_high_vol_threshold = self.config.get("very_high_vol_threshold", 1.5)
        self.high_vol_threshold = self.config.get("high_vol_threshold", 1.2)
        self.low_vol_threshold = self.config.get("low_vol_threshold", 0.8)
        self.very_low_vol_threshold = self.config.get("very_low_vol_threshold", 0.5)

    def analyze_market_state(
        self, data: pd.DataFrame, symbol: str, timeframe: str
    ) -> Dict[str, Any]:
        """
        Analyze market state from price data.

        Args:
            data: Price data with OHLCV columns
            symbol: Trading symbol (e.g., 'EUR/USD')
            timeframe: Timeframe of the data

        Returns:
            Dictionary containing market state analysis
        """
        try:
            logger.info(f"Analyzing market state for {symbol} on {timeframe}")

            # Ensure we have enough data
            if len(data) < max(self.trend_lookback, self.volatility_lookback) + 10:
                logger.warning(
                    f"Insufficient data for market state analysis: {len(data)} bars"
                )
                return {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "state": MarketState.UNDEFINED.name,
                    "confidence": 0.0,
                    "error": "Insufficient data",
                }

            # Calculate trend indicators
            data = self._calculate_trend_indicators(data)

            # Calculate volatility
            data = self._calculate_volatility_indicators(data)

            # Identify market state
            current_state, confidence = self._identify_market_state(data)

            # Determine volatility regime
            volatility_regime = self._determine_volatility_regime(data)

            # Create analysis result
            result = {
                "symbol": symbol,
                "timeframe": timeframe,
                "state": current_state.name,
                "volatility_regime": volatility_regime.name,
                "confidence": confidence,
                "analysis_time": datetime.now().isoformat(),
                "metrics": {
                    "atr": float(data["atr"].iloc[-1]),
                    "atr_percent": float(data["atr_percent"].iloc[-1]),
                    "rsi": (
                        float(data["rsi"].iloc[-1]) if "rsi" in data.columns else None
                    ),
                    "adx": (
                        float(data["adx"].iloc[-1]) if "adx" in data.columns else None
                    ),
                    "ma_direction": int(data["ma_direction"].iloc[-1]),
                    "price_distance_from_ma": float(
                        data["price_distance_from_ma"].iloc[-1]
                    ),
                },
            }

            return result

        except Exception as e:
            logger.error(f"Error analyzing market state: {str(e)}")
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "state": MarketState.UNDEFINED.name,
                "confidence": 0.0,
                "error": str(e),
            }

    def _calculate_trend_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trend indicators on the price data.

        Args:
            data: Price data

        Returns:
            DataFrame with added indicators
        """
        # Make a copy to avoid modifying the original
        df = data.copy()

        # Calculate moving averages
        df["ma20"] = df["close"].rolling(window=20).mean()
        df["ma50"] = df["close"].rolling(window=50).mean()
        df["ma200"] = df["close"].rolling(window=200).mean()

        # Calculate MA directions (1 for up, -1 for down, 0 for flat)
        df["ma20_direction"] = np.where(
            df["ma20"].diff() > 0, 1, np.where(df["ma20"].diff() < 0, -1, 0)
        )
        df["ma50_direction"] = np.where(
            df["ma50"].diff() > 0, 1, np.where(df["ma50"].diff() < 0, -1, 0)
        )
        df["ma200_direction"] = np.where(
            df["ma200"].diff() > 0, 1, np.where(df["ma200"].diff() < 0, -1, 0)
        )

        # Overall MA direction (weighted average)
        df["ma_direction"] = (
            df["ma20_direction"] * 0.5
            + df["ma50_direction"] * 0.3
            + df["ma200_direction"] * 0.2
        )

        # Price distance from MA (as percentage)
        df["price_distance_from_ma"] = (df["close"] - df["ma20"]) / df["ma20"] * 100

        # Calculate RSI
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()

        rs = avg_gain / avg_loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # Calculate ADX (Simplified version)
        df["tr"] = np.maximum(
            df["high"] - df["low"],
            np.maximum(
                abs(df["high"] - df["close"].shift(1)),
                abs(df["low"] - df["close"].shift(1)),
            ),
        )

        df["plus_dm"] = np.where(
            (df["high"] - df["high"].shift(1)) > (df["low"].shift(1) - df["low"]),
            np.maximum(df["high"] - df["high"].shift(1), 0),
            0,
        )

        df["minus_dm"] = np.where(
            (df["low"].shift(1) - df["low"]) > (df["high"] - df["high"].shift(1)),
            np.maximum(df["low"].shift(1) - df["low"], 0),
            0,
        )

        # Smooth these values
        df["tr14"] = df["tr"].rolling(window=14).mean()
        df["plus_di14"] = 100 * (df["plus_dm"].rolling(window=14).mean() / df["tr14"])
        df["minus_di14"] = 100 * (df["minus_dm"].rolling(window=14).mean() / df["tr14"])

        # Calculate DX and ADX
        df["dx"] = (
            100
            * abs(df["plus_di14"] - df["minus_di14"])
            / (df["plus_di14"] + df["minus_di14"])
        )
        df["adx"] = df["dx"].rolling(window=14).mean()

        return df

    def _calculate_volatility_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volatility indicators on the price data.

        Args:
            data: Price data

        Returns:
            DataFrame with added indicators
        """
        # Make a copy to avoid modifying the original
        df = data.copy()

        # Calculate ATR (Average True Range)
        df["tr"] = np.maximum(
            df["high"] - df["low"],
            np.maximum(
                abs(df["high"] - df["close"].shift(1)),
                abs(df["low"] - df["close"].shift(1)),
            ),
        )

        df["atr"] = df["tr"].rolling(window=self.volatility_lookback).mean()

        # ATR as percentage of price
        df["atr_percent"] = df["atr"] / df["close"] * 100

        # Normalized ATR (Z-score)
        atr_mean = df["atr"].rolling(window=50).mean()
        atr_std = df["atr"].rolling(window=50).std()
        df["atr_z"] = (df["atr"] - atr_mean) / atr_std

        # Bollinger Bands width
        df["sma20"] = df["close"].rolling(window=20).mean()
        df["std20"] = df["close"].rolling(window=20).std()
        df["upper_band"] = df["sma20"] + (df["std20"] * 2)
        df["lower_band"] = df["sma20"] - (df["std20"] * 2)
        df["bb_width"] = (df["upper_band"] - df["lower_band"]) / df["sma20"] * 100

        return df

    def _identify_market_state(self, data: pd.DataFrame) -> Tuple[MarketState, float]:
        """
        Identify the current market state.

        Args:
            data: Price data with indicators

        Returns:
            Tuple of (MarketState, confidence)
        """
        # Use the most recent data for analysis
        recent_data = data.iloc[-10:].copy()

        # Get current indicator values
        ma_direction = recent_data["ma_direction"].iloc[-1]
        rsi = recent_data["rsi"].iloc[-1]
        adx = recent_data["adx"].iloc[-1]
        atr_z = recent_data["atr_z"].iloc[-1]

        # Define threshold values
        strong_trend_adx = 30
        trend_adx = 20
        range_adx = 15

        # Initialize state and confidence
        state = MarketState.UNDEFINED
        confidence = 0.5

        # Strong trend conditions
        if adx > strong_trend_adx:
            if ma_direction > 0.5:
                state = MarketState.STRONG_UPTREND
                confidence = min(0.9, 0.6 + (adx - strong_trend_adx) / 50)
            elif ma_direction < -0.5:
                state = MarketState.STRONG_DOWNTREND
                confidence = min(0.9, 0.6 + (adx - strong_trend_adx) / 50)

        # Normal trend conditions
        elif adx > trend_adx:
            if ma_direction > 0.3:
                state = MarketState.UPTREND
                confidence = 0.6 + (adx - trend_adx) / 100
            elif ma_direction < -0.3:
                state = MarketState.DOWNTREND
                confidence = 0.6 + (adx - trend_adx) / 100
            else:
                # Mixed signals
                if rsi > 60:
                    state = MarketState.WEAK_UPTREND
                    confidence = 0.5
                elif rsi < 40:
                    state = MarketState.WEAK_DOWNTREND
                    confidence = 0.5
                else:
                    state = MarketState.CONSOLIDATION
                    confidence = 0.5

        # Range-bound conditions
        elif adx < range_adx:
            state = MarketState.RANGE_BOUND
            confidence = 0.6 + (range_adx - adx) / 50

        # Volatility-based states (can override previous state)
        if atr_z > 2.0:
            # Very high volatility could indicate a breakout
            bb_width_change = (
                (data["bb_width"].iloc[-1] - data["bb_width"].iloc[-5])
                / data["bb_width"].iloc[-5]
                * 100
            )

            if bb_width_change > 20:
                state = MarketState.BREAKOUT
                confidence = min(0.9, 0.6 + atr_z / 10)
            else:
                state = MarketState.HIGH_VOLATILITY
                confidence = min(0.8, 0.5 + atr_z / 10)

        # Check for potential reversal
        price_ma_cross = (
            (data["close"].iloc[-2] - data["ma20"].iloc[-2])
            * (data["close"].iloc[-1] - data["ma20"].iloc[-1])
        ) < 0

        if price_ma_cross and (rsi > 70 or rsi < 30):
            state = MarketState.REVERSAL
            confidence = 0.6

        return state, confidence

    def _determine_volatility_regime(self, data: pd.DataFrame) -> VolatilityRegime:
        """
        Determine the current volatility regime.

        Args:
            data: Price data with indicators

        Returns:
            VolatilityRegime
        """
        # Get current ATR percentage
        current_atr_pct = data["atr_percent"].iloc[-1]

        # Compare recent ATR with historical ATR
        atr_50_mean = data["atr_percent"].rolling(window=50).mean().iloc[-1]
        atr_50_std = data["atr_percent"].rolling(window=50).std().iloc[-1]

        # Z-score of current ATR
        atr_z = (current_atr_pct - atr_50_mean) / atr_50_std if atr_50_std > 0 else 0

        # Check if volatility is increasing or decreasing
        recent_atr_trend = data["atr_percent"].iloc[-5:].diff().mean()

        if recent_atr_trend > 0.05:
            return VolatilityRegime.INCREASING
        elif recent_atr_trend < -0.05:
            return VolatilityRegime.DECREASING

        # Static volatility state
        if atr_z > self.very_high_vol_threshold:
            return VolatilityRegime.VERY_HIGH
        elif atr_z > self.high_vol_threshold:
            return VolatilityRegime.HIGH
        elif atr_z < self.very_low_vol_threshold:
            return VolatilityRegime.VERY_LOW
        elif atr_z < self.low_vol_threshold:
            return VolatilityRegime.LOW
        else:
            return VolatilityRegime.MEDIUM


def get_market_state(
    symbol: str, timeframe: str, lookback_bars: int = 100
) -> Dict[str, Any]:
    """
    Convenience function to get the current market state for a symbol.

    Args:
        symbol: Trading symbol (e.g., 'EUR/USD')
        timeframe: Timeframe (e.g., '1h', '4h', '1d')
        lookback_bars: Number of bars to analyze

    Returns:
        Market state analysis
    """
    try:
        # Get market data
        data_service = get_market_data_service()
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=lookback_bars)).strftime(
            "%Y-%m-%d"
        )

        # Fetch data
        data = data_service.fetch_data(
            symbol=symbol, timeframe=timeframe, start_date=start_date, end_date=end_date
        )

        # Convert to DataFrame if needed
        if not isinstance(data, pd.DataFrame):
            df = pd.DataFrame(data)
        else:
            df = data

        # Analyze market state
        analyzer = MarketStateAnalyzer()
        return analyzer.analyze_market_state(df, symbol, timeframe)

    except Exception as e:
        logger.error(f"Error getting market state: {str(e)}")
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "state": MarketState.UNDEFINED.name,
            "confidence": 0.0,
            "error": str(e),
        }

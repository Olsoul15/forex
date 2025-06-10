"""
Multi-Timeframe Analysis Module for the AI Forex Trading System.

This module provides tools for analyzing forex data across multiple timeframes,
helping to identify trends, support/resistance levels, and other patterns that
may not be visible on a single timeframe.
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import pandas as pd
import numpy as np
from enum import Enum
from datetime import datetime, timedelta
import talib

from forex_ai.utils.logging import get_logger

logger = get_logger(__name__)


class Timeframe(Enum):
    """Timeframe enumeration for forex analysis."""

    M1 = "1min"  # 1 minute
    M5 = "5min"  # 5 minutes
    M15 = "15min"  # 15 minutes
    M30 = "30min"  # 30 minutes
    H1 = "1H"  # 1 hour
    H4 = "4H"  # 4 hours
    D1 = "D"  # 1 day
    W1 = "W"  # 1 week
    MN = "M"  # 1 month

    @classmethod
    def from_string(cls, timeframe_str: str) -> "Timeframe":
        """Convert string to Timeframe enum."""
        for tf in cls:
            if tf.value == timeframe_str or tf.name == timeframe_str:
                return tf
        raise ValueError(f"Unknown timeframe: {timeframe_str}")

    @property
    def minutes(self) -> int:
        """Get timeframe in minutes."""
        if self == Timeframe.M1:
            return 1
        elif self == Timeframe.M5:
            return 5
        elif self == Timeframe.M15:
            return 15
        elif self == Timeframe.M30:
            return 30
        elif self == Timeframe.H1:
            return 60
        elif self == Timeframe.H4:
            return 240
        elif self == Timeframe.D1:
            return 1440
        elif self == Timeframe.W1:
            return 10080
        elif self == Timeframe.MN:
            return 43200
        else:
            raise ValueError(f"Unknown timeframe: {self}")

    def __lt__(self, other: "Timeframe") -> bool:
        """Compare timeframes."""
        return self.minutes < other.minutes

    def __gt__(self, other: "Timeframe") -> bool:
        """Compare timeframes."""
        return self.minutes > other.minutes

    def __le__(self, other: "Timeframe") -> bool:
        """Compare timeframes."""
        return self.minutes <= other.minutes

    def __ge__(self, other: "Timeframe") -> bool:
        """Compare timeframes."""
        return self.minutes >= other.minutes


def resample_ohlcv(
    data: pd.DataFrame, target_timeframe: Union[str, Timeframe]
) -> pd.DataFrame:
    """
    Resample OHLCV data to a different timeframe.

    Args:
        data: OHLCV DataFrame with DatetimeIndex
        target_timeframe: Target timeframe (e.g., '1H', '4H', 'D')

    Returns:
        Resampled DataFrame
    """
    if isinstance(target_timeframe, Timeframe):
        target_timeframe = target_timeframe.value

    # Define resampling rules
    ohlc_dict = {"open": "first", "high": "max", "low": "min", "close": "last"}

    # Check for volume column
    if "volume" in data.columns:
        ohlc_dict["volume"] = "sum"

    # Resample data
    resampled = data.resample(target_timeframe).agg(ohlc_dict)

    return resampled.dropna()


def align_timeframes(
    data_dict: Dict[Union[str, Timeframe], pd.DataFrame],
) -> Dict[Union[str, Timeframe], pd.DataFrame]:
    """
    Align data from multiple timeframes to have the same start and end dates.

    Args:
        data_dict: Dictionary mapping timeframes to DataFrames

    Returns:
        Dictionary with aligned DataFrames
    """
    # Find common date range
    start_dates = [df.index[0] for df in data_dict.values() if not df.empty]
    end_dates = [df.index[-1] for df in data_dict.values() if not df.empty]

    if not start_dates or not end_dates:
        return data_dict

    common_start = max(start_dates)
    common_end = min(end_dates)

    # Align DataFrames
    aligned_dict = {}
    for tf, df in data_dict.items():
        if df.empty:
            aligned_dict[tf] = df
            continue

        aligned_df = df.loc[common_start:common_end].copy()
        aligned_dict[tf] = aligned_df

    return aligned_dict


def apply_indicator(
    data_dict: Dict[Union[str, Timeframe], pd.DataFrame],
    indicator_func: Callable,
    column_prefix: str,
    **kwargs,
) -> Dict[Union[str, Timeframe], pd.DataFrame]:
    """
    Apply indicator function to data from multiple timeframes.

    Args:
        data_dict: Dictionary mapping timeframes to DataFrames
        indicator_func: Indicator function to apply
        column_prefix: Prefix for indicator columns
        **kwargs: Additional arguments for indicator function

    Returns:
        Dictionary with DataFrames including indicator values
    """
    result_dict = {}

    for tf, df in data_dict.items():
        if df.empty:
            result_dict[tf] = df
            continue

        # Apply indicator
        result = indicator_func(df, **kwargs)

        # If result is a Series, convert to DataFrame
        if isinstance(result, pd.Series):
            result = pd.DataFrame({f"{column_prefix}": result})

        # If result is a tuple of Series/Arrays, convert to DataFrame
        elif isinstance(result, tuple):
            result = pd.DataFrame(
                {
                    f"{column_prefix}_{i+1}": pd.Series(r, index=df.index)
                    for i, r in enumerate(result)
                }
            )

        # Merge result with original data
        result_dict[tf] = pd.concat([df, result], axis=1)

    return result_dict


class MTFIndicator:
    """Base class for multi-timeframe indicators."""

    def __init__(self, timeframes: List[Union[str, Timeframe]]):
        """
        Initialize with list of timeframes.

        Args:
            timeframes: List of timeframes to analyze
        """
        self.timeframes = [
            tf if isinstance(tf, Timeframe) else Timeframe.from_string(tf)
            for tf in timeframes
        ]

        # Sort timeframes from smallest to largest
        self.timeframes.sort()

    def calculate(self, data: pd.DataFrame) -> Dict[Timeframe, pd.DataFrame]:
        """
        Calculate indicator values for each timeframe.

        Args:
            data: OHLCV DataFrame for smallest timeframe

        Returns:
            Dictionary mapping timeframes to DataFrames with indicator values
        """
        raise NotImplementedError("Subclasses must implement this method")

    def analyze(self, data_dict: Dict[Timeframe, pd.DataFrame]) -> Dict[str, Any]:
        """
        Analyze indicator values across timeframes.

        Args:
            data_dict: Dictionary mapping timeframes to DataFrames with indicator values

        Returns:
            Dictionary with analysis results
        """
        raise NotImplementedError("Subclasses must implement this method")


class MTFTrend(MTFIndicator):
    """Multi-timeframe trend analysis."""

    def __init__(
        self,
        timeframes: List[Union[str, Timeframe]],
        ma_type: str = "sma",
        fast_period: int = 20,
        slow_period: int = 50,
    ):
        """
        Initialize multi-timeframe trend analyzer.

        Args:
            timeframes: List of timeframes to analyze
            ma_type: Moving average type ('sma', 'ema', 'wma')
            fast_period: Fast moving average period
            slow_period: Slow moving average period
        """
        super().__init__(timeframes)
        self.ma_type = ma_type.lower()
        self.fast_period = fast_period
        self.slow_period = slow_period

    def calculate(self, data: pd.DataFrame) -> Dict[Timeframe, pd.DataFrame]:
        """
        Calculate trend indicators for each timeframe.

        Args:
            data: OHLCV DataFrame for smallest timeframe

        Returns:
            Dictionary mapping timeframes to DataFrames with trend indicators
        """
        # Create dictionary to hold resampled data
        data_dict = {}

        # Resample data for each timeframe
        for tf in self.timeframes:
            if tf == self.timeframes[0]:
                # For the smallest timeframe, use original data
                data_dict[tf] = data.copy()
            else:
                # Resample to higher timeframes
                data_dict[tf] = resample_ohlcv(data, tf)

        # Calculate moving averages for each timeframe
        for tf, df in data_dict.items():
            # Select MA function based on type
            if self.ma_type == "ema":
                ma_func = talib.EMA
            elif self.ma_type == "wma":
                ma_func = talib.WMA
            else:  # default to SMA
                ma_func = talib.SMA

            # Calculate MAs
            df[f"fast_ma"] = ma_func(df["close"].values, timeperiod=self.fast_period)
            df[f"slow_ma"] = ma_func(df["close"].values, timeperiod=self.slow_period)

            # Calculate trend direction
            df["trend"] = 0  # Neutral
            df.loc[df["fast_ma"] > df["slow_ma"], "trend"] = 1  # Bullish
            df.loc[df["fast_ma"] < df["slow_ma"], "trend"] = -1  # Bearish

            # Calculate trend strength
            df["trend_strength"] = (
                abs(df["fast_ma"] - df["slow_ma"]) / df["close"] * 100
            )

            # Calculate MA slope
            df["fast_ma_slope"] = df["fast_ma"].diff(5) / df["fast_ma"].shift(5) * 100
            df["slow_ma_slope"] = df["slow_ma"].diff(5) / df["slow_ma"].shift(5) * 100

            # Drop NaN values
            data_dict[tf] = df.dropna()

        return data_dict

    def analyze(self, data_dict: Dict[Timeframe, pd.DataFrame]) -> Dict[str, Any]:
        """
        Analyze trend indicators across timeframes.

        Args:
            data_dict: Dictionary mapping timeframes to DataFrames with trend indicators

        Returns:
            Dictionary with trend analysis results
        """
        # Get latest trend values for each timeframe
        latest_trends = {}
        for tf, df in data_dict.items():
            if df.empty:
                continue

            latest = df.iloc[-1]
            latest_trends[tf] = {
                "trend": latest["trend"],
                "trend_strength": latest["trend_strength"],
                "fast_ma_slope": latest["fast_ma_slope"],
                "slow_ma_slope": latest["slow_ma_slope"],
            }

        # Count timeframes with bullish, bearish, neutral trends
        bullish_count = sum(
            1 for tf_data in latest_trends.values() if tf_data["trend"] > 0
        )
        bearish_count = sum(
            1 for tf_data in latest_trends.values() if tf_data["trend"] < 0
        )
        neutral_count = sum(
            1 for tf_data in latest_trends.values() if tf_data["trend"] == 0
        )

        # Calculate overall trend alignment
        total_timeframes = len(latest_trends)
        if total_timeframes == 0:
            trend_alignment = 0.0
        else:
            trend_alignment = (
                max(bullish_count, bearish_count, neutral_count) / total_timeframes
            )

        # Determine overall trend
        if bullish_count > bearish_count and bullish_count > neutral_count:
            overall_trend = 1  # Bullish
        elif bearish_count > bullish_count and bearish_count > neutral_count:
            overall_trend = -1  # Bearish
        else:
            overall_trend = 0  # Neutral or conflicted

        # Calculate weighted trend strength
        # Higher timeframes have more weight
        weighted_trend_strength = 0.0
        total_weight = 0.0

        for i, tf in enumerate(sorted(latest_trends.keys())):
            # Weight increases with timeframe
            weight = i + 1
            weighted_trend_strength += (
                latest_trends[tf]["trend"]
                * latest_trends[tf]["trend_strength"]
                * weight
            )
            total_weight += weight

        if total_weight > 0:
            weighted_trend_strength /= total_weight

        # Return analysis results
        return {
            "latest_trends": latest_trends,
            "bullish_timeframes": bullish_count,
            "bearish_timeframes": bearish_count,
            "neutral_timeframes": neutral_count,
            "trend_alignment": trend_alignment,
            "overall_trend": overall_trend,
            "weighted_trend_strength": weighted_trend_strength,
            "analysis_time": datetime.now().isoformat(),
        }


class MTFSupport:
    """Multi-timeframe support and resistance analysis."""

    def __init__(
        self,
        timeframes: List[Union[str, Timeframe]],
        lookback: int = 20,
        swing_strength: int = 3,
        price_tolerance: float = 0.001,
    ):
        """
        Initialize support and resistance analyzer.

        Args:
            timeframes: List of timeframes to analyze
            lookback: Number of bars to look back for swing points
            swing_strength: Number of bars on each side to confirm swing
            price_tolerance: Price tolerance for merging levels (% of price)
        """
        self.timeframes = [
            tf if isinstance(tf, Timeframe) else Timeframe.from_string(tf)
            for tf in timeframes
        ]

        # Sort timeframes from smallest to largest
        self.timeframes.sort()

        self.lookback = lookback
        self.swing_strength = swing_strength
        self.price_tolerance = price_tolerance

    def find_swing_points(self, data: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """
        Find swing high and low points in price data.

        Args:
            data: OHLCV DataFrame

        Returns:
            Tuple of (support_levels, resistance_levels)
        """
        highs = data["high"].values
        lows = data["low"].values

        support_levels = []
        resistance_levels = []

        n = len(data)

        # Find swing highs
        for i in range(self.swing_strength, n - self.swing_strength):
            is_swing_high = True
            for j in range(1, self.swing_strength + 1):
                if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                    is_swing_high = False
                    break

            if is_swing_high:
                resistance_levels.append(highs[i])

        # Find swing lows
        for i in range(self.swing_strength, n - self.swing_strength):
            is_swing_low = True
            for j in range(1, self.swing_strength + 1):
                if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                    is_swing_low = False
                    break

            if is_swing_low:
                support_levels.append(lows[i])

        return support_levels, resistance_levels

    def merge_levels(
        self, levels: List[float], price: float
    ) -> List[Tuple[float, int]]:
        """
        Merge nearby price levels and count occurrences.

        Args:
            levels: List of price levels
            price: Current price for calculating percentage tolerance

        Returns:
            List of (level_price, occurrence_count) tuples
        """
        tolerance = price * self.price_tolerance
        merged = {}

        for level in sorted(levels):
            # Check if we can merge with an existing level
            merged_level = None
            for existing_level in merged:
                if abs(existing_level - level) < tolerance:
                    merged_level = existing_level
                    break

            if merged_level is not None:
                # Merge with existing level
                merged[merged_level] += 1
            else:
                # Create new level
                merged[level] = 1

        return [(level, count) for level, count in merged.items()]

    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate support and resistance levels across timeframes.

        Args:
            data: OHLCV DataFrame for smallest timeframe

        Returns:
            Dictionary with support and resistance levels
        """
        # Current price
        current_price = data["close"].iloc[-1]

        # Create dictionary for levels
        all_support_levels = []
        all_resistance_levels = []

        timeframe_levels = {}

        # Process each timeframe
        for tf in self.timeframes:
            # Resample data
            if tf == self.timeframes[0]:
                resampled = data.copy()
            else:
                resampled = resample_ohlcv(data, tf)

            # Get last N bars
            last_n_bars = resampled.iloc[-self.lookback :]

            # Find swing points
            support, resistance = self.find_swing_points(last_n_bars)

            # Add to overall levels
            all_support_levels.extend(support)
            all_resistance_levels.extend(resistance)

            # Store for this timeframe
            timeframe_levels[tf.name] = {
                "support": sorted(support),
                "resistance": sorted(resistance),
            }

        # Merge levels
        merged_support = self.merge_levels(all_support_levels, current_price)
        merged_resistance = self.merge_levels(all_resistance_levels, current_price)

        # Sort by strength (occurrence count)
        merged_support.sort(key=lambda x: x[1], reverse=True)
        merged_resistance.sort(key=lambda x: x[1], reverse=True)

        # Find nearest levels
        support_below = [level for level, _ in merged_support if level < current_price]
        resistance_above = [
            level for level, _ in merged_resistance if level > current_price
        ]

        nearest_support = max(support_below) if support_below else None
        nearest_resistance = min(resistance_above) if resistance_above else None

        return {
            "current_price": current_price,
            "support_levels": merged_support,
            "resistance_levels": merged_resistance,
            "nearest_support": nearest_support,
            "nearest_resistance": nearest_resistance,
            "timeframe_levels": timeframe_levels,
            "analysis_time": datetime.now().isoformat(),
        }


class MTFAligner:
    """Multi-timeframe analysis aligner.

    This class helps align insights from different timeframes and indicators
    to produce a consolidated analysis.
    """

    def __init__(self, timeframes: List[Union[str, Timeframe]]):
        """
        Initialize multi-timeframe aligner.

        Args:
            timeframes: List of timeframes to analyze
        """
        self.timeframes = [
            tf if isinstance(tf, Timeframe) else Timeframe.from_string(tf)
            for tf in timeframes
        ]
        self.timeframes.sort()

        self.analyzers = {}
        self.data_dict = {tf: None for tf in self.timeframes}

    def add_analyzer(self, name: str, analyzer: MTFIndicator) -> None:
        """
        Add an analyzer to the aligner.

        Args:
            name: Name for the analyzer
            analyzer: MTFIndicator instance
        """
        self.analyzers[name] = analyzer

    def add_data(self, timeframe: Union[str, Timeframe], data: pd.DataFrame) -> None:
        """
        Add data for a specific timeframe.

        Args:
            timeframe: Timeframe for the data
            data: OHLCV DataFrame
        """
        if isinstance(timeframe, str):
            timeframe = Timeframe.from_string(timeframe)

        self.data_dict[timeframe] = data

    def prepare_data(self, data: pd.DataFrame) -> None:
        """
        Prepare data for all timeframes by resampling.

        Args:
            data: OHLCV DataFrame for smallest timeframe
        """
        for tf in self.timeframes:
            if tf == self.timeframes[0]:
                self.data_dict[tf] = data.copy()
            else:
                self.data_dict[tf] = resample_ohlcv(data, tf)

    def analyze(self) -> Dict[str, Any]:
        """
        Run all analyzers and produce consolidated analysis.

        Returns:
            Dictionary with consolidated analysis results
        """
        results = {}

        # Run each analyzer
        for name, analyzer in self.analyzers.items():
            analyzer_data = {}

            # Check if analyzer uses MTFIndicator interface
            if isinstance(analyzer, MTFIndicator):
                # Use smallest timeframe data and let analyzer handle resampling
                smallest_tf = self.timeframes[0]
                if self.data_dict[smallest_tf] is not None:
                    analyzer_data = analyzer.calculate(self.data_dict[smallest_tf])
                    results[name] = analyzer.analyze(analyzer_data)
            else:
                # Assume custom analyzer interface
                if any(df is not None for df in self.data_dict.values()):
                    results[name] = analyzer.calculate(self.data_dict)

        # Calculate overall trend consensus
        if "trend" in results:
            trend_results = results["trend"]
            overall_trend = trend_results.get("overall_trend", 0)
            trend_alignment = trend_results.get("trend_alignment", 0)
        else:
            overall_trend = 0
            trend_alignment = 0

        # Check for support/resistance
        if "support_resistance" in results:
            sr_results = results["support_resistance"]
            nearest_support = sr_results.get("nearest_support")
            nearest_resistance = sr_results.get("nearest_resistance")

            current_price = sr_results.get("current_price")

            # Calculate distance to nearest levels
            if current_price and nearest_support:
                support_distance = (current_price - nearest_support) / current_price
            else:
                support_distance = None

            if current_price and nearest_resistance:
                resistance_distance = (
                    nearest_resistance - current_price
                ) / current_price
            else:
                resistance_distance = None

            # Determine if price is near level
            if support_distance is not None and support_distance < 0.01:
                near_support = True
            else:
                near_support = False

            if resistance_distance is not None and resistance_distance < 0.01:
                near_resistance = True
            else:
                near_resistance = False
        else:
            support_distance = None
            resistance_distance = None
            near_support = False
            near_resistance = False

        # Build consensus analysis
        consensus = {
            "overall_trend": overall_trend,
            "trend_alignment": trend_alignment,
            "near_support": near_support,
            "near_resistance": near_resistance,
            "support_distance": support_distance,
            "resistance_distance": resistance_distance,
            "timeframes_analyzed": [
                tf.name for tf in self.timeframes if self.data_dict[tf] is not None
            ],
            "analysis_time": datetime.now().isoformat(),
        }

        # Overall trading bias
        if overall_trend > 0 and trend_alignment > 0.7:
            if near_resistance:
                consensus["trading_bias"] = "Bullish but near resistance"
            else:
                consensus["trading_bias"] = "Strongly bullish"
        elif overall_trend < 0 and trend_alignment > 0.7:
            if near_support:
                consensus["trading_bias"] = "Bearish but near support"
            else:
                consensus["trading_bias"] = "Strongly bearish"
        elif overall_trend > 0:
            consensus["trading_bias"] = "Moderately bullish"
        elif overall_trend < 0:
            consensus["trading_bias"] = "Moderately bearish"
        else:
            if near_support:
                consensus["trading_bias"] = "Neutral near support"
            elif near_resistance:
                consensus["trading_bias"] = "Neutral near resistance"
            else:
                consensus["trading_bias"] = "Neutral"

        # Add consensus to results
        results["consensus"] = consensus

        return results

    def get_current_timeframe_data(
        self, timeframe: Union[str, Timeframe]
    ) -> Optional[pd.DataFrame]:
        """
        Get current data for a specific timeframe.

        Args:
            timeframe: Timeframe to get data for

        Returns:
            DataFrame for the timeframe or None if not available
        """
        if isinstance(timeframe, str):
            timeframe = Timeframe.from_string(timeframe)

        return self.data_dict.get(timeframe)

    def get_aligned_data(self) -> Dict[Timeframe, pd.DataFrame]:
        """
        Get aligned data for all timeframes.

        Returns:
            Dictionary with aligned DataFrames
        """
        return align_timeframes(self.data_dict)


class MTFAnalysis:
    """Multi-timeframe analysis manager.

    Convenience class for running a complete multi-timeframe analysis.
    """

    def __init__(self, timeframes: List[Union[str, Timeframe]]):
        """
        Initialize multi-timeframe analysis manager.

        Args:
            timeframes: List of timeframes to analyze
        """
        self.aligner = MTFAligner(timeframes)

        # Add default analyzers
        self.aligner.add_analyzer("trend", MTFTrend(timeframes))
        self.aligner.add_analyzer("support_resistance", MTFSupport(timeframes))

    def add_custom_analyzer(self, name: str, analyzer) -> None:
        """
        Add a custom analyzer.

        Args:
            name: Name for the analyzer
            analyzer: Analyzer instance
        """
        self.aligner.add_analyzer(name, analyzer)

    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run multi-timeframe analysis.

        Args:
            data: OHLCV DataFrame for the smallest timeframe

        Returns:
            Dictionary with analysis results
        """
        # Prepare data for all timeframes
        self.aligner.prepare_data(data)

        # Run analysis
        return self.aligner.analyze()

    def get_timeframe_data(
        self, timeframe: Union[str, Timeframe]
    ) -> Optional[pd.DataFrame]:
        """
        Get data for a specific timeframe.

        Args:
            timeframe: Timeframe to get data for

        Returns:
            DataFrame for the timeframe or None if not available
        """
        return self.aligner.get_current_timeframe_data(timeframe)


def run_mtf_analysis(
    data: pd.DataFrame,
    timeframes: List[str] = ["M15", "H1", "H4", "D1"],
    show_details: bool = False,
) -> Dict[str, Any]:
    """
    Run a complete multi-timeframe analysis on the provided data.

    Args:
        data: OHLCV DataFrame for smallest timeframe
        timeframes: List of timeframes to analyze
        show_details: Whether to include detailed analysis results

    Returns:
        Dictionary with analysis results
    """
    mtf = MTFAnalysis(timeframes)
    results = mtf.analyze(data)

    # If details not requested, remove them to make output more concise
    if not show_details:
        for key in list(results.keys()):
            if key != "consensus":
                if "latest_trends" in results[key]:
                    del results[key]["latest_trends"]
                if "timeframe_levels" in results[key]:
                    del results[key]["timeframe_levels"]

    return results

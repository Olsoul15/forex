"""
Pattern Recognition Module for the AI Forex Trading System.

This module implements detection of candlestick and chart patterns commonly used
in technical analysis of forex markets. It provides functions to identify patterns,
score their strength, and analyze them across multiple timeframes.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
from forex_ai.utils.logging import get_logger
from forex_ai.exceptions import PatternAnalysisError

logger = get_logger(__name__)


class PatternType(Enum):
    """Enumeration of supported pattern types."""

    # Candlestick patterns
    DOJI = "doji"
    HAMMER = "hammer"
    SHOOTING_STAR = "shooting_star"
    ENGULFING = "engulfing"
    PIERCING = "piercing"
    DARK_CLOUD_COVER = "dark_cloud_cover"
    MORNING_STAR = "morning_star"
    EVENING_STAR = "evening_star"
    THREE_WHITE_SOLDIERS = "three_white_soldiers"
    THREE_BLACK_CROWS = "three_black_crows"

    # Chart patterns
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    INVERSE_HEAD_AND_SHOULDERS = "inverse_head_and_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    TRIPLE_TOP = "triple_top"
    TRIPLE_BOTTOM = "triple_bottom"
    ASCENDING_TRIANGLE = "ascending_triangle"
    DESCENDING_TRIANGLE = "descending_triangle"
    SYMMETRICAL_TRIANGLE = "symmetrical_triangle"
    RECTANGLE = "rectangle"
    WEDGE = "wedge"
    FLAG = "flag"
    PENNANT = "pennant"


class PatternDirection(Enum):
    """Enumeration of pattern directions (bullish or bearish)."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class PatternResult:
    """Data class for pattern detection results."""

    pattern_type: PatternType
    direction: PatternDirection
    confidence: float  # 0.0 to 1.0
    index: int  # Index position in the dataframe where pattern was detected
    additional_info: Dict[str, Any] = (
        None  # Any additional pattern-specific information
    )

    def __post_init__(self):
        """Validate confidence score."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence score must be between 0.0 and 1.0")

        if self.additional_info is None:
            self.additional_info = {}


def detect_candlestick_patterns(
    data: pd.DataFrame,
    patterns: List[PatternType] = None,
    fib_level: float = 0.333,
    atr_min_filter: float = 0.0,
    atr_max_filter: float = 3.0,
) -> List[PatternResult]:
    """
    Detect candlestick patterns in the provided price data.

    Args:
        data: DataFrame with OHLCV data (must contain 'open', 'high', 'low', 'close', 'volume' columns)
        patterns: List of specific patterns to detect (if None, all patterns are detected)
        fib_level: Fibonacci level used for certain pattern calculations (default: 0.333)
        atr_min_filter: Minimum size of candle relative to ATR (default: 0.0)
        atr_max_filter: Maximum size of candle relative to ATR (default: 3.0)

    Returns:
        List of PatternResult objects containing detected patterns

    Raises:
        PatternAnalysisError: If data is invalid or an error occurs during pattern detection
    """
    try:
        # Validate input data
        required_columns = ["open", "high", "low", "close"]
        if not all(col in data.columns for col in required_columns):
            raise PatternAnalysisError(
                f"Input data missing required columns. Required: {required_columns}, "
                f"Got: {data.columns.tolist()}"
            )

        # Initialize results list
        results = []

        # Select patterns to detect
        all_candlestick_patterns = [
            pattern
            for pattern in PatternType
            if pattern.value
            not in [
                p.value
                for p in PatternType
                if p.value.startswith(
                    (
                        "head",
                        "inverse",
                        "double",
                        "triple",
                        "ascending",
                        "descending",
                        "symmetrical",
                        "rectangle",
                        "wedge",
                        "flag",
                        "pennant",
                    )
                )
            ]
        ]

        patterns_to_detect = patterns if patterns else all_candlestick_patterns

        # Calculate ATR if filters are used
        if atr_min_filter > 0.0 or atr_max_filter < float("inf"):
            atr = calculate_atr(data, period=14)
        else:
            atr = None

        # Detect each pattern
        for pattern in patterns_to_detect:
            if pattern == PatternType.DOJI:
                doji_results = detect_doji(data)
                results.extend(doji_results)

            elif pattern == PatternType.HAMMER or pattern == PatternType.SHOOTING_STAR:
                hammer_star_results = detect_hammer_shooting_star(
                    data, fib_level, atr, atr_min_filter, atr_max_filter
                )
                results.extend(hammer_star_results)

            elif pattern == PatternType.ENGULFING:
                engulfing_results = detect_engulfing(data)
                results.extend(engulfing_results)

            # Add other pattern detection function calls here

        return results

    except Exception as e:
        logger.error(f"Error detecting candlestick patterns: {str(e)}")
        raise PatternAnalysisError(
            f"Failed to detect candlestick patterns: {str(e)}"
        ) from e


def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR) for the given data.

    Args:
        data: DataFrame with OHLC data
        period: Period for ATR calculation

    Returns:
        Pandas Series with ATR values
    """
    try:
        high = data["high"]
        low = data["low"]
        close = data["close"].shift(1)

        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)

        tr = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr
    except Exception as e:
        logger.error(f"Error calculating ATR: {str(e)}")
        raise PatternAnalysisError(f"Failed to calculate ATR: {str(e)}") from e


def detect_doji(
    data: pd.DataFrame, doji_threshold: float = 0.05
) -> List[PatternResult]:
    """
    Detect Doji candlestick patterns in the provided price data.

    A Doji occurs when the opening and closing prices are virtually equal.

    Args:
        data: DataFrame with OHLC data
        doji_threshold: Maximum percentage difference between open and close to be considered a Doji

    Returns:
        List of PatternResult objects for detected Doji patterns
    """
    results = []

    try:
        for i in range(len(data)):
            if i < 1:  # Skip first candle as we need previous candle for context
                continue

            row = data.iloc[i]
            prev_row = data.iloc[i - 1]

            candle_size = abs(row["high"] - row["low"])
            body_size = abs(row["open"] - row["close"])

            # Avoid division by zero
            if candle_size == 0:
                continue

            body_to_candle_ratio = body_size / candle_size

            # Doji detection - body is very small compared to the total range
            if body_to_candle_ratio <= doji_threshold:
                # Determine direction based on previous candle trend
                if prev_row["close"] > prev_row["open"]:  # Previous candle was bullish
                    direction = (
                        PatternDirection.BEARISH
                    )  # Doji after uptrend is potentially bearish
                    confidence = 0.6
                elif (
                    prev_row["close"] < prev_row["open"]
                ):  # Previous candle was bearish
                    direction = (
                        PatternDirection.BULLISH
                    )  # Doji after downtrend is potentially bullish
                    confidence = 0.6
                else:
                    direction = PatternDirection.NEUTRAL
                    confidence = 0.5

                # Adjust confidence based on where Doji appears in the trend
                # Additional logic could be added here

                results.append(
                    PatternResult(
                        pattern_type=PatternType.DOJI,
                        direction=direction,
                        confidence=confidence,
                        index=i,
                        additional_info={
                            "body_to_candle_ratio": body_to_candle_ratio,
                            "body_size": body_size,
                            "candle_size": candle_size,
                        },
                    )
                )

        return results
    except Exception as e:
        logger.error(f"Error detecting Doji patterns: {str(e)}")
        raise PatternAnalysisError(f"Failed to detect Doji patterns: {str(e)}") from e


def detect_hammer_shooting_star(
    data: pd.DataFrame,
    fib_level: float = 0.333,
    atr: Optional[pd.Series] = None,
    atr_min_filter: float = 0.0,
    atr_max_filter: float = 3.0,
) -> List[PatternResult]:
    """
    Detect Hammer and Shooting Star candlestick patterns in the provided price data.

    Implementation based on the Hammers and Stars Pine Script strategy.

    Args:
        data: DataFrame with OHLC data
        fib_level: Fibonacci level used for pattern calculation (default: 0.333)
        atr: Optional pre-calculated ATR Series
        atr_min_filter: Minimum size of candle relative to ATR (default: 0.0)
        atr_max_filter: Maximum size of candle relative to ATR (default: 3.0)

    Returns:
        List of PatternResult objects for detected patterns
    """
    results = []

    try:
        # Calculate ATR if not provided
        if atr is None and (atr_min_filter > 0.0 or atr_max_filter < float("inf")):
            atr = calculate_atr(data, period=14)

        for i in range(len(data)):
            if i < 1:  # Skip first candle as we need previous candle for context
                continue

            row = data.iloc[i]
            prev_row = data.iloc[i - 1]

            # Skip if open equals close (to avoid division by zero later)
            if row["open"] == row["close"]:
                continue

            # Check ATR filter if enabled
            if atr is not None:
                candle_size = abs(row["high"] - row["low"])
                if not (
                    candle_size >= (atr_min_filter * atr[i])
                    and candle_size <= (atr_max_filter * atr[i])
                ):
                    continue

            # Calculate fibonacci levels for the candle
            bull_fib = (row["low"] - row["high"]) * fib_level + row["high"]
            bear_fib = (row["high"] - row["low"]) * fib_level + row["low"]

            # Determine body range
            lowest_body = min(row["close"], row["open"])
            highest_body = max(row["close"], row["open"])

            # Hammer detection
            if lowest_body >= bull_fib:
                # Check if we have more confirmation based on previous candle
                if prev_row["close"] < prev_row["open"]:  # Previous candle was bearish
                    confidence = 0.8  # Higher confidence as it's potentially a reversal
                else:
                    confidence = 0.6

                results.append(
                    PatternResult(
                        pattern_type=PatternType.HAMMER,
                        direction=PatternDirection.BULLISH,
                        confidence=confidence,
                        index=i,
                        additional_info={
                            "fib_level": fib_level,
                            "bull_fib": bull_fib,
                            "candle_size": abs(row["high"] - row["low"]),
                            "body_location": lowest_body
                            - row["low"],  # Distance of body from low
                        },
                    )
                )

            # Shooting Star detection
            if highest_body <= bear_fib:
                # Check if we have more confirmation based on previous candle
                if prev_row["close"] > prev_row["open"]:  # Previous candle was bullish
                    confidence = 0.8  # Higher confidence as it's potentially a reversal
                else:
                    confidence = 0.6

                results.append(
                    PatternResult(
                        pattern_type=PatternType.SHOOTING_STAR,
                        direction=PatternDirection.BEARISH,
                        confidence=confidence,
                        index=i,
                        additional_info={
                            "fib_level": fib_level,
                            "bear_fib": bear_fib,
                            "candle_size": abs(row["high"] - row["low"]),
                            "body_location": row["high"]
                            - highest_body,  # Distance of body from high
                        },
                    )
                )

        return results
    except Exception as e:
        logger.error(f"Error detecting Hammer/Shooting Star patterns: {str(e)}")
        raise PatternAnalysisError(
            f"Failed to detect Hammer/Shooting Star patterns: {str(e)}"
        ) from e


def detect_engulfing(data: pd.DataFrame) -> List[PatternResult]:
    """
    Detect Bullish and Bearish Engulfing candlestick patterns.

    Args:
        data: DataFrame with OHLC data

    Returns:
        List of PatternResult objects for detected Engulfing patterns
    """
    results = []

    try:
        for i in range(len(data)):
            if i < 1:  # Skip first candle as we need previous candle
                continue

            current = data.iloc[i]
            previous = data.iloc[i - 1]

            current_body_size = abs(current["close"] - current["open"])
            previous_body_size = abs(previous["close"] - previous["open"])

            # Bullish Engulfing
            if (
                current["close"] > current["open"]  # Current candle is bullish
                and previous["close"] < previous["open"]  # Previous candle is bearish
                and current["open"]
                < previous["close"]  # Current opens below previous close
                and current["close"] > previous["open"]
            ):  # Current closes above previous open

                # Calculate confidence based on size comparison
                confidence = min(
                    0.9, 0.5 + 0.4 * (current_body_size / previous_body_size - 1)
                )

                results.append(
                    PatternResult(
                        pattern_type=PatternType.ENGULFING,
                        direction=PatternDirection.BULLISH,
                        confidence=confidence,
                        index=i,
                        additional_info={
                            "current_body_size": current_body_size,
                            "previous_body_size": previous_body_size,
                            "size_ratio": (
                                current_body_size / previous_body_size
                                if previous_body_size > 0
                                else float("inf")
                            ),
                        },
                    )
                )

            # Bearish Engulfing
            if (
                current["close"] < current["open"]  # Current candle is bearish
                and previous["close"] > previous["open"]  # Previous candle is bullish
                and current["open"]
                > previous["close"]  # Current opens above previous close
                and current["close"] < previous["open"]
            ):  # Current closes below previous open

                # Calculate confidence based on size comparison
                confidence = min(
                    0.9, 0.5 + 0.4 * (current_body_size / previous_body_size - 1)
                )

                results.append(
                    PatternResult(
                        pattern_type=PatternType.ENGULFING,
                        direction=PatternDirection.BEARISH,
                        confidence=confidence,
                        index=i,
                        additional_info={
                            "current_body_size": current_body_size,
                            "previous_body_size": previous_body_size,
                            "size_ratio": (
                                current_body_size / previous_body_size
                                if previous_body_size > 0
                                else float("inf")
                            ),
                        },
                    )
                )

        return results
    except Exception as e:
        logger.error(f"Error detecting Engulfing patterns: {str(e)}")
        raise PatternAnalysisError(
            f"Failed to detect Engulfing patterns: {str(e)}"
        ) from e


def detect_chart_patterns(
    data: pd.DataFrame,
    patterns: List[PatternType] = None,
    lookback_period: int = 50,
    min_pattern_bars: int = 5,
) -> List[PatternResult]:
    """
    Detect chart patterns in the provided price data.

    Args:
        data: DataFrame with OHLCV data
        patterns: List of specific patterns to detect (if None, all patterns are detected)
        lookback_period: Maximum number of bars to look back for pattern formation
        min_pattern_bars: Minimum number of bars required to form a valid pattern

    Returns:
        List of PatternResult objects containing detected patterns

    Raises:
        PatternAnalysisError: If data is invalid or an error occurs during pattern detection
    """
    try:
        # Validate input data
        required_columns = ["open", "high", "low", "close"]
        if not all(col in data.columns for col in required_columns):
            raise PatternAnalysisError(
                f"Input data missing required columns. Required: {required_columns}, "
                f"Got: {data.columns.tolist()}"
            )

        # Initialize results list
        results = []

        # Select patterns to detect
        all_chart_patterns = [
            pattern
            for pattern in PatternType
            if pattern.value
            in [
                "head_and_shoulders",
                "inverse_head_and_shoulders",
                "double_top",
                "double_bottom",
                "triple_top",
                "triple_bottom",
                "ascending_triangle",
                "descending_triangle",
                "symmetrical_triangle",
                "rectangle",
                "wedge",
                "flag",
                "pennant",
            ]
        ]

        patterns_to_detect = patterns if patterns else all_chart_patterns

        # Implement individual pattern detection methods
        # These would be added as the pattern recognition module is expanded

        return results

    except Exception as e:
        logger.error(f"Error detecting chart patterns: {str(e)}")
        raise PatternAnalysisError(f"Failed to detect chart patterns: {str(e)}") from e


def analyze_multi_timeframe(
    timeframes: Dict[str, pd.DataFrame],
    patterns: List[PatternType] = None,
    fib_level: float = 0.333,
    atr_min_filter: float = 0.0,
    atr_max_filter: float = 3.0,
) -> Dict[str, List[PatternResult]]:
    """
    Analyze patterns across multiple timeframes to find confluent signals.

    Args:
        timeframes: Dictionary mapping timeframe names to corresponding dataframes
        patterns: List of patterns to detect (if None, all patterns are detected)
        fib_level: Fibonacci level for pattern calculation
        atr_min_filter: Minimum ATR filter
        atr_max_filter: Maximum ATR filter

    Returns:
        Dictionary mapping timeframe names to lists of detected patterns
    """
    results = {}

    try:
        for timeframe, data in timeframes.items():
            # Detect candlestick patterns
            candlestick_results = detect_candlestick_patterns(
                data, patterns, fib_level, atr_min_filter, atr_max_filter
            )

            # Detect chart patterns
            chart_results = detect_chart_patterns(data, patterns)

            # Combine results
            results[timeframe] = candlestick_results + chart_results

        return results
    except Exception as e:
        logger.error(f"Error analyzing multi-timeframe patterns: {str(e)}")
        raise PatternAnalysisError(
            f"Failed to analyze multi-timeframe patterns: {str(e)}"
        ) from e


def score_pattern_confluence(
    multi_tf_results: Dict[str, List[PatternResult]],
) -> List[Dict]:
    """
    Score the confluence of patterns across multiple timeframes.

    Args:
        multi_tf_results: Dictionary mapping timeframe names to lists of pattern results

    Returns:
        List of dictionaries containing consolidated pattern signals with confluence scores
    """
    signals = []

    try:
        # Group patterns by type and direction
        grouped_patterns = {}
        for timeframe, patterns in multi_tf_results.items():
            for pattern in patterns:
                key = (pattern.pattern_type, pattern.direction)
                if key not in grouped_patterns:
                    grouped_patterns[key] = []
                grouped_patterns[key].append((timeframe, pattern))

        # Score each group
        for (pattern_type, direction), occurrences in grouped_patterns.items():
            # Base score starts at the highest confidence in the group
            base_confidence = max(p[1].confidence for p in occurrences)

            # Increase score based on number of timeframes where pattern appears
            timeframes = set(tf for tf, _ in occurrences)
            tf_bonus = min(
                0.3, len(timeframes) * 0.1
            )  # Up to 0.3 bonus for timeframe confluence

            # Final score (capped at 0.95)
            confidence = min(0.95, base_confidence + tf_bonus)

            signals.append(
                {
                    "pattern_type": pattern_type.value,
                    "direction": direction.value,
                    "confidence": confidence,
                    "timeframes": list(timeframes),
                    "occurrences": len(occurrences),
                }
            )

        # Sort signals by confidence (highest first)
        signals.sort(key=lambda x: x["confidence"], reverse=True)

        return signals
    except Exception as e:
        logger.error(f"Error scoring pattern confluence: {str(e)}")
        raise PatternAnalysisError(
            f"Failed to score pattern confluence: {str(e)}"
        ) from e

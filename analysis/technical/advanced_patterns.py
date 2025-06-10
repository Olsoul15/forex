"""
Advanced Pattern Recognition Module for the AI Forex Trading System.

This module implements detection of complex chart patterns including
harmonic patterns and Elliott Wave analysis.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
from scipy import signal

from forex_ai.utils.logging import get_logger

logger = get_logger(__name__)


class HarmonicPatternType(Enum):
    """Enumeration of harmonic pattern types."""

    GARTLEY = "gartley"
    BUTTERFLY = "butterfly"
    BAT = "bat"
    CRAB = "crab"
    SHARK = "shark"
    CYPHER = "cypher"


class ElliottWaveType(Enum):
    """Enumeration of Elliott Wave patterns."""

    IMPULSE = "impulse_wave"
    CORRECTIVE = "corrective_wave"
    DIAGONAL = "diagonal_triangle"


class PatternDirection(Enum):
    """Pattern direction - bullish or bearish."""

    BULLISH = "bullish"
    BEARISH = "bearish"


@dataclass
class PatternPoint:
    """Data class representing a point in a harmonic pattern."""

    index: int
    price: float
    label: str  # X, A, B, C, D, etc.


@dataclass
class HarmonicPattern:
    """Data class for a detected harmonic pattern."""

    pattern_type: HarmonicPatternType
    direction: PatternDirection
    points: List[PatternPoint]
    completion_percentage: float  # 0.0 to 1.0
    potential_reversal_zone: Tuple[float, float]
    confidence: float  # 0.0 to 1.0


@dataclass
class ElliottWavePattern:
    """Data class for a detected Elliott Wave pattern."""

    wave_type: ElliottWaveType
    direction: PatternDirection
    wave_points: List[PatternPoint]
    wave_count: int  # Which wave we're in (1-5 for impulse, A-C for corrective)
    confidence: float  # 0.0 to 1.0


def detect_swings(
    data: pd.DataFrame,
    method: str = "zigzag",
    threshold: float = 0.05,
    min_bars: int = 5,
) -> List[PatternPoint]:
    """
    Detect swing highs and lows in price data.

    Args:
        data: DataFrame with OHLC data
        method: Method to use for swing detection ('zigzag', 'peaks', 'fractals')
        threshold: Minimum price movement threshold (as fraction)
        min_bars: Minimum number of bars between swings

    Returns:
        List of PatternPoint objects representing swing points
    """
    if len(data) < min_bars * 2:
        return []

    swing_points = []

    if method == "zigzag":
        # ZigZag implementation
        highs = data["high"].values
        lows = data["low"].values
        close_prices = data["close"].values

        # Initial reference points
        is_uptrend = True
        ref_idx = 0
        ref_price = lows[0]

        for i in range(1, len(data)):
            if is_uptrend:
                # Looking for higher highs
                if highs[i] > ref_price + (ref_price * threshold):
                    # Found a significant higher high
                    if i - ref_idx >= min_bars:
                        swing_points.append(
                            PatternPoint(index=ref_idx, price=ref_price, label="Low")
                        )
                        ref_idx = i
                        ref_price = highs[i]
                        is_uptrend = False
                # Check for lower lows to potentially switch trend
                elif lows[i] < ref_price - (ref_price * threshold):
                    ref_idx = i
                    ref_price = lows[i]
            else:
                # Looking for lower lows
                if lows[i] < ref_price - (ref_price * threshold):
                    # Found a significant lower low
                    if i - ref_idx >= min_bars:
                        swing_points.append(
                            PatternPoint(index=ref_idx, price=ref_price, label="High")
                        )
                        ref_idx = i
                        ref_price = lows[i]
                        is_uptrend = True
                # Check for higher highs to potentially switch trend
                elif highs[i] > ref_price + (ref_price * threshold):
                    ref_idx = i
                    ref_price = highs[i]

    elif method == "peaks":
        # Use scipy's find_peaks for swing detection
        highs = data["high"].values
        lows = data["low"].values

        # Find peaks (swing highs)
        high_peaks, _ = signal.find_peaks(highs, distance=min_bars)
        for idx in high_peaks:
            swing_points.append(PatternPoint(index=idx, price=highs[idx], label="High"))

        # Find troughs (swing lows)
        inv_lows = -1 * lows
        low_peaks, _ = signal.find_peaks(inv_lows, distance=min_bars)
        for idx in low_peaks:
            swing_points.append(PatternPoint(index=idx, price=lows[idx], label="Low"))

    elif method == "fractals":
        # Williams' Fractals implementation
        for i in range(2, len(data) - 2):
            # Bull fractal (swing low)
            if (
                data["low"].iloc[i] < data["low"].iloc[i - 1]
                and data["low"].iloc[i] < data["low"].iloc[i - 2]
                and data["low"].iloc[i] < data["low"].iloc[i + 1]
                and data["low"].iloc[i] < data["low"].iloc[i + 2]
            ):
                swing_points.append(
                    PatternPoint(index=i, price=data["low"].iloc[i], label="Low")
                )

            # Bear fractal (swing high)
            if (
                data["high"].iloc[i] > data["high"].iloc[i - 1]
                and data["high"].iloc[i] > data["high"].iloc[i - 2]
                and data["high"].iloc[i] > data["high"].iloc[i + 1]
                and data["high"].iloc[i] > data["high"].iloc[i + 2]
            ):
                swing_points.append(
                    PatternPoint(index=i, price=data["high"].iloc[i], label="High")
                )

    # Sort by index
    swing_points.sort(key=lambda x: x.index)

    # Alternate labeling based on sequence
    if len(swing_points) > 1:
        is_high = swing_points[0].label == "High"
        for i, point in enumerate(swing_points):
            if i % 2 == 0 and is_high:
                point.label = "High"
            elif i % 2 == 1 and is_high:
                point.label = "Low"
            elif i % 2 == 0 and not is_high:
                point.label = "Low"
            else:
                point.label = "High"

    return swing_points


def detect_harmonic_patterns(
    data: pd.DataFrame,
    swing_points: Optional[List[PatternPoint]] = None,
    pattern_types: Optional[List[HarmonicPatternType]] = None,
    tolerance: float = 0.05,  # Tolerance for Fibonacci ratios
) -> List[HarmonicPattern]:
    """
    Detect harmonic patterns in price data.

    Args:
        data: DataFrame with OHLC data
        swing_points: Pre-calculated swing points (if None, will be detected)
        pattern_types: Specific harmonic patterns to detect
        tolerance: Tolerance for Fibonacci ratio matching

    Returns:
        List of HarmonicPattern objects detected in the data
    """
    if len(data) < 20:
        return []

    # Default to all pattern types if none specified
    if pattern_types is None:
        pattern_types = list(HarmonicPatternType)

    # Detect swing points if not provided
    if swing_points is None:
        swing_points = detect_swings(data)

    if len(swing_points) < 5:  # Need at least 5 points for a complete pattern
        return []

    patterns = []

    # Fibonacci ratios
    fib_ratios = {
        "gartley": {
            "XA": 1.0,
            "AB": 0.618,  # 61.8% retracement of XA
            "BC": 0.382,  # 38.2% retracement of AB
            "CD": 1.272,  # 127.2% extension of BC
        },
        "butterfly": {
            "XA": 1.0,
            "AB": 0.786,  # 78.6% retracement of XA
            "BC": 0.382,  # 38.2% retracement of AB
            "CD": 1.618,  # 161.8% extension of BC
        },
        "bat": {
            "XA": 1.0,
            "AB": 0.382,  # 38.2% retracement of XA
            "BC": 0.382,  # 38.2% retracement of AB
            "CD": 1.618,  # 161.8% extension of BC
        },
        "crab": {
            "XA": 1.0,
            "AB": 0.382,  # 38.2% retracement of XA
            "BC": 0.382,  # 38.2% retracement of AB
            "CD": 2.618,  # 261.8% extension of BC
        },
        "shark": {
            "XA": 1.0,
            "AB": 0.886,  # 88.6% retracement of XA
            "BC": 1.13,  # 113% retracement of AB
            "CD": 1.618,  # 161.8% extension of BC
        },
        "cypher": {
            "XA": 1.0,
            "AB": 0.382,  # 38.2% retracement of XA
            "BC": 1.272,  # 127.2% extension of AB
            "CD": 0.786,  # 78.6% retracement of BC
        },
    }

    # Iterate through possible combinations of 5 consecutive points
    for i in range(len(swing_points) - 4):
        points = swing_points[i : i + 5]

        # Label points as X, A, B, C, D
        x, a, b, c, d = points

        # Calculate price movements
        xa_move = abs(a.price - x.price)
        ab_move = abs(b.price - a.price)
        bc_move = abs(c.price - b.price)
        cd_move = abs(d.price - c.price)

        # Check for pattern matches
        for pattern_type in pattern_types:
            pattern_name = pattern_type.value

            if pattern_name not in fib_ratios:
                continue

            ratios = fib_ratios[pattern_name]

            # Check if the pattern fits within tolerance
            ab_ratio = ab_move / xa_move
            bc_ratio = bc_move / ab_move
            cd_ratio = cd_move / bc_move

            is_valid = (
                abs(ab_ratio - ratios["AB"]) <= tolerance
                and abs(bc_ratio - ratios["BC"]) <= tolerance
                and abs(cd_ratio - ratios["CD"]) <= tolerance
            )

            if is_valid:
                # Calculate direction based on final leg
                direction = (
                    PatternDirection.BULLISH
                    if d.price > c.price
                    else PatternDirection.BEARISH
                )

                # Calculate potential reversal zone
                if direction == PatternDirection.BULLISH:
                    prz_low = d.price
                    prz_high = d.price * (1 + 0.05)  # 5% above D point
                else:
                    prz_low = d.price * (1 - 0.05)  # 5% below D point
                    prz_high = d.price

                # Calculate confidence based on pattern clarity
                confidence = (
                    1.0
                    - (
                        abs(ab_ratio - ratios["AB"]) / ratios["AB"]
                        + abs(bc_ratio - ratios["BC"]) / ratios["BC"]
                        + abs(cd_ratio - ratios["CD"]) / ratios["CD"]
                    )
                    / 3
                )

                pattern = HarmonicPattern(
                    pattern_type=HarmonicPatternType(pattern_name),
                    direction=direction,
                    points=[x, a, b, c, d],
                    completion_percentage=1.0,  # Fully completed pattern
                    potential_reversal_zone=(prz_low, prz_high),
                    confidence=confidence,
                )

                patterns.append(pattern)

    return patterns


def detect_elliott_waves(
    data: pd.DataFrame,
    swing_points: Optional[List[PatternPoint]] = None,
    min_confidence: float = 0.7,
) -> List[ElliottWavePattern]:
    """
    Detect Elliott Wave patterns in price data.

    Args:
        data: DataFrame with OHLC data
        swing_points: Pre-calculated swing points (if None, will be detected)
        min_confidence: Minimum confidence threshold for pattern detection

    Returns:
        List of ElliottWavePattern objects detected in the data
    """
    if len(data) < 30:  # Need sufficient data for Elliott Wave analysis
        return []

    # Detect swing points if not provided
    if swing_points is None:
        swing_points = detect_swings(data)

    if len(swing_points) < 9:  # Need at least 9 points for a 5-3 Elliott wave sequence
        return []

    patterns = []

    # Define Elliott Wave rules
    # 1. Wave 2 cannot retrace more than 100% of Wave 1
    # 2. Wave 3 cannot be the shortest of waves 1, 3, 5
    # 3. Wave 4 cannot overlap Wave 1

    # Scan for potential 5-wave impulse patterns
    for i in range(len(swing_points) - 8):
        # Check if we have alternating high/low points
        points = swing_points[i : i + 9]

        # Check if the sequence starts with the right type (high or low)
        if not (
            (points[0].label == "Low" and points[1].label == "High")
            or (points[0].label == "High" and points[1].label == "Low")
        ):
            continue

        # Label potential wave points
        if points[0].label == "Low":  # Bullish impulse
            wave_0 = points[0]  # Start
            wave_1 = points[1]  # High
            wave_2 = points[2]  # Low
            wave_3 = points[3]  # High
            wave_4 = points[4]  # Low
            wave_5 = points[5]  # High
            wave_a = points[6]  # Low
            wave_b = points[7]  # High
            wave_c = points[8]  # Low
            direction = PatternDirection.BULLISH
        else:  # Bearish impulse
            wave_0 = points[0]  # Start
            wave_1 = points[1]  # Low
            wave_2 = points[2]  # High
            wave_3 = points[3]  # Low
            wave_4 = points[4]  # High
            wave_5 = points[5]  # Low
            wave_a = points[6]  # High
            wave_b = points[7]  # Low
            wave_c = points[8]  # High
            direction = PatternDirection.BEARISH

        # Check Rule 1: Wave 2 cannot retrace more than 100% of Wave 1
        if direction == PatternDirection.BULLISH:
            wave_1_move = wave_1.price - wave_0.price
            wave_2_retracement = wave_1.price - wave_2.price
            if wave_2_retracement > wave_1_move:
                continue
        else:
            wave_1_move = wave_0.price - wave_1.price
            wave_2_retracement = wave_2.price - wave_1.price
            if wave_2_retracement > wave_1_move:
                continue

        # Check Rule 2: Wave 3 cannot be the shortest of waves 1, 3, 5
        if direction == PatternDirection.BULLISH:
            wave_1_length = wave_1.price - wave_0.price
            wave_3_length = wave_3.price - wave_2.price
            wave_5_length = wave_5.price - wave_4.price
            if wave_3_length < wave_1_length and wave_3_length < wave_5_length:
                continue
        else:
            wave_1_length = wave_0.price - wave_1.price
            wave_3_length = wave_2.price - wave_3.price
            wave_5_length = wave_4.price - wave_5.price
            if wave_3_length < wave_1_length and wave_3_length < wave_5_length:
                continue

        # Check Rule 3: Wave 4 cannot overlap Wave 1
        if direction == PatternDirection.BULLISH:
            if wave_4.price < wave_1.price:
                continue
        else:
            if wave_4.price > wave_1.price:
                continue

        # Check a-b-c corrective wave formation
        if direction == PatternDirection.BULLISH:
            # In a bullish impulse, correction should be downward
            if not (wave_a.price < wave_5.price and wave_c.price < wave_b.price):
                continue
        else:
            # In a bearish impulse, correction should be upward
            if not (wave_a.price > wave_5.price and wave_c.price > wave_b.price):
                continue

        # Calculate pattern confidence
        # Based on wave proportions and adherence to Elliott Wave rules
        confidence_factors = []

        # Ideal Wave 3 is 1.618 * Wave 1
        if direction == PatternDirection.BULLISH:
            ideal_wave3 = wave_2.price + (wave_1_length * 1.618)
            wave3_score = 1 - min(abs(wave_3.price - ideal_wave3) / ideal_wave3, 0.5)
        else:
            ideal_wave3 = wave_2.price - (wave_1_length * 1.618)
            wave3_score = 1 - min(abs(wave_3.price - ideal_wave3) / ideal_wave3, 0.5)

        confidence_factors.append(wave3_score)

        # Wave 4 typically retraces 38.2% of Wave 3
        if direction == PatternDirection.BULLISH:
            ideal_wave4 = wave_3.price - (wave_3_length * 0.382)
            wave4_score = 1 - min(abs(wave_4.price - ideal_wave4) / ideal_wave4, 0.5)
        else:
            ideal_wave4 = wave_3.price + (wave_3_length * 0.382)
            wave4_score = 1 - min(abs(wave_4.price - ideal_wave4) / ideal_wave4, 0.5)

        confidence_factors.append(wave4_score)

        # Check proportions in the correction waves (a-b-c)
        if direction == PatternDirection.BULLISH:
            wave_a_length = wave_5.price - wave_a.price
            wave_b_length = wave_b.price - wave_a.price
            wave_c_length = wave_b.price - wave_c.price

            # Wave B typically retraces 61.8% of Wave A
            ideal_wave_b = wave_a.price + (wave_a_length * 0.618)
            wave_b_score = 1 - min(abs(wave_b.price - ideal_wave_b) / ideal_wave_b, 0.5)

            # Wave C is often equal to Wave A
            ideal_wave_c = wave_b.price - wave_a_length
            wave_c_score = 1 - min(abs(wave_c.price - ideal_wave_c) / ideal_wave_c, 0.5)
        else:
            wave_a_length = wave_a.price - wave_5.price
            wave_b_length = wave_a.price - wave_b.price
            wave_c_length = wave_c.price - wave_b.price

            # Wave B typically retraces 61.8% of Wave A
            ideal_wave_b = wave_a.price - (wave_a_length * 0.618)
            wave_b_score = 1 - min(abs(wave_b.price - ideal_wave_b) / ideal_wave_b, 0.5)

            # Wave C is often equal to Wave A
            ideal_wave_c = wave_b.price + wave_a_length
            wave_c_score = 1 - min(abs(wave_c.price - ideal_wave_c) / ideal_wave_c, 0.5)

        confidence_factors.append(wave_b_score)
        confidence_factors.append(wave_c_score)

        # Calculate overall confidence
        confidence = sum(confidence_factors) / len(confidence_factors)

        if confidence >= min_confidence:
            # Determine pattern type
            wave_type = ElliottWaveType.IMPULSE

            # Create pattern object
            pattern = ElliottWavePattern(
                wave_type=wave_type,
                direction=direction,
                wave_points=[
                    wave_0,
                    wave_1,
                    wave_2,
                    wave_3,
                    wave_4,
                    wave_5,
                    wave_a,
                    wave_b,
                    wave_c,
                ],
                wave_count=5,  # Complete impulse pattern
                confidence=confidence,
            )

            patterns.append(pattern)

    return patterns

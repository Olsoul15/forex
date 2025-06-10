"""
Technical indicators for the Forex AI Trading System.

This module provides functions to calculate various technical indicators
used in forex analysis and trading strategies.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any, Callable

from forex_ai.exceptions import IndicatorError, InvalidDataError

logger = logging.getLogger(__name__)


def validate_dataframe(df: pd.DataFrame, required_columns: List[str] = None) -> bool:
    """
    Validate that a dataframe has the required columns for indicator calculation.

    Args:
        df: DataFrame containing market data.
        required_columns: List of required columns. If None, checks for OHLCV columns.

    Returns:
        True if the DataFrame is valid.

    Raises:
        InvalidDataError: If the DataFrame is invalid.
    """
    if df is None or df.empty:
        raise InvalidDataError("DataFrame is empty or None")

    if required_columns is None:
        required_columns = ["open", "high", "low", "close"]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise InvalidDataError(f"DataFrame missing required columns: {missing_columns}")

    return True


def simple_moving_average(
    df: pd.DataFrame,
    period: int = 20,
    column: str = "close",
    result_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Calculate Simple Moving Average (SMA).

    Args:
        df: DataFrame containing market data.
        period: Number of periods to average.
        column: Column to calculate SMA on.
        result_column: Column name for the result. If None, uses 'sma_{period}'.

    Returns:
        DataFrame with SMA column added.

    Raises:
        IndicatorError: If calculation fails.
        InvalidDataError: If DataFrame is invalid.
    """
    try:
        validate_dataframe(df, [column])

        if period <= 0:
            raise ValueError(f"Period must be positive, got {period}")

        if result_column is None:
            result_column = f"sma_{period}"

        df_copy = df.copy()
        df_copy[result_column] = df_copy[column].rolling(window=period).mean()

        return df_copy
    except Exception as e:
        if isinstance(e, InvalidDataError):
            raise
        error_msg = f"Error calculating SMA: {str(e)}"
        logger.error(error_msg)
        raise IndicatorError(error_msg) from e


def exponential_moving_average(
    df: pd.DataFrame,
    period: int = 20,
    column: str = "close",
    result_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Calculate Exponential Moving Average (EMA).

    Args:
        df: DataFrame containing market data.
        period: Number of periods for EMA calculation.
        column: Column to calculate EMA on.
        result_column: Column name for the result. If None, uses 'ema_{period}'.

    Returns:
        DataFrame with EMA column added.

    Raises:
        IndicatorError: If calculation fails.
        InvalidDataError: If DataFrame is invalid.
    """
    try:
        validate_dataframe(df, [column])

        if period <= 0:
            raise ValueError(f"Period must be positive, got {period}")

        if result_column is None:
            result_column = f"ema_{period}"

        df_copy = df.copy()
        df_copy[result_column] = df_copy[column].ewm(span=period, adjust=False).mean()

        return df_copy
    except Exception as e:
        if isinstance(e, InvalidDataError):
            raise
        error_msg = f"Error calculating EMA: {str(e)}"
        logger.error(error_msg)
        raise IndicatorError(error_msg) from e


def bollinger_bands(
    df: pd.DataFrame,
    period: int = 20,
    std_dev: float = 2.0,
    column: str = "close",
    result_upper: Optional[str] = None,
    result_middle: Optional[str] = None,
    result_lower: Optional[str] = None,
) -> pd.DataFrame:
    """
    Calculate Bollinger Bands.

    Args:
        df: DataFrame containing market data.
        period: Number of periods for moving average calculation.
        std_dev: Number of standard deviations for upper/lower bands.
        column: Column to calculate Bollinger Bands on.
        result_upper: Column name for the upper band. If None, uses 'bb_upper_{period}'.
        result_middle: Column name for the middle band. If None, uses 'bb_middle_{period}'.
        result_lower: Column name for the lower band. If None, uses 'bb_lower_{period}'.

    Returns:
        DataFrame with Bollinger Bands columns added.

    Raises:
        IndicatorError: If calculation fails.
        InvalidDataError: If DataFrame is invalid.
    """
    try:
        validate_dataframe(df, [column])

        if period <= 0:
            raise ValueError(f"Period must be positive, got {period}")

        if result_upper is None:
            result_upper = f"bb_upper_{period}"
        if result_middle is None:
            result_middle = f"bb_middle_{period}"
        if result_lower is None:
            result_lower = f"bb_lower_{period}"

        df_copy = df.copy()

        # Calculate middle band (SMA)
        middle_band = df_copy[column].rolling(window=period).mean()

        # Calculate standard deviation
        std = df_copy[column].rolling(window=period).std()

        # Calculate upper and lower bands
        df_copy[result_upper] = middle_band + (std_dev * std)
        df_copy[result_middle] = middle_band
        df_copy[result_lower] = middle_band - (std_dev * std)

        return df_copy
    except Exception as e:
        if isinstance(e, InvalidDataError):
            raise
        error_msg = f"Error calculating Bollinger Bands: {str(e)}"
        logger.error(error_msg)
        raise IndicatorError(error_msg) from e


def relative_strength_index(
    df: pd.DataFrame,
    period: int = 14,
    column: str = "close",
    result_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Calculate Relative Strength Index (RSI).

    Args:
        df: DataFrame containing market data.
        period: Number of periods for RSI calculation.
        column: Column to calculate RSI on.
        result_column: Column name for the result. If None, uses 'rsi_{period}'.

    Returns:
        DataFrame with RSI column added.

    Raises:
        IndicatorError: If calculation fails.
        InvalidDataError: If DataFrame is invalid.
    """
    try:
        validate_dataframe(df, [column])

        if period <= 0:
            raise ValueError(f"Period must be positive, got {period}")

        if result_column is None:
            result_column = f"rsi_{period}"

        df_copy = df.copy()

        # Calculate price change
        delta = df_copy[column].diff()

        # Create gain (up) and loss (down) series
        gain = delta.copy()
        loss = delta.copy()

        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = abs(loss)

        # Calculate average gain and average loss
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        df_copy[result_column] = 100 - (100 / (1 + rs))

        return df_copy
    except Exception as e:
        if isinstance(e, InvalidDataError):
            raise
        if isinstance(e, ZeroDivisionError):
            error_msg = (
                "Error calculating RSI: Division by zero (no losses in the period)"
            )
            logger.error(error_msg)
            raise IndicatorError(error_msg) from e

        error_msg = f"Error calculating RSI: {str(e)}"
        logger.error(error_msg)
        raise IndicatorError(error_msg) from e


def moving_average_convergence_divergence(
    df: pd.DataFrame,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    column: str = "close",
    result_macd: Optional[str] = None,
    result_signal: Optional[str] = None,
    result_histogram: Optional[str] = None,
) -> pd.DataFrame:
    """
    Calculate Moving Average Convergence Divergence (MACD).

    Args:
        df: DataFrame containing market data.
        fast_period: Number of periods for fast EMA.
        slow_period: Number of periods for slow EMA.
        signal_period: Number of periods for signal line.
        column: Column to calculate MACD on.
        result_macd: Column name for MACD. If None, uses 'macd'.
        result_signal: Column name for signal line. If None, uses 'macd_signal'.
        result_histogram: Column name for histogram. If None, uses 'macd_histogram'.

    Returns:
        DataFrame with MACD columns added.

    Raises:
        IndicatorError: If calculation fails.
        InvalidDataError: If DataFrame is invalid.
    """
    try:
        validate_dataframe(df, [column])

        if fast_period <= 0 or slow_period <= 0 or signal_period <= 0:
            raise ValueError("Periods must be positive")

        if fast_period >= slow_period:
            raise ValueError("Fast period must be smaller than slow period")

        if result_macd is None:
            result_macd = "macd"
        if result_signal is None:
            result_signal = "macd_signal"
        if result_histogram is None:
            result_histogram = "macd_histogram"

        df_copy = df.copy()

        # Calculate fast and slow EMAs
        df_copy["ema_fast"] = df_copy[column].ewm(span=fast_period, adjust=False).mean()
        df_copy["ema_slow"] = df_copy[column].ewm(span=slow_period, adjust=False).mean()

        # Calculate MACD line
        df_copy[result_macd] = df_copy["ema_fast"] - df_copy["ema_slow"]

        # Calculate signal line
        df_copy[result_signal] = (
            df_copy[result_macd].ewm(span=signal_period, adjust=False).mean()
        )

        # Calculate histogram
        df_copy[result_histogram] = df_copy[result_macd] - df_copy[result_signal]

        # Drop temporary columns
        df_copy = df_copy.drop(["ema_fast", "ema_slow"], axis=1)

        return df_copy
    except Exception as e:
        if isinstance(e, InvalidDataError):
            raise
        error_msg = f"Error calculating MACD: {str(e)}"
        logger.error(error_msg)
        raise IndicatorError(error_msg) from e


def average_true_range(
    df: pd.DataFrame, period: int = 14, result_column: Optional[str] = None
) -> pd.DataFrame:
    """
    Calculate Average True Range (ATR).

    Args:
        df: DataFrame containing market data.
        period: Number of periods for ATR calculation.
        result_column: Column name for the result. If None, uses 'atr_{period}'.

    Returns:
        DataFrame with ATR column added.

    Raises:
        IndicatorError: If calculation fails.
        InvalidDataError: If DataFrame is invalid.
    """
    try:
        validate_dataframe(df, ["high", "low", "close"])

        if period <= 0:
            raise ValueError(f"Period must be positive, got {period}")

        if result_column is None:
            result_column = f"atr_{period}"

        df_copy = df.copy()

        # Calculate true range
        df_copy["tr1"] = df_copy["high"] - df_copy["low"]
        df_copy["tr2"] = abs(df_copy["high"] - df_copy["close"].shift())
        df_copy["tr3"] = abs(df_copy["low"] - df_copy["close"].shift())
        df_copy["tr"] = df_copy[["tr1", "tr2", "tr3"]].max(axis=1)

        # Calculate ATR
        df_copy[result_column] = df_copy["tr"].rolling(window=period).mean()

        # Drop temporary columns
        df_copy = df_copy.drop(["tr1", "tr2", "tr3", "tr"], axis=1)

        return df_copy
    except Exception as e:
        if isinstance(e, InvalidDataError):
            raise
        error_msg = f"Error calculating ATR: {str(e)}"
        logger.error(error_msg)
        raise IndicatorError(error_msg) from e


def stochastic_oscillator(
    df: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
    result_k: Optional[str] = None,
    result_d: Optional[str] = None,
) -> pd.DataFrame:
    """
    Calculate Stochastic Oscillator.

    Args:
        df: DataFrame containing market data.
        k_period: Number of periods for %K calculation.
        d_period: Number of periods for %D calculation.
        result_k: Column name for %K. If None, uses 'stoch_k'.
        result_d: Column name for %D. If None, uses 'stoch_d'.

    Returns:
        DataFrame with Stochastic Oscillator columns added.

    Raises:
        IndicatorError: If calculation fails.
        InvalidDataError: If DataFrame is invalid.
    """
    try:
        validate_dataframe(df, ["high", "low", "close"])

        if k_period <= 0 or d_period <= 0:
            raise ValueError("Periods must be positive")

        if result_k is None:
            result_k = "stoch_k"
        if result_d is None:
            result_d = "stoch_d"

        df_copy = df.copy()

        # Calculate %K
        lowest_low = df_copy["low"].rolling(window=k_period).min()
        highest_high = df_copy["high"].rolling(window=k_period).max()
        df_copy[result_k] = 100 * (
            (df_copy["close"] - lowest_low) / (highest_high - lowest_low)
        )

        # Calculate %D (SMA of %K)
        df_copy[result_d] = df_copy[result_k].rolling(window=d_period).mean()

        return df_copy
    except Exception as e:
        if isinstance(e, InvalidDataError):
            raise
        if isinstance(e, ZeroDivisionError):
            error_msg = "Error calculating Stochastic Oscillator: Division by zero (highest high equals lowest low)"
            logger.error(error_msg)
            raise IndicatorError(error_msg) from e

        error_msg = f"Error calculating Stochastic Oscillator: {str(e)}"
        logger.error(error_msg)
        raise IndicatorError(error_msg) from e


def ichimoku_cloud(
    df: pd.DataFrame,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52,
    displacement: int = 26,
) -> pd.DataFrame:
    """
    Calculate Ichimoku Cloud.

    Args:
        df: DataFrame containing market data.
        tenkan_period: Number of periods for Tenkan-sen (Conversion Line).
        kijun_period: Number of periods for Kijun-sen (Base Line).
        senkou_b_period: Number of periods for Senkou Span B.
        displacement: Number of periods for displacement.

    Returns:
        DataFrame with Ichimoku Cloud columns added.

    Raises:
        IndicatorError: If calculation fails.
        InvalidDataError: If DataFrame is invalid.
    """
    try:
        validate_dataframe(df, ["high", "low", "close"])

        if (
            tenkan_period <= 0
            or kijun_period <= 0
            or senkou_b_period <= 0
            or displacement <= 0
        ):
            raise ValueError("Periods must be positive")

        df_copy = df.copy()

        # Calculate Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for the past tenkan_period
        df_copy["tenkan_sen"] = (
            df_copy["high"].rolling(window=tenkan_period).max()
            + df_copy["low"].rolling(window=tenkan_period).min()
        ) / 2

        # Calculate Kijun-sen (Base Line): (highest high + lowest low)/2 for the past kijun_period
        df_copy["kijun_sen"] = (
            df_copy["high"].rolling(window=kijun_period).max()
            + df_copy["low"].rolling(window=kijun_period).min()
        ) / 2

        # Calculate Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2, displaced forward by displacement
        df_copy["senkou_span_a"] = (
            (df_copy["tenkan_sen"] + df_copy["kijun_sen"]) / 2
        ).shift(displacement)

        # Calculate Senkou Span B (Leading Span B): (highest high + lowest low)/2 for the past senkou_b_period, displaced forward by displacement
        df_copy["senkou_span_b"] = (
            (
                df_copy["high"].rolling(window=senkou_b_period).max()
                + df_copy["low"].rolling(window=senkou_b_period).min()
            )
            / 2
        ).shift(displacement)

        # Calculate Chikou Span (Lagging Span): Current closing price, displaced backward by displacement
        df_copy["chikou_span"] = df_copy["close"].shift(-displacement)

        return df_copy
    except Exception as e:
        if isinstance(e, InvalidDataError):
            raise
        error_msg = f"Error calculating Ichimoku Cloud: {str(e)}"
        logger.error(error_msg)
        raise IndicatorError(error_msg) from e


def fibonacci_retracement(
    df: pd.DataFrame,
    high_period: int = 20,
    low_period: int = 20,
    levels: List[float] = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0],
) -> pd.DataFrame:
    """
    Calculate Fibonacci Retracement levels.

    Args:
        df: DataFrame containing market data.
        high_period: Number of periods to look back for highest high.
        low_period: Number of periods to look back for lowest low.
        levels: Fibonacci retracement levels to calculate.

    Returns:
        DataFrame with Fibonacci Retracement levels added.

    Raises:
        IndicatorError: If calculation fails.
        InvalidDataError: If DataFrame is invalid.
    """
    try:
        validate_dataframe(df, ["high", "low"])

        if high_period <= 0 or low_period <= 0:
            raise ValueError("Periods must be positive")

        df_copy = df.copy()

        # Calculate highest high and lowest low
        df_copy["highest_high"] = df_copy["high"].rolling(window=high_period).max()
        df_copy["lowest_low"] = df_copy["low"].rolling(window=low_period).min()

        # Calculate range
        df_copy["range"] = df_copy["highest_high"] - df_copy["lowest_low"]

        # Calculate Fibonacci Retracement levels
        for level in levels:
            column_name = f"fib_{str(level).replace('.', '_')}"
            df_copy[column_name] = df_copy["highest_high"] - df_copy["range"] * level

        # Drop temporary columns
        df_copy = df_copy.drop(["highest_high", "lowest_low", "range"], axis=1)

        return df_copy
    except Exception as e:
        if isinstance(e, InvalidDataError):
            raise
        error_msg = f"Error calculating Fibonacci Retracement: {str(e)}"
        logger.error(error_msg)
        raise IndicatorError(error_msg) from e


def pivot_points(
    df: pd.DataFrame,
    method: str = "standard",
) -> pd.DataFrame:
    """
    Calculate Pivot Points.

    Args:
        df: DataFrame containing market data.
        method: Method for pivot point calculation. Options: 'standard', 'fibonacci', 'woodie', 'camarilla', 'demark'.

    Returns:
        DataFrame with Pivot Points added.

    Raises:
        IndicatorError: If calculation fails.
        InvalidDataError: If DataFrame is invalid.
    """
    try:
        validate_dataframe(df, ["high", "low", "close"])

        methods = ["standard", "fibonacci", "woodie", "camarilla", "demark"]
        if method not in methods:
            raise ValueError(f"Method must be one of {methods}, got {method}")

        df_copy = df.copy()

        # Get previous day's high, low, and close
        prev_high = df_copy["high"].shift(1)
        prev_low = df_copy["low"].shift(1)
        prev_close = df_copy["close"].shift(1)

        if method == "standard":
            # Standard Pivot Point
            pivot = (prev_high + prev_low + prev_close) / 3
            s1 = (2 * pivot) - prev_high
            s2 = pivot - (prev_high - prev_low)
            s3 = s2 - (prev_high - prev_low)
            r1 = (2 * pivot) - prev_low
            r2 = pivot + (prev_high - prev_low)
            r3 = r2 + (prev_high - prev_low)

        elif method == "fibonacci":
            # Fibonacci Pivot Point
            pivot = (prev_high + prev_low + prev_close) / 3
            s1 = pivot - 0.382 * (prev_high - prev_low)
            s2 = pivot - 0.618 * (prev_high - prev_low)
            s3 = pivot - 1.0 * (prev_high - prev_low)
            r1 = pivot + 0.382 * (prev_high - prev_low)
            r2 = pivot + 0.618 * (prev_high - prev_low)
            r3 = pivot + 1.0 * (prev_high - prev_low)

        elif method == "woodie":
            # Woodie Pivot Point
            pivot = (prev_high + prev_low + 2 * df_copy["open"]) / 4
            s1 = (2 * pivot) - prev_high
            s2 = pivot - (prev_high - prev_low)
            s3 = s2 - (prev_high - prev_low)
            r1 = (2 * pivot) - prev_low
            r2 = pivot + (prev_high - prev_low)
            r3 = r2 + (prev_high - prev_low)

        elif method == "camarilla":
            # Camarilla Pivot Point
            pivot = (prev_high + prev_low + prev_close) / 3
            s1 = prev_close - 1.1 * (prev_high - prev_low) / 12
            s2 = prev_close - 1.1 * (prev_high - prev_low) / 6
            s3 = prev_close - 1.1 * (prev_high - prev_low) / 4
            r1 = prev_close + 1.1 * (prev_high - prev_low) / 12
            r2 = prev_close + 1.1 * (prev_high - prev_low) / 6
            r3 = prev_close + 1.1 * (prev_high - prev_low) / 4

        elif method == "demark":
            # DeMark Pivot Point
            pivot = np.where(
                prev_close < df_copy["open"].shift(1),
                prev_high + (2 * prev_low) + prev_close,
                np.where(
                    prev_close > df_copy["open"].shift(1),
                    (2 * prev_high) + prev_low + prev_close,
                    prev_high + prev_low + (2 * prev_close),
                ),
            )
            pivot = pivot / 4
            s1 = pivot * 2 - prev_high
            r1 = pivot * 2 - prev_low
            # DeMark's method only defines pivot, S1, and R1
            s2 = s1 - (r1 - s1) / 2
            s3 = s1 - (r1 - s1)
            r2 = r1 + (r1 - s1) / 2
            r3 = r1 + (r1 - s1)

        # Store results in DataFrame
        df_copy[f"pivot_{method}"] = pivot
        df_copy[f"r1_{method}"] = r1
        df_copy[f"r2_{method}"] = r2
        df_copy[f"r3_{method}"] = r3
        df_copy[f"s1_{method}"] = s1
        df_copy[f"s2_{method}"] = s2
        df_copy[f"s3_{method}"] = s3

        return df_copy
    except Exception as e:
        if isinstance(e, InvalidDataError):
            raise
        error_msg = f"Error calculating Pivot Points: {str(e)}"
        logger.error(error_msg)
        raise IndicatorError(error_msg) from e


def apply_indicators(
    df: pd.DataFrame, indicators: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """
    Apply multiple indicators to a DataFrame.

    Args:
        df: DataFrame containing market data.
        indicators: Dictionary mapping indicator names to parameters dictionaries.
                   Example: {"sma": {"period": 20}, "rsi": {"period": 14}}

    Returns:
        DataFrame with all specified indicators added.

    Raises:
        IndicatorError: If calculation fails.
        InvalidDataError: If DataFrame is invalid.
    """
    indicator_map = {
        "sma": simple_moving_average,
        "ema": exponential_moving_average,
        "bollinger": bollinger_bands,
        "rsi": relative_strength_index,
        "macd": moving_average_convergence_divergence,
        "atr": average_true_range,
        "stochastic": stochastic_oscillator,
        "ichimoku": ichimoku_cloud,
        "fibonacci": fibonacci_retracement,
        "pivot": pivot_points,
    }

    result_df = df.copy()

    for indicator_name, params in indicators.items():
        if indicator_name not in indicator_map:
            logger.warning(f"Unknown indicator: {indicator_name}, skipping")
            continue

        indicator_func = indicator_map[indicator_name]
        result_df = indicator_func(result_df, **params)

    return result_df

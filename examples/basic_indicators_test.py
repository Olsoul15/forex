"""
Basic technical indicators test using historical data.

This script tests the technical indicator calculation capabilities without requiring
agent functionality or pattern recognition.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Sample historical data (simulated OHLC data for EUR/USD)
SAMPLE_DATA = [
    {
        "time": "2023-03-01T00:00:00Z",
        "open": 1.0632,
        "high": 1.0691,
        "low": 1.0622,
        "close": 1.0665,
        "volume": 10241,
    },
    {
        "time": "2023-03-02T00:00:00Z",
        "open": 1.0665,
        "high": 1.0729,
        "low": 1.0653,
        "close": 1.0725,
        "volume": 11452,
    },
    {
        "time": "2023-03-03T00:00:00Z",
        "open": 1.0725,
        "high": 1.0774,
        "low": 1.0697,
        "close": 1.0735,
        "volume": 12553,
    },
    {
        "time": "2023-03-04T00:00:00Z",
        "open": 1.0735,
        "high": 1.0759,
        "low": 1.0682,
        "close": 1.0694,
        "volume": 10874,
    },
    {
        "time": "2023-03-05T00:00:00Z",
        "open": 1.0694,
        "high": 1.0732,
        "low": 1.0662,
        "close": 1.0681,
        "volume": 10365,
    },
    {
        "time": "2023-03-06T00:00:00Z",
        "open": 1.0681,
        "high": 1.0715,
        "low": 1.0657,
        "close": 1.0702,
        "volume": 11246,
    },
    {
        "time": "2023-03-07T00:00:00Z",
        "open": 1.0702,
        "high": 1.0748,
        "low": 1.0685,
        "close": 1.0729,
        "volume": 11877,
    },
    {
        "time": "2023-03-08T00:00:00Z",
        "open": 1.0729,
        "high": 1.0773,
        "low": 1.0695,
        "close": 1.0759,
        "volume": 12538,
    },
    {
        "time": "2023-03-09T00:00:00Z",
        "open": 1.0759,
        "high": 1.0791,
        "low": 1.0726,
        "close": 1.0784,
        "volume": 13029,
    },
    {
        "time": "2023-03-10T00:00:00Z",
        "open": 1.0784,
        "high": 1.0814,
        "low": 1.0753,
        "close": 1.0803,
        "volume": 13450,
    },
    {
        "time": "2023-03-11T00:00:00Z",
        "open": 1.0803,
        "high": 1.0847,
        "low": 1.0784,
        "close": 1.0830,
        "volume": 13961,
    },
    {
        "time": "2023-03-12T00:00:00Z",
        "open": 1.0830,
        "high": 1.0856,
        "low": 1.0797,
        "close": 1.0842,
        "volume": 13512,
    },
    {
        "time": "2023-03-13T00:00:00Z",
        "open": 1.0842,
        "high": 1.0863,
        "low": 1.0795,
        "close": 1.0818,
        "volume": 13103,
    },
    {
        "time": "2023-03-14T00:00:00Z",
        "open": 1.0818,
        "high": 1.0836,
        "low": 1.0741,
        "close": 1.0767,
        "volume": 13274,
    },
    {
        "time": "2023-03-15T00:00:00Z",
        "open": 1.0767,
        "high": 1.0795,
        "low": 1.0710,
        "close": 1.0742,
        "volume": 13425,
    },
    {
        "time": "2023-03-16T00:00:00Z",
        "open": 1.0742,
        "high": 1.0784,
        "low": 1.0668,
        "close": 1.0689,
        "volume": 14696,
    },
    {
        "time": "2023-03-17T00:00:00Z",
        "open": 1.0689,
        "high": 1.0732,
        "low": 1.0639,
        "close": 1.0724,
        "volume": 14927,
    },
    {
        "time": "2023-03-18T00:00:00Z",
        "open": 1.0724,
        "high": 1.0768,
        "low": 1.0697,
        "close": 1.0742,
        "volume": 13838,
    },
    {
        "time": "2023-03-19T00:00:00Z",
        "open": 1.0742,
        "high": 1.0792,
        "low": 1.0715,
        "close": 1.0776,
        "volume": 13699,
    },
    {
        "time": "2023-03-20T00:00:00Z",
        "open": 1.0776,
        "high": 1.0825,
        "low": 1.0756,
        "close": 1.0817,
        "volume": 14150,
    },
]


class SimpleIndicatorCalculator:
    """
    Simple calculator for technical indicators.
    """

    def calculate_moving_average(self, prices, period=20):
        """
        Calculate Simple Moving Average (SMA).

        Args:
            prices: List of price values
            period: MA period

        Returns:
            List of MA values
        """
        prices = np.array(prices)
        result = np.zeros_like(prices)
        for i in range(len(prices)):
            if i < period - 1:
                result[i] = np.nan
            else:
                result[i] = np.mean(prices[i - period + 1 : i + 1])
        return result

    def calculate_rsi(self, prices, period=14):
        """
        Calculate Relative Strength Index (RSI).

        Args:
            prices: List of price values
            period: RSI period

        Returns:
            List of RSI values
        """
        prices = np.array(prices)
        deltas = np.diff(prices)
        seed = deltas[: period + 1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else float("inf")
        rsi = np.zeros_like(prices)
        rsi[:period] = np.nan
        rsi[period] = 100.0 - 100.0 / (1.0 + rs)

        for i in range(period + 1, len(prices)):
            delta = deltas[i - 1]
            if delta > 0:
                upval = delta
                downval = 0.0
            else:
                upval = 0.0
                downval = -delta

            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down if down != 0 else float("inf")
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)

        return rsi

    def calculate_macd(self, prices, fast_period=12, slow_period=26, signal_period=9):
        """
        Calculate Moving Average Convergence Divergence (MACD).

        Args:
            prices: List of price values
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal EMA period

        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        prices = np.array(prices)
        # Calculate EMAs
        ema_fast = self.calculate_ema(prices, fast_period)
        ema_slow = self.calculate_ema(prices, slow_period)

        # Calculate MACD line
        macd_line = ema_fast - ema_slow

        # Calculate Signal line
        signal_line = self.calculate_ema(macd_line, signal_period)

        # Calculate histogram
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def calculate_ema(self, prices, period):
        """
        Calculate Exponential Moving Average (EMA).

        Args:
            prices: List of price values
            period: EMA period

        Returns:
            List of EMA values
        """
        prices = np.array(prices)
        result = np.zeros_like(prices)

        # Check if period is valid
        if period > len(prices):
            # Handle case when period is greater than data length
            result[:] = np.nan
            return result

        # Seed with SMA
        result[:period] = np.nan
        result[period - 1] = np.mean(prices[:period])

        # Calculate multiplier
        multiplier = 2.0 / (period + 1)

        # Calculate EMA
        for i in range(period, len(prices)):
            result[i] = prices[i] * multiplier + result[i - 1] * (1 - multiplier)

        return result

    def calculate_bollinger_bands(self, prices, period=20, num_std=2):
        """
        Calculate Bollinger Bands.

        Args:
            prices: List of price values
            period: Bollinger Bands period
            num_std: Number of standard deviations

        Returns:
            Tuple of (middle_band, upper_band, lower_band)
        """
        prices = np.array(prices)

        # Check if period is valid
        if period > len(prices):
            # Return arrays of NaN when period is greater than data length
            result = np.full_like(prices, np.nan)
            return result, result, result

        middle_band = self.calculate_moving_average(prices, period)

        std = np.zeros_like(prices)
        for i in range(len(prices)):
            if i < period - 1:
                std[i] = np.nan
            else:
                std[i] = np.std(prices[i - period + 1 : i + 1])

        upper_band = middle_band + (std * num_std)
        lower_band = middle_band - (std * num_std)

        return middle_band, upper_band, lower_band


def main():
    """Run the basic indicators test."""
    print("Starting basic indicators test...")

    # Convert sample data to DataFrame
    df = pd.DataFrame(SAMPLE_DATA)

    # Extract arrays for analysis
    dates = [
        datetime.fromisoformat(item["time"].replace("Z", "+00:00"))
        for item in SAMPLE_DATA
    ]
    opens = [item["open"] for item in SAMPLE_DATA]
    highs = [item["high"] for item in SAMPLE_DATA]
    lows = [item["low"] for item in SAMPLE_DATA]
    closes = [item["close"] for item in SAMPLE_DATA]
    volumes = [item["volume"] for item in SAMPLE_DATA]

    # Initialize indicator calculator
    calculator = SimpleIndicatorCalculator()

    # Calculate indicators
    print("\nCalculating indicators...")

    # RSI
    rsi = calculator.calculate_rsi(closes, period=14)
    print(f"RSI (latest): {rsi[-1]:.2f}")

    # MACD
    macd, macd_signal, macd_hist = calculator.calculate_macd(
        closes, fast_period=5, slow_period=10, signal_period=3
    )
    print(f"MACD (latest): {macd[-1]:.6f}")
    print(f"MACD Signal (latest): {macd_signal[-1]:.6f}")
    print(f"MACD Histogram (latest): {macd_hist[-1]:.6f}")

    # Moving Averages
    ma_5 = calculator.calculate_moving_average(closes, period=5)
    ma_10 = calculator.calculate_moving_average(closes, period=10)
    print(f"5-period MA (latest): {ma_5[-1]:.4f}")
    print(f"10-period MA (latest): {ma_10[-1]:.4f}")

    # Bollinger Bands
    middle, upper, lower = calculator.calculate_bollinger_bands(
        closes, period=10, num_std=2
    )
    print(f"Bollinger Middle (latest): {middle[-1]:.4f}")
    print(f"Bollinger Upper (latest): {upper[-1]:.4f}")
    print(f"Bollinger Lower (latest): {lower[-1]:.4f}")

    # Simple analysis
    print("\nBasic Analysis:")

    # Trend analysis based on MA crossover
    ma_trend = "Bullish" if ma_5[-1] > ma_10[-1] else "Bearish"
    print(f"MA Trend: {ma_trend}")

    # Momentum analysis based on RSI
    rsi_latest = rsi[-1]
    if np.isnan(rsi_latest):
        momentum = "Unknown"
    elif rsi_latest > 70:
        momentum = "Overbought"
    elif rsi_latest < 30:
        momentum = "Oversold"
    else:
        momentum = "Neutral"
    print(f"RSI Momentum: {momentum}")

    # MACD signal
    if np.isnan(macd_hist[-1]):
        macd_signal_text = "Unknown"
    elif macd_hist[-1] > 0 and macd_hist[-1] > macd_hist[-2]:
        macd_signal_text = "Strong Bullish"
    elif macd_hist[-1] > 0:
        macd_signal_text = "Bullish"
    elif macd_hist[-1] < 0 and macd_hist[-1] < macd_hist[-2]:
        macd_signal_text = "Strong Bearish"
    else:
        macd_signal_text = "Bearish"
    print(f"MACD Signal: {macd_signal_text}")

    # Bollinger Band position
    latest_close = closes[-1]
    if np.isnan(upper[-1]) or np.isnan(lower[-1]):
        bb_position = "Unknown"
    elif latest_close > upper[-1]:
        bb_position = "Above Upper Band (Potentially Overbought)"
    elif latest_close < lower[-1]:
        bb_position = "Below Lower Band (Potentially Oversold)"
    else:
        percent_position = (latest_close - lower[-1]) / (upper[-1] - lower[-1]) * 100
        bb_position = f"Within Bands ({percent_position:.1f}% from lower band)"
    print(f"Bollinger Position: {bb_position}")

    print("\nBasic indicators test completed")

    # Plot indicators
    try:
        plt.figure(figsize=(12, 10))

        # Plot 1: Price and MAs
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(dates, closes, label="Close")
        ax1.plot(dates, ma_5, label="MA(5)")
        ax1.plot(dates, ma_10, label="MA(10)")
        ax1.set_title("EUR/USD Price and Moving Averages")
        ax1.legend()
        ax1.grid(True)

        # Plot 2: RSI
        ax2 = plt.subplot(3, 1, 2)
        ax2.plot(dates, rsi)
        ax2.axhline(y=70, color="r", linestyle="-")
        ax2.axhline(y=30, color="g", linestyle="-")
        ax2.set_title("RSI(14)")
        ax2.grid(True)

        # Plot 3: MACD
        ax3 = plt.subplot(3, 1, 3)
        ax3.plot(dates, macd, label="MACD")
        ax3.plot(dates, macd_signal, label="Signal")
        ax3.bar(dates, macd_hist, label="Histogram")
        ax3.set_title("MACD")
        ax3.legend()
        ax3.grid(True)

        plt.tight_layout()
        plt.savefig("technical_analysis.png")
        print("Chart saved as technical_analysis.png")
    except Exception as e:
        print(f"Error creating chart: {str(e)}")


if __name__ == "__main__":
    main()

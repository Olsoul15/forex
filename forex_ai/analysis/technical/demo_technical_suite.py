#!/usr/bin/env python3
"""
Technical Analysis Suite Demo

This script demonstrates the capabilities of the AI Forex Trading System's
technical analysis modules, including:
- Multi-timeframe analysis
- Advanced pattern recognition
- Deep learning price prediction
- Backtesting with risk-adjusted metrics

The demo generates sample forex data and runs analyses with all components.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import sys
import talib
from typing import Dict, List, Any, Tuple

# Add project root to path
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)

# Import modules
from forex_ai.analysis.technical.multi_timeframe import (
    MTFAnalysis,
    Timeframe,
    run_mtf_analysis,
)
from forex_ai.analysis.technical.advanced_patterns import (
    detect_harmonic_patterns,
    detect_elliott_waves,
    PatternDirection,
)
from forex_ai.analysis.technical.backtesting.engine import Backtest
from forex_ai.analysis.technical.backtesting.example_strategy import MACrossoverStrategy
from forex_ai.analysis.technical.risk_adjusted import RiskAdjustedReturns


def generate_sample_data(
    days: int = 500, volatility: float = 0.01, trend: float = 0.0001, seed: int = 42
) -> pd.DataFrame:
    """
    Generate sample forex OHLCV data with a trend and some patterns.

    Args:
        days: Number of days of data to generate
        volatility: Price volatility (standard deviation of returns)
        trend: Trend strength (drift)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with OHLCV data and DatetimeIndex
    """
    np.random.seed(seed)

    # Generate dates
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    # Generate close prices with a trend and random walk
    close_prices = [100]  # Starting price

    # Create cycles
    cycle1 = np.sin(np.linspace(0, 4 * np.pi, days))  # Two complete cycles
    cycle2 = 0.5 * np.sin(np.linspace(0, 12 * np.pi, days))  # Six complete cycles

    for i in range(1, days):
        # Random component
        random_return = np.random.normal(trend, volatility)

        # Cyclical component
        cycle_component = 0.002 * cycle1[i] + 0.001 * cycle2[i]

        # Calculate new price
        new_price = close_prices[-1] * (1 + random_return + cycle_component)
        close_prices.append(new_price)

    # Generate OHLCV data
    data = pd.DataFrame(
        {
            "open": [p * (1 + np.random.normal(0, 0.001)) for p in close_prices],
            "high": [p * (1 + abs(np.random.normal(0, 0.003))) for p in close_prices],
            "low": [p * (1 - abs(np.random.normal(0, 0.003))) for p in close_prices],
            "close": close_prices,
            "volume": [np.random.randint(1000, 10000) for _ in range(days)],
        },
        index=dates,
    )

    # Ensure high/low are actually high/low
    for i in range(len(data)):
        row = data.iloc[i]
        high = max(row["open"], row["close"], row["high"])
        low = min(row["open"], row["close"], row["low"])
        data.loc[data.index[i], "high"] = high
        data.loc[data.index[i], "low"] = low

    # Add symbol and timeframe information
    data.symbol = "EURUSD"
    data.timeframe = "D1"

    return data


def run_multi_timeframe_analysis(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Run multi-timeframe analysis on the sample data.

    Args:
        data: OHLCV DataFrame for the smallest timeframe

    Returns:
        Dictionary with multi-timeframe analysis results
    """
    print("\n=== Multi-Timeframe Analysis ===")

    # Define timeframes for analysis
    timeframes = ["H4", "D1", "W1"]
    print(f"Analyzing timeframes: {', '.join(timeframes)}")

    # Run analysis
    mtf_results = run_mtf_analysis(data, timeframes, show_details=True)

    # Extract and print consensus summary
    consensus = mtf_results["consensus"]
    print(f"\nTrading Bias: {consensus['trading_bias']}")
    print(
        f"Overall Trend: {'Bullish' if consensus['overall_trend'] > 0 else 'Bearish' if consensus['overall_trend'] < 0 else 'Neutral'}"
    )
    print(f"Trend Alignment: {consensus['trend_alignment']:.2f}")

    if consensus["near_support"]:
        print(f"Price is near support (distance: {consensus['support_distance']:.2%})")
    if consensus["near_resistance"]:
        print(
            f"Price is near resistance (distance: {consensus['resistance_distance']:.2%})"
        )

    return mtf_results


def run_pattern_detection(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Run advanced pattern detection on the sample data.

    Args:
        data: OHLCV DataFrame

    Returns:
        Dictionary with pattern detection results
    """
    print("\n=== Advanced Pattern Detection ===")

    # Detect harmonic patterns
    harmonic_patterns = detect_harmonic_patterns(data, swing_strength=3)

    # Detect Elliott Wave patterns
    elliott_patterns = detect_elliott_waves(data, swing_strength=3)

    # Print summary
    harmonic_count = len(harmonic_patterns)
    elliott_count = len(elliott_patterns)

    print(f"Detected {harmonic_count} harmonic patterns")
    print(f"Detected {elliott_count} Elliott Wave patterns")

    # Print details of the most recent patterns
    if harmonic_patterns:
        recent_harmonic = harmonic_patterns[-1]
        print(f"\nMost recent harmonic pattern: {recent_harmonic.pattern_type.name}")
        print(f"Direction: {recent_harmonic.direction.name}")
        print(f"Completion: {recent_harmonic.completion:.2%}")
        print(f"Confidence: {recent_harmonic.confidence:.2f}")

    if elliott_patterns:
        recent_elliott = elliott_patterns[-1]
        print(f"\nMost recent Elliott Wave pattern: {recent_elliott.wave_type.name}")
        print(f"Direction: {recent_elliott.direction.name}")
        print(f"Wave count: {recent_elliott.wave_count}")
        print(f"Confidence: {recent_elliott.confidence:.2f}")

    return {
        "harmonic_patterns": harmonic_patterns,
        "elliott_patterns": elliott_patterns,
    }


def run_backtest(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Run backtest with the sample data.

    Args:
        data: OHLCV DataFrame

    Returns:
        Dictionary with backtest results
    """
    print("\n=== Backtesting with Risk-Adjusted Metrics ===")

    # Create strategy
    strategy = MACrossoverStrategy(
        fast_period=20,
        slow_period=50,
        rsi_period=14,
        rsi_overbought=70,
        rsi_oversold=30,
    )

    # Create backtest
    backtest = Backtest(
        data=data,
        strategy=strategy,
        cash=10000.0,
        commission=0.0001,  # 0.01%
        slippage=0.0001,  # 0.01%
        symbol=data.symbol,
        timeframe=data.timeframe,
    )

    # Run backtest
    result = backtest.run()

    # Print summary
    print(result.get_summary())

    # Run optimization
    print("\nOptimizing strategy parameters...")
    parameter_ranges = {
        "fast_period": [10, 20, 30],
        "slow_period": [40, 50, 60],
        "rsi_period": [7, 14, 21],
        "rsi_overbought": [70, 75, 80],
        "rsi_oversold": [20, 25, 30],
    }

    opt_result = backtest.optimize(
        parameter_ranges=parameter_ranges, metric="sharpe_ratio", maximize=True
    )

    # Print optimization results
    print(f"\nBest parameters: {opt_result['best_params']}")
    best_value = opt_result["best_value"]
    print(f"Best Sharpe ratio: {best_value:.2f}")

    return {"initial_result": result, "optimization_result": opt_result}


def plot_results(
    data: pd.DataFrame, mtf_results: Dict[str, Any], backtest_results: Dict[str, Any]
) -> None:
    """
    Plot analysis results.

    Args:
        data: OHLCV DataFrame
        mtf_results: Multi-timeframe analysis results
        backtest_results: Backtest results
    """
    print("\n=== Generating Plots ===")

    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))

    # Plot 1: Price chart with support/resistance levels
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(data.index[-100:], data["close"][-100:], label="Close Price")

    if "support_resistance" in mtf_results:
        sr_results = mtf_results["support_resistance"]
        current_price = sr_results["current_price"]
        nearest_support = sr_results.get("nearest_support")
        nearest_resistance = sr_results.get("nearest_resistance")

        if nearest_support is not None:
            ax1.axhline(
                y=nearest_support, color="g", linestyle="--", alpha=0.5, label="Support"
            )

        if nearest_resistance is not None:
            ax1.axhline(
                y=nearest_resistance,
                color="r",
                linestyle="--",
                alpha=0.5,
                label="Resistance",
            )

    ax1.set_title("Price with Support/Resistance")
    ax1.set_ylabel("Price")
    ax1.legend()

    # Plot 2: Equity curve from backtest
    ax2 = fig.add_subplot(2, 2, 2)
    result = backtest_results["initial_result"]
    ax2.plot(result.equity_curve.index, result.equity_curve, label="Equity Curve")
    ax2.set_title("Backtest Equity Curve")
    ax2.set_ylabel("Equity")

    # Plot 3: Drawdowns
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.fill_between(result.drawdowns.index, 0, result.drawdowns.values * 100)
    ax3.set_title("Drawdowns")
    ax3.set_ylabel("Drawdown %")
    ax3.set_ylim(result.drawdowns.min() * 100 * 1.1, 5)

    # Plot 4: Monthly returns heatmap
    ax4 = fig.add_subplot(2, 2, 4)

    # Calculate monthly returns
    monthly_returns = result.returns.resample("M").apply(lambda x: (1 + x).prod() - 1)

    # Convert to DataFrame with year and month
    monthly_returns_df = pd.DataFrame(
        {
            "return": monthly_returns,
            "year": monthly_returns.index.year,
            "month": monthly_returns.index.month,
        }
    )

    # Pivot for heatmap
    pivot = monthly_returns_df.pivot("year", "month", "return")

    # Create heatmap directly with pyplot
    im = ax4.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=-0.1, vmax=0.1)

    # Set labels
    month_labels = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    ax4.set_xticks(np.arange(len(month_labels)))
    ax4.set_xticklabels(month_labels)

    years = pivot.index.astype(str).tolist()
    ax4.set_yticks(np.arange(len(years)))
    ax4.set_yticklabels(years)

    plt.colorbar(im, ax=ax4, label="Monthly Return")
    ax4.set_title("Monthly Returns Heatmap")

    plt.tight_layout()
    plt.savefig("technical_analysis_results.png")
    print("Plots saved to 'technical_analysis_results.png'")
    plt.close()


def main():
    """Main function to run the demo."""
    print("=== Technical Analysis Suite Demo ===")

    # Generate sample data
    print("Generating sample data...")
    data = generate_sample_data(days=500)
    print(f"Generated {len(data)} days of sample data for {data.symbol}")

    # Run multi-timeframe analysis
    mtf_results = run_multi_timeframe_analysis(data)

    # Run pattern detection
    pattern_results = run_pattern_detection(data)

    # Run backtest
    backtest_results = run_backtest(data)

    # Plot results
    plot_results(data, mtf_results, backtest_results)

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()

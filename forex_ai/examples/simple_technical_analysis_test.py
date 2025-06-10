"""
Simple technical analysis test using historical data.

This script tests the technical analysis capabilities without requiring
a live connection to OANDA's WebSocket API.
"""

import asyncio
import logging
import os
import json
from datetime import datetime, timedelta

from forex_ai.utils.logging import get_logger, setup_file_logging
from forex_ai.agents.technical import TechnicalAnalysisAgent
from forex_ai.integration.patterns.enhanced_pattern_recognition import (
    EnhancedPatternRecognition,
)
from forex_ai.analysis.technical.indicators import IndicatorCalculator

# Set up logging
setup_file_logging()
logger = get_logger(__name__)

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


async def run_technical_analysis():
    """Run technical analysis on sample data."""
    print("Starting technical analysis test...")

    # Initialize components
    indicator_calculator = IndicatorCalculator()
    pattern_recognizer = EnhancedPatternRecognition()
    tech_agent = TechnicalAnalysisAgent()

    # Extract arrays for analysis
    dates = [item["time"] for item in SAMPLE_DATA]
    opens = [item["open"] for item in SAMPLE_DATA]
    highs = [item["high"] for item in SAMPLE_DATA]
    lows = [item["low"] for item in SAMPLE_DATA]
    closes = [item["close"] for item in SAMPLE_DATA]
    volumes = [item["volume"] for item in SAMPLE_DATA]

    # Calculate some indicators
    print("\nCalculating indicators...")
    rsi = indicator_calculator.calculate_rsi(closes, period=14)
    macd, macd_signal, macd_hist = indicator_calculator.calculate_macd(closes)
    ma_50 = indicator_calculator.calculate_moving_average(closes, period=5)
    ma_200 = indicator_calculator.calculate_moving_average(closes, period=10)

    print(f"RSI (latest): {rsi[-1]:.2f}")
    print(f"MACD (latest): {macd[-1]:.6f}")
    print(f"MACD Signal (latest): {macd_signal[-1]:.6f}")
    print(f"MACD Histogram (latest): {macd_hist[-1]:.6f}")
    print(f"5-period MA (latest): {ma_50[-1]:.4f}")
    print(f"10-period MA (latest): {ma_200[-1]:.4f}")

    # Detect patterns
    print("\nDetecting patterns...")
    patterns = pattern_recognizer.find_patterns(
        opens=opens, highs=highs, lows=lows, closes=closes, volumes=volumes
    )

    # Print detected patterns
    if patterns:
        print(f"Found {len(patterns)} patterns:")
        for pattern in patterns:
            print(f"  - {pattern['name']} (confidence: {pattern['confidence']:.2f})")
    else:
        print("No patterns detected in sample data")

    # Get technical analysis summary
    print("\nGenerating analysis summary...")
    analysis = await tech_agent.analyze_data(
        instrument="EUR_USD",
        timeframe="daily",
        ohlc_data={
            "dates": dates,
            "opens": opens,
            "highs": highs,
            "lows": lows,
            "closes": closes,
            "volumes": volumes,
        },
    )

    # Print analysis results
    print("\nTechnical Analysis Results:")
    print(f"Direction: {analysis.get('direction', 'unknown')}")
    print(f"Confidence: {analysis.get('confidence', 0):.2f}")

    if "support_levels" in analysis:
        print(
            f"Support levels: {', '.join([str(level) for level in analysis.get('support_levels', [])])}"
        )

    if "resistance_levels" in analysis:
        print(
            f"Resistance levels: {', '.join([str(level) for level in analysis.get('resistance_levels', [])])}"
        )

    if "key_insights" in analysis:
        print("\nKey Insights:")
        for insight in analysis.get("key_insights", []):
            print(f"  - {insight}")

    print("\nTechnical analysis test completed")


if __name__ == "__main__":
    asyncio.run(run_technical_analysis())

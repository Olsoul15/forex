# Features Module

This module provides feature extraction and analysis functionality for the Forex AI Trading System, enabling the identification of tradable patterns and market conditions.

## Overview

The features module extracts meaningful patterns, characteristics, and states from raw market data. These extracted features provide a higher-level representation of market conditions that can be used by trading strategies and analysis modules.

## Key Components

### Market State Analysis

- **market_states.py**: Provides advanced market state detection and analysis
  - Identifies market regimes (trending, ranging, volatile, etc.)
  - Detects transitions between market states
  - Quantifies state characteristics like strength and confidence
  - Offers volatility regime classification

## Market State Analysis

The `market_states.py` file provides a robust framework for identifying and analyzing the current state of a market. It includes:

### Enums and Types

- **MarketState**: Enumeration of possible market states:
  - STRONG_UPTREND, UPTREND, WEAK_UPTREND
  - STRONG_DOWNTREND, DOWNTREND, WEAK_DOWNTREND
  - CONSOLIDATION, RANGE_BOUND
  - BREAKOUT, REVERSAL
  - HIGH_VOLATILITY, LOW_VOLATILITY

- **VolatilityRegime**: Enumeration of volatility regimes:
  - VERY_LOW, LOW, MEDIUM, HIGH, VERY_HIGH
  - INCREASING, DECREASING

### Core Components

- **MarketStateAnalyzer**: Main class for analyzing market states
  - Calculates technical indicators for state detection
  - Identifies the current market state based on indicator patterns
  - Determines the current volatility regime
  - Provides confidence scores for state classification

- **get_market_state()**: Convenience function to quickly get the current market state for a symbol and timeframe

## Usage Examples

### Analyzing Market State

```python
from forex_ai.features.market_states import MarketStateAnalyzer
import pandas as pd

# Create a market state analyzer
analyzer = MarketStateAnalyzer()

# Analyze market state from price data
market_state = analyzer.analyze_market_state(
    data=price_data_dataframe,
    symbol="EUR/USD",
    timeframe="1h"
)

# Check the detected state
print(f"Current market state: {market_state['state']}")
print(f"Confidence: {market_state['confidence']}")
print(f"Volatility regime: {market_state['volatility_regime']}")
```

### Using the Convenience Function

```python
from forex_ai.features.market_states import get_market_state

# Get the current market state for a currency pair
state_info = get_market_state(
    symbol="EUR/USD",
    timeframe="1h",
    lookback_bars=100
)

# Use the state information
if state_info["state"] == "STRONG_UPTREND" and state_info["confidence"] > 0.7:
    print("Strong uptrend detected with high confidence")
```

## Use Cases

The market state analysis provided by this module can be used for:

- **Adaptive Trading Strategies**: Adjust strategy parameters based on the current market state
- **Strategy Selection**: Choose the most appropriate strategy for the current market conditions
- **Risk Management**: Adjust position sizing and risk parameters based on volatility regime
- **Analysis Filtering**: Filter trading signals based on market conditions
- **Execution Timing**: Determine the optimal time to enter or exit positions

## Dependencies

- **Data Module**: For retrieving market data
- **NumPy and Pandas**: For data manipulation and calculations 
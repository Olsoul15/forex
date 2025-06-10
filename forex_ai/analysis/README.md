# Analysis Module

This module provides market analysis capabilities for the Forex AI Trading System, including technical, fundamental, and sentiment analysis.

## Overview

The analysis module implements various techniques for analyzing forex markets, from traditional technical indicators and chart patterns to advanced sentiment analysis and correlation studies. These analysis techniques provide the foundation for trading strategies and decision-making in the system.

## Key Components

### Technical Analysis

The `technical/` directory contains components for technical analysis:

- **Indicators**: Implementation of technical indicators (RSI, MACD, Bollinger Bands, etc.)
- **Patterns**: Chart pattern recognition (head and shoulders, double tops, etc.)
- **Multi-timeframe Analysis**: Analysis across different timeframes
- **Backtesting**: Integration with the backtesting framework

### Sentiment Analysis

- **sentiment_analysis.py**: Analysis of market sentiment from various sources
- **social_media_sentiment.py**: Extraction and analysis of sentiment from social media
- **entity_extraction.py**: Named entity recognition for financial texts

### Strategy Optimization

- **strategy_optimization.py**: Tools for optimizing trading strategy parameters
- **impact_prediction.py**: Prediction of market impact from various factors

### Pine Script Integration

- **pine_script.py**: Parser and interpreter for Pine Script (TradingView's scripting language)
- Allows integration with existing TradingView strategies

## Usage Examples

### Technical Analysis

```python
from forex_ai.analysis.technical.indicators import calculate_rsi, calculate_macd
from forex_ai.analysis.technical.patterns import detect_patterns
import pandas as pd

# Load data
data = pd.DataFrame(...)  # OHLCV data

# Calculate indicators
rsi = calculate_rsi(data, period=14)
macd, signal, hist = calculate_macd(data, fast=12, slow=26, signal=9)

# Detect patterns
patterns = detect_patterns(data)
print(f"Detected patterns: {patterns}")
```

### Sentiment Analysis

```python
from forex_ai.analysis.sentiment_analysis import analyze_sentiment
from forex_ai.analysis.social_media_sentiment import analyze_social_media

# Analyze news sentiment
news_sentiment = analyze_sentiment("EUR/USD", source="news")
print(f"News sentiment score: {news_sentiment['score']}")

# Analyze social media sentiment
social_sentiment = analyze_social_media("EUR/USD", platforms=["twitter", "reddit"])
print(f"Social media sentiment: {social_sentiment['score']}")
```

## Dependencies

- **Data Module**: For retrieving market data
- **NumPy and Pandas**: For data manipulation and calculations
- **TA-Lib** (optional): For technical indicators
- **scikit-learn**: For machine learning components
- **NLTK and spaCy**: For natural language processing
- **TensorFlow/PyTorch**: For deep learning models (where applicable) 
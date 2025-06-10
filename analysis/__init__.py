"""
Analysis components for the Forex AI Trading System.

This package provides analytical components for forex market analysis, including:
- Technical analysis (indicators, patterns)
- Pine Script integration (strategy management, optimization)
- Market condition analysis
- Sentiment analysis and news impact prediction
- Social media sentiment analysis
"""

# Import public interfaces from technical submodule
from forex_ai.analysis.technical import (
    # Indicators
    simple_moving_average,
    exponential_moving_average,
    bollinger_bands,
    relative_strength_index,
    moving_average_convergence_divergence,
    average_true_range,
    stochastic_oscillator,
    ichimoku_cloud,
    fibonacci_retracement,
    pivot_points,
    apply_indicators,
    # Patterns
    detect_doji,
    detect_hammer_shooting_star,
    detect_engulfing,
    detect_chart_patterns,
    analyze_multi_timeframe,
    score_pattern_confluence,
    detect_candlestick_patterns,
    PatternType,
    PatternDirection,
    PatternResult,
)

# Import sentiment analysis components
from forex_ai.analysis.sentiment_analysis import (
    SentimentAnalyzer,
    ForexEntityExtractor,
    NewsImpactPredictor,
    RAGNewsProcessor,
)

# Import social media sentiment analysis components
from forex_ai.analysis.social_media_sentiment import (
    SocialMediaSentimentAnalyzer,
    SocialMediaConnector,
    SocialSentimentAggregator,
)

__all__ = [
    # Indicators
    "simple_moving_average",
    "exponential_moving_average",
    "bollinger_bands",
    "relative_strength_index",
    "moving_average_convergence_divergence",
    "average_true_range",
    "stochastic_oscillator",
    "ichimoku_cloud",
    "fibonacci_retracement",
    "pivot_points",
    "apply_indicators",
    # Patterns
    "detect_doji",
    "detect_hammer_shooting_star",
    "detect_engulfing",
    "detect_chart_patterns",
    "analyze_multi_timeframe",
    "score_pattern_confluence",
    "detect_candlestick_patterns",
    "PatternType",
    "PatternDirection",
    "PatternResult",
    # Sentiment Analysis
    "SentimentAnalyzer",
    "ForexEntityExtractor",
    "NewsImpactPredictor",
    "RAGNewsProcessor",
    # Social Media Sentiment Analysis
    "SocialMediaSentimentAnalyzer",
    "SocialMediaConnector",
    "SocialSentimentAggregator",
]

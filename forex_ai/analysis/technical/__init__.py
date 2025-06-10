"""
Technical Analysis Package for the AI Forex Trading System.

This package provides a comprehensive suite of technical analysis tools
for forex trading, including:

1. Multi-timeframe analysis
2. Advanced pattern recognition
3. Deep learning predictions
4. Backtesting and risk-adjusted metrics
"""

# Import key components for easier access
# Remove multi_timeframe imports as TA-Lib is not available/needed
# from forex_ai.analysis.technical.multi_timeframe import (
#     MTFAnalysis, Timeframe, run_mtf_analysis
# )

# REMOVED: Imports from old backtesting engine and strategy
# from forex_ai.analysis.technical.backtesting.engine import (
#     Backtest, BacktestResult
# )
# from forex_ai.analysis.technical.backtesting.strategy import (
#     Strategy, SignalType
# )

# Import indicators
from forex_ai.analysis.technical.indicators import (
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
)

# Import patterns
from forex_ai.analysis.technical.patterns import (
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

# Version
__version__ = "0.1.0"

# Import public interfaces from submodules
from forex_ai.analysis.technical.pine_script import (
    PineScriptStrategy,
    PineScriptStrategyManager,
    PineScriptOptimizer,
    HammersAndStarsStrategy,
)

__all__ = [
    # Multi-timeframe analysis - Removed
    # 'MTFAnalysis', 'Timeframe', 'run_mtf_analysis',
    # REMOVED: Old Backtesting exports
    # 'Backtest', 'BacktestResult', 'Strategy', 'SignalType',
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
    # Pine Script Integration
    "PineScriptStrategy",
    "PineScriptStrategyManager",
    "PineScriptOptimizer",
    "HammersAndStarsStrategy",
]

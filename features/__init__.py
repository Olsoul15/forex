"""
Features package for the Forex AI Trading System.

This package provides feature extraction and analysis functionality
for forex market data, enabling the system to identify market states,
patterns, and other tradable features.
"""

from forex_ai.features.market_states import (
    MarketState,
    VolatilityRegime,
    MarketStateAnalyzer,
    get_market_state,
)

__all__ = [
    # Market state analysis
    "MarketState",
    "VolatilityRegime",
    "MarketStateAnalyzer",
    "get_market_state",
]

"""
Pine Script integration for the Forex AI Trading System.

This package provides Pine Script integration, including:
- Strategy management (loading, parsing, execution)
- Parameter extraction and optimization
- Strategy deployment
"""

# Import public interfaces from submodules
from forex_ai.analysis.technical.pine_script.manager import (
    PineScriptStrategy,
    PineScriptStrategyManager,
)

from forex_ai.analysis.technical.pine_script.optimizer import PineScriptOptimizer

# Import strategies
from forex_ai.analysis.technical.strategies.hammers_and_stars import (
    HammersAndStarsStrategy,
)

__all__ = [
    # Manager
    "PineScriptStrategy",
    "PineScriptStrategyManager",
    # Optimizer
    "PineScriptOptimizer",
    # Strategies
    "HammersAndStarsStrategy",
]

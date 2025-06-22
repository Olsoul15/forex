"""
Strategy components for the Forex AI Trading System dashboard.

This module provides strategy components for the dashboard.
"""

from typing import Dict, Any, List
from datetime import datetime


class StrategyComponent:
    """Strategy component for the dashboard."""

    def __init__(self):
        """Initialize the strategy component."""
        self.config = {}

    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the strategy component with configuration.

        Args:
            config: Configuration for the strategy component.
        """
        self.config = config

    def get_strategies(self) -> List[Dict[str, Any]]:
        """
        Get strategies.

        Returns:
            List[Dict[str, Any]]: List of strategies.
        """
        # Mock data for now
        return [
            {
                "id": "strategy-1",
                "name": "Moving Average Crossover",
                "description": "A simple moving average crossover strategy",
                "status": "active",
                "performance": {
                    "win_rate": 0.65,
                    "profit_factor": 1.8,
                    "sharpe_ratio": 1.2,
                },
                "last_signal": "2023-04-15T10:30:00Z",
            },
            {
                "id": "strategy-2",
                "name": "RSI Divergence",
                "description": "RSI divergence detection strategy",
                "status": "inactive",
                "performance": {
                    "win_rate": 0.58,
                    "profit_factor": 1.5,
                    "sharpe_ratio": 0.9,
                },
                "last_signal": "2023-04-10T14:45:00Z",
            },
        ] 
"""
Performance components for the Forex AI Trading System dashboard.

This module provides performance components for the dashboard.
"""

from typing import Dict, Any, List
from datetime import datetime


class PerformanceComponent:
    """Performance component for the dashboard."""

    def __init__(self):
        """Initialize the performance component."""
        self.config = {}

    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the performance component with configuration.

        Args:
            config: Configuration for the performance component.
        """
        self.config = config

    def get_summary_metrics(self) -> Dict[str, Any]:
        """
        Get performance summary metrics.

        Returns:
            Dict[str, Any]: Performance summary metrics.
        """
        # Mock data for now
        return {
            "win_rate": 0.62,
            "profit_factor": 1.75,
            "drawdown_max": 5.2,
            "sharpe_ratio": 1.1,
            "total_trades": 145,
            "period": "1m",
            "updated_at": datetime.now().isoformat(),
        } 
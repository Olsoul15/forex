"""
Signal components for the Forex AI Trading System dashboard.

This module provides signal components for the dashboard.
"""

from typing import Dict, Any, List
from datetime import datetime


class SignalComponent:
    """Signal component for the dashboard."""

    def __init__(self):
        """Initialize the signal component."""
        self.config = {}

    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the signal component with configuration.

        Args:
            config: Configuration for the signal component.
        """
        self.config = config

    def get_active_signals(self) -> List[Dict[str, Any]]:
        """
        Get active signals.

        Returns:
            List[Dict[str, Any]]: List of active signals.
        """
        # Mock data for now
        return [
            {
                "id": "signal-1",
                "instrument": "EUR/USD",
                "direction": "buy",
                "entry_price": 1.0950,
                "stop_loss": 1.0920,
                "take_profit": 1.1000,
                "strategy": "Moving Average Crossover",
                "timeframe": "H1",
                "signal_time": datetime.now().isoformat(),
                "expiry_time": (datetime.now().replace(hour=datetime.now().hour + 8)).isoformat(),
                "status": "active",
            }
        ]

    def get_recent_signals(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent signals.

        Args:
            limit: Maximum number of signals to return.

        Returns:
            List[Dict[str, Any]]: List of recent signals.
        """
        # Mock data for now
        signals = [
            {
                "id": f"signal-{i}",
                "instrument": "EUR/USD" if i % 2 == 0 else "GBP/USD",
                "direction": "buy" if i % 2 == 0 else "sell",
                "entry_price": 1.0950 + (i * 0.001),
                "stop_loss": 1.0920 + (i * 0.001),
                "take_profit": 1.1000 + (i * 0.001),
                "strategy": "Moving Average Crossover" if i % 2 == 0 else "RSI Divergence",
                "timeframe": "H1" if i % 3 == 0 else "H4",
                "signal_time": datetime.now().isoformat(),
                "status": "closed",
                "result": "win" if i % 3 == 0 else "loss",
                "profit_pips": 50 if i % 3 == 0 else -30,
            }
            for i in range(1, limit + 1)
        ]
        return signals 
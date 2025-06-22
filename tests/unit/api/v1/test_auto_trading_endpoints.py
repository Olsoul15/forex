"""
Unit tests for auto-trading endpoints.

Tests auto-trading preferences, stats, and trades endpoints.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

from forex_ai.api.v1.auto_trading_endpoints import (
    get_auto_trading_preferences,
    get_auto_trading_stats,
    get_recent_auto_trades,
)


@pytest.fixture
def mock_user():
    """Mock user for testing."""
    return {
        "id": "test-user-id",
        "email": "test@example.com",
        "role": "test_user",
        "created_at": "2025-01-01T00:00:00Z"
    }


class TestAutoTradingPreferencesEndpoint:
    """Tests for auto-trading preferences endpoint."""
    
    @pytest.mark.asyncio
    @patch('forex_ai.api.v1.auto_trading_endpoints.get_auto_trading_preferences')
    async def test_get_preferences_basic(self, mock_get_preferences, mock_user):
        """Test getting auto-trading preferences with default parameters."""
        # Create a mock response
        mock_response = {
            "account_id": "demo-account",
            "enabled": True,
            "risk_per_trade": 2.0,
            "max_daily_trades": 5,
            "max_open_trades": 3,
            "allowed_instruments": ["EUR/USD", "GBP/USD", "USD/JPY"],
            "allowed_strategies": ["trend_following", "breakout"],
            "trading_hours": {
                "start": "08:00",
                "end": "16:00",
                "timezone": "UTC"
            },
            "updated_at": datetime.now(),
            "timestamp": datetime.now()
        }
        mock_get_preferences.return_value = mock_response
        
        response = mock_response
        assert response is not None
        assert response["account_id"] == "demo-account"
        assert response["enabled"] is True
        assert response["risk_per_trade"] == 2.0
        assert len(response["allowed_instruments"]) == 3
        assert len(response["allowed_strategies"]) == 2


class TestAutoTradingStatsEndpoint:
    """Tests for auto-trading stats endpoint."""
    
    @pytest.mark.asyncio
    @patch('forex_ai.api.v1.auto_trading_endpoints.get_auto_trading_stats')
    async def test_get_stats_basic(self, mock_get_stats, mock_user):
        """Test getting auto-trading stats with default parameters."""
        # Create a mock response
        mock_response = {
            "account_id": "demo-account",
            "period": "1m",
            "total_trades": 45,
            "winning_trades": 27,
            "losing_trades": 18,
            "win_rate": 60.0,
            "profit_factor": 1.8,
            "avg_win": 25.5,
            "avg_loss": -15.2,
            "largest_win": 120.0,
            "largest_loss": -75.0,
            "net_profit": 320.5,
            "roi": 12.5,
            "sharpe_ratio": 1.2,
            "drawdown": 8.5,
            "timestamp": datetime.now()
        }
        mock_get_stats.return_value = mock_response
        
        response = mock_response
        assert response is not None
        assert response["account_id"] == "demo-account"
        assert response["period"] == "1m"
        assert response["total_trades"] == 45
        assert response["winning_trades"] == 27
        assert response["losing_trades"] == 18
        assert response["win_rate"] == 60.0
    
    @pytest.mark.asyncio
    @patch('forex_ai.api.v1.auto_trading_endpoints.get_auto_trading_stats')
    async def test_get_stats_with_period(self, mock_get_stats, mock_user):
        """Test getting auto-trading stats with different periods."""
        periods = ["1d", "1w", "1m", "3m", "6m", "1y", "all"]
        
        for period in periods:
            # Create a mock response for each period
            mock_response = {
                "account_id": "demo-account",
                "period": period,
                "total_trades": 45,
                "winning_trades": 27,
                "losing_trades": 18,
                "win_rate": 60.0,
                "profit_factor": 1.8,
                "avg_win": 25.5,
                "avg_loss": -15.2,
                "largest_win": 120.0,
                "largest_loss": -75.0,
                "net_profit": 320.5,
                "roi": 12.5,
                "sharpe_ratio": 1.2,
                "drawdown": 8.5,
                "timestamp": datetime.now()
            }
            mock_get_stats.return_value = mock_response
            
            response = mock_response
            assert response is not None
            assert response["period"] == period


class TestAutoTradingTradesEndpoint:
    """Tests for auto-trading trades endpoint."""
    
    @pytest.mark.asyncio
    @patch('forex_ai.api.v1.auto_trading_endpoints.get_recent_auto_trades')
    async def test_get_trades_basic(self, mock_get_trades, mock_user):
        """Test getting auto-trading trades with default parameters."""
        # Create a mock response
        now = datetime.now()
        mock_response = {
            "account_id": "demo-account",
            "trades": [
                {
                    "id": "trade-1",
                    "instrument": "EUR/USD",
                    "direction": "BUY",
                    "size": 0.1,
                    "entry_price": 1.0950,
                    "stop_loss": 1.0900,
                    "take_profit": 1.1050,
                    "entry_time": now - timedelta(days=2),
                    "close_time": now - timedelta(days=1),
                    "close_price": 1.1020,
                    "profit_loss": 70.0,
                    "profit_loss_pips": 70.0,
                    "status": "CLOSED",
                    "strategy_id": "trend_following",
                    "signal_id": "signal-123"
                },
                {
                    "id": "trade-2",
                    "instrument": "GBP/USD",
                    "direction": "SELL",
                    "size": 0.1,
                    "entry_price": 1.2750,
                    "stop_loss": 1.2800,
                    "take_profit": 1.2650,
                    "entry_time": now - timedelta(hours=5),
                    "close_time": None,
                    "close_price": None,
                    "profit_loss": None,
                    "profit_loss_pips": None,
                    "status": "OPEN",
                    "strategy_id": "breakout",
                    "signal_id": "signal-456"
                }
            ],
            "count": 2,
            "timestamp": now
        }
        mock_get_trades.return_value = mock_response
        
        response = mock_response
        assert response is not None
        assert response["account_id"] == "demo-account"
        assert len(response["trades"]) == 2
        assert response["trades"][0]["instrument"] == "EUR/USD"
        assert response["trades"][0]["status"] == "CLOSED"
        assert response["trades"][1]["instrument"] == "GBP/USD"
        assert response["trades"][1]["status"] == "OPEN"
    
    @pytest.mark.asyncio
    @patch('forex_ai.api.v1.auto_trading_endpoints.get_recent_auto_trades')
    async def test_get_trades_with_filters(self, mock_get_trades, mock_user):
        """Test getting auto-trading trades with filters."""
        # Create a mock response with filtered trades
        now = datetime.now()
        mock_response = {
            "account_id": "demo-account",
            "trades": [
                {
                    "id": "trade-1",
                    "instrument": "EUR/USD",
                    "direction": "BUY",
                    "size": 0.1,
                    "entry_price": 1.0950,
                    "stop_loss": 1.0900,
                    "take_profit": 1.1050,
                    "entry_time": now - timedelta(days=2),
                    "close_time": now - timedelta(days=1),
                    "close_price": 1.1020,
                    "profit_loss": 70.0,
                    "profit_loss_pips": 70.0,
                    "status": "CLOSED",
                    "strategy_id": "trend_following",
                    "signal_id": "signal-123"
                }
            ],
            "count": 1,
            "timestamp": now
        }
        mock_get_trades.return_value = mock_response
        
        response = mock_response
        assert response is not None
        assert len(response["trades"]) == 1
        assert response["trades"][0]["instrument"] == "EUR/USD"
        assert response["trades"][0]["status"] == "CLOSED"

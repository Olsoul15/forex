"""
Unit tests for account endpoints.

Tests account metrics, performance, balance, and equity endpoints.
"""

import pytest
from datetime import datetime, timedelta
from fastapi import HTTPException, status

from forex_ai.api.v1.account_endpoints import (
    get_account_metrics,
    get_account_performance,
    get_account_balance,
    get_account_equity,
)
from forex_ai.backend_api.db import account_db


@pytest.fixture
def mock_user():
    """Mock user for testing."""
    return {
        "id": "test-user-id",
        "email": "test@example.com",
        "role": "test_user",
        "created_at": "2025-01-01T00:00:00Z"
    }


@pytest.fixture
def mock_account_db(monkeypatch):
    """Mock account database functions."""
    
    def mock_user_has_account_access(user_id, account_id):
        """Mock user_has_account_access function."""
        return user_id == "test-user-id" and account_id in ["demo-account", "test-account"]
    
    def mock_get_account_by_id(account_id):
        """Mock get_account_by_id function."""
        if account_id == "demo-account":
            return {
                "account_id": "demo-account",
                "name": "Demo Account",
                "currency": "USD",
                "balance": 10000.0,
                "equity": 10500.0,
                "margin_used": 500.0,
                "margin_available": 9500.0,
                "unrealized_pl": 500.0,
                "realized_pl": 0.0,
                "open_position_count": 2,
                "pending_order_count": 1,
                "account_type": "DEMO",
                "leverage": 100.0,
                "margin_rate": 0.01,
                "created_at": datetime.now() - timedelta(days=30),
                "provider": "OANDA",
            }
        elif account_id == "test-account":
            return {
                "account_id": "test-account",
                "name": "Test Account",
                "currency": "EUR",
                "balance": 5000.0,
                "equity": 4800.0,
                "margin_used": 300.0,
                "margin_available": 4500.0,
                "unrealized_pl": -200.0,
                "realized_pl": 100.0,
                "open_position_count": 1,
                "pending_order_count": 0,
                "account_type": "DEMO",
                "leverage": 50.0,
                "margin_rate": 0.02,
                "created_at": datetime.now() - timedelta(days=15),
                "provider": "OANDA",
            }
        return None
    
    def mock_get_account_metrics(account_id, period="1m"):
        """Mock get_account_metrics function."""
        if account_id in ["demo-account", "test-account"]:
            return {
                "win_rate": 65.0,
                "profit_factor": 2.0,
                "expectancy": 0.5,
                "avg_win": 100.0,
                "avg_loss": -50.0,
                "max_drawdown": 10.0,
                "sharpe_ratio": 1.5,
                "sortino_ratio": 2.0,
                "total_trades": 20,
                "profitable_trades": 13,
                "losing_trades": 7,
                "average_win": 100.0,
                "average_loss": -50.0,
                "largest_win": 200.0,
                "largest_loss": -100.0,
                "drawdown_max": 10.0,
                "drawdown_current": 5.0,
                "time_period": period,
                "timestamp": datetime.now(),
            }
        return None
    
    def mock_get_account_performance(account_id, period="1m"):
        """Mock get_account_performance function."""
        if account_id in ["demo-account", "test-account"]:
            now = datetime.now()
            daily_returns = {}
            cumulative_returns = {}
            equity_curve = {}
            drawdown_curve = {}
            
            # Generate mock data for the past 30 days
            for i in range(30):
                date = (now - timedelta(days=i)).strftime("%Y-%m-%d")
                daily_returns[date] = (i % 5 - 2) / 100  # Values between -2% and +2%
                cumulative_returns[date] = (1 + (i % 10) / 100)  # Values between 1% and 10%
                equity_curve[date] = 10000 * (1 + (i % 15) / 100)  # Starting from 10000
                drawdown_curve[date] = (i % 5) / 100  # Values between 0% and 5%
            
            return {
                "daily_returns": daily_returns,
                "cumulative_returns": cumulative_returns,
                "monthly_returns": {"2025-06": 0.05, "2025-05": -0.02},
                "equity_curve": equity_curve,
                "drawdown_curve": drawdown_curve,
                "start_date": now - timedelta(days=30),
            }
        return None
    
    # Apply monkeypatches
    monkeypatch.setattr(account_db, "user_has_account_access", mock_user_has_account_access)
    monkeypatch.setattr(account_db, "get_account_by_id", mock_get_account_by_id)
    monkeypatch.setattr(account_db, "get_account_metrics", mock_get_account_metrics)
    monkeypatch.setattr(account_db, "get_account_performance", mock_get_account_performance)
    
    return {
        "user_has_account_access": mock_user_has_account_access,
        "get_account_by_id": mock_get_account_by_id,
        "get_account_metrics": mock_get_account_metrics,
        "get_account_performance": mock_get_account_performance,
    }


class TestAccountMetricsEndpoint:
    """Tests for account metrics endpoint."""
    
    @pytest.mark.asyncio
    async def test_get_metrics_valid_account(self, mock_user, mock_account_db):
        """Test getting metrics for a valid account."""
        response = await get_account_metrics(accountId="demo-account", current_user=mock_user)
        assert response is not None
        assert response.win_rate == 65.0
        assert response.profit_factor == 2.0
        assert response.total_trades == 20
        assert response.profitable_trades == 13
        assert response.losing_trades == 7
    
    @pytest.mark.asyncio
    async def test_get_metrics_invalid_account(self, mock_user, mock_account_db):
        """Test getting metrics for an invalid account."""
        with pytest.raises(HTTPException) as excinfo:
            await get_account_metrics(accountId="invalid-account", current_user=mock_user)
        assert excinfo.value.status_code == status.HTTP_403_FORBIDDEN
    
    @pytest.mark.asyncio
    async def test_get_metrics_unauthorized(self, mock_user, mock_account_db, monkeypatch):
        """Test getting metrics for an account the user doesn't have access to."""
        # Override the mock to return False for access check
        monkeypatch.setattr(account_db, "user_has_account_access", lambda user_id, account_id: False)
        
        with pytest.raises(HTTPException) as excinfo:
            await get_account_metrics(accountId="demo-account", current_user=mock_user)
        assert excinfo.value.status_code == status.HTTP_403_FORBIDDEN


class TestAccountPerformanceEndpoint:
    """Tests for account performance endpoint."""
    
    @pytest.mark.asyncio
    async def test_get_performance_valid_account(self, mock_user, mock_account_db):
        """Test getting performance for a valid account."""
        response = await get_account_performance(
            accountId="demo-account", 
            period="1m", 
            current_user=mock_user
        )
        assert response is not None
        response_data = response.model_dump()
        assert "daily_returns" in response_data
        assert "cumulative_returns" in response_data
        assert "monthly_returns" in response_data
        assert "equity_curve" in response_data
        assert "drawdown_curve" in response_data
    
    @pytest.mark.asyncio
    async def test_get_performance_different_periods(self, mock_user, mock_account_db):
        """Test getting performance for different time periods."""
        periods = ["1d", "1w", "1m", "3m", "6m", "1y", "all"]
        
        for period in periods:
            response = await get_account_performance(
                accountId="demo-account", 
                period=period, 
                current_user=mock_user
            )
            assert response is not None
            assert response.period == period


class TestAccountBalanceEndpoint:
    """Tests for account balance endpoint."""
    
    @pytest.mark.asyncio
    async def test_get_balance_valid_account(self, mock_user, mock_account_db):
        """Test getting balance for a valid account."""
        response = await get_account_balance(accountId="demo-account", current_user=mock_user)
        assert response is not None
        assert response.balance == 10000.0
        assert response.currency == "USD"
    
    @pytest.mark.asyncio
    async def test_get_balance_different_account(self, mock_user, mock_account_db):
        """Test getting balance for a different account."""
        response = await get_account_balance(accountId="test-account", current_user=mock_user)
        assert response is not None
        assert response.balance == 5000.0
        assert response.currency == "EUR"


class TestAccountEquityEndpoint:
    """Tests for account equity endpoint."""
    
    @pytest.mark.asyncio
    async def test_get_equity_valid_account(self, mock_user, mock_account_db):
        """Test getting equity for a valid account."""
        response = await get_account_equity(accountId="demo-account", current_user=mock_user)
        assert response is not None
        assert response.equity == 10500.0
        assert response.balance == 10000.0
        assert response.floating_pnl == 500.0
        assert response.currency == "USD"
    
    @pytest.mark.asyncio
    async def test_get_equity_negative_pnl(self, mock_user, mock_account_db):
        """Test getting equity with negative floating P/L."""
        response = await get_account_equity(accountId="test-account", current_user=mock_user)
        assert response is not None
        assert response.equity == 4800.0
        assert response.balance == 5000.0
        assert response.floating_pnl == -200.0
        assert response.currency == "EUR"

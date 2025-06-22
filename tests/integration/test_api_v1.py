"""
Integration tests for the v1 API endpoints.

This module contains tests for the v1 API endpoints to ensure
they are working correctly.
"""

import pytest
from fastapi.testclient import TestClient
import json
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from forex_ai.api.main import app
from forex_ai.data.storage.supabase_client import get_supabase_db_client

# Test constants
TEST_USER_ID = "test-user-id"
TEST_ACCOUNT_ID = "test-account-id"
TEST_STRATEGY_ID = "test-strategy-id"
TEST_SIGNAL_ID = "test-signal-id"
TEST_JOB_ID = "test-job-id"

client = TestClient(app)

# Test endpoints
def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "api_version" in data
    assert "timestamp" in data
    assert "version" in data

class TestAccountEndpoints:
    """Tests for account endpoints."""

    def test_get_account_metrics(self, test_client, mock_db):
        """Test getting account metrics."""
        response = test_client.get(f"/api/v1/account/metrics?accountId={TEST_ACCOUNT_ID}")
        assert response.status_code == 200
        data = response.json()
        assert "win_rate" in data
        assert "profit_factor" in data
        assert "sharpe_ratio" in data

    def test_get_account_performance(self, test_client, mock_db):
        """Test getting account performance."""
        response = test_client.get(f"/api/v1/account/performance?accountId={TEST_ACCOUNT_ID}")
        assert response.status_code == 200
        data = response.json()
        assert "daily_returns" in data
        assert "cumulative_returns" in data
        assert "equity_curve" in data

    def test_get_account_balance(self, test_client, mock_db):
        """Test getting account balance."""
        response = test_client.get(f"/api/v1/account/balance?accountId={TEST_ACCOUNT_ID}")
        assert response.status_code == 200
        data = response.json()
        assert "balance" in data
        assert "currency" in data

    def test_get_account_equity(self, test_client, mock_db):
        """Test getting account equity."""
        response = test_client.get(f"/api/v1/account/equity?accountId={TEST_ACCOUNT_ID}")
        assert response.status_code == 200
        data = response.json()
        assert "equity" in data
        assert "balance" in data
        assert "floating_pnl" in data

class TestAutoTradingEndpoints:
    """Tests for auto-trading endpoints."""

    def test_get_auto_trading_preferences(self, test_client, mock_db):
        """Test getting auto-trading preferences."""
        response = test_client.get(f"/api/v1/auto-trading/preferences?userId={TEST_USER_ID}")
        assert response.status_code == 200
        data = response.json()
        assert "preferences" in data

    def test_update_auto_trading_preferences(self, test_client, mock_db):
        """Test updating auto-trading preferences."""
        request_data = {
            "userId": TEST_USER_ID,
            "preferences": {
                "enabled": True,
                "risk_per_trade": 1.5,
                "max_daily_trades": 10,
                "max_open_trades": 5,
            }
        }
        response = test_client.put("/api/v1/auto-trading/preferences", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["preferences"]["enabled"] == True
        assert data["preferences"]["risk_per_trade"] == 1.5

    def test_enable_auto_trading(self, test_client, mock_db):
        """Test enabling auto-trading."""
        request_data = {
            "userId": TEST_USER_ID,
            "accountId": TEST_ACCOUNT_ID,
            "enabled": True
        }
        response = test_client.post("/api/v1/auto-trading/enable", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["enabled"] == True

    def test_disable_auto_trading(self, test_client, mock_db):
        """Test disabling auto-trading."""
        request_data = {
            "userId": TEST_USER_ID,
            "accountId": TEST_ACCOUNT_ID,
            "enabled": False
        }
        response = test_client.post("/api/v1/auto-trading/enable", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["enabled"] == False

    def test_get_auto_trading_stats(self, test_client, mock_db):
        """Test getting auto-trading stats."""
        response = test_client.get(f"/api/v1/auto-trading/stats?userId={TEST_USER_ID}")
        assert response.status_code == 200
        data = response.json()
        assert "total_trades" in data
        assert "win_rate" in data

    def test_get_auto_trading_trades(self, test_client, mock_db):
        """Test getting auto-trading trades."""
        response = test_client.get(f"/api/v1/auto-trading/trades?userId={TEST_USER_ID}")
        assert response.status_code == 200
        data = response.json()
        assert "trades" in data
        assert isinstance(data["trades"], list)

class TestSignalEndpoints:
    """Tests for signal endpoints."""

    def test_get_signal_history(self, test_client, mock_db):
        """Test getting signal history."""
        response = test_client.get(f"/api/v1/signals/history?userId={TEST_USER_ID}")
        assert response.status_code == 200
        data = response.json()
        assert "signals" in data
        assert "total" in data
        assert isinstance(data["signals"], list)

    def test_get_signal_performance(self, test_client, mock_db):
        """Test getting signal performance."""
        response = test_client.get(f"/api/v1/signals/performance?userId={TEST_USER_ID}")
        assert response.status_code == 200
        data = response.json()
        assert "metrics" in data
        assert "win_rate" in data["metrics"]

    def test_execute_signal(self, test_client, mock_db):
        """Test executing a signal."""
        request_data = {
            "signalId": TEST_SIGNAL_ID,
            "accountId": TEST_ACCOUNT_ID,
            "size": 0.1,
            "stopLoss": 1.0950,
            "takeProfit": 1.1100
        }
        response = test_client.post("/api/v1/signals/execute", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "trade_id" in data
        assert "execution_price" in data

class TestForexOptimizerEndpoints:
    """Tests for forex optimizer endpoints."""

    def test_get_optimization_jobs(self, test_client, mock_db):
        """Test getting optimization jobs."""
        response = test_client.get(f"/api/v1/forex-optimizer/jobs?userId={TEST_USER_ID}")
        assert response.status_code == 200
        data = response.json()
        assert "jobs" in data
        assert isinstance(data["jobs"], list) 
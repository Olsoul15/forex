"""
Integration tests for the v1 API endpoints.

This module contains tests for the v1 API endpoints to ensure
they are working correctly.
"""

import pytest
from fastapi.testclient import TestClient
import json
from datetime import datetime, timedelta

from forex_ai.api.main import app
from tests.integration.conftest import TEST_USER_ID, TEST_ACCOUNT_ID, TEST_STRATEGY_ID, TEST_SIGNAL_ID, TEST_JOB_ID

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
        }
        response = test_client.post("/api/v1/auto-trading/disable", json=request_data)
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
        assert "count" in data

class TestSignalEndpoints:
    """Tests for signal endpoints."""

    def test_get_signal_history(self, test_client, mock_db):
        """Test getting signal history."""
        response = test_client.get("/api/v1/signals/history")
        assert response.status_code == 200
        data = response.json()
        assert "signals" in data
        assert "count" in data

    def test_get_signal_performance(self, test_client, mock_db):
        """Test getting signal performance."""
        response = test_client.get("/api/v1/signals/performance")
        assert response.status_code == 200
        data = response.json()
        assert "metrics" in data

    def test_execute_signal(self, test_client, mock_db):
        """Test executing a signal."""
        request_data = {
            "signalId": TEST_SIGNAL_ID,
            "accountId": TEST_ACCOUNT_ID,
            "size": 0.1,
        }
        response = test_client.post("/api/v1/signals/execute", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "orderId" in data

class TestForexOptimizerEndpoints:
    """Tests for forex optimizer endpoints."""

    def test_start_optimization(self, test_client, mock_db):
        """Test starting an optimization job."""
        request_data = {
            "strategyId": TEST_STRATEGY_ID,
            "instrument": "EUR_USD",
            "timeframe": "H1",
            "start_date": (datetime.now() - timedelta(days=30)).isoformat(),
            "end_date": datetime.now().isoformat(),
            "parameters": [
                {
                    "name": "fast_ma_period",
                    "min_value": 5,
                    "max_value": 20,
                    "step": 1,
                    "type": "int"
                },
                {
                    "name": "slow_ma_period",
                    "min_value": 20,
                    "max_value": 50,
                    "step": 1,
                    "type": "int"
                }
            ],
            "optimization_metric": "profit_factor",
        }
        response = test_client.post("/api/v1/forex-optimizer/optimize", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert "status" in data

    def test_get_optimization_status(self, test_client, mock_db):
        """Test getting optimization job status."""
        response = test_client.get(f"/api/v1/forex-optimizer/optimize/{TEST_JOB_ID}/status")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "progress" in data

    def test_get_optimization_results(self, test_client, mock_db):
        """Test getting optimization job results."""
        response = test_client.get(f"/api/v1/forex-optimizer/optimize/{TEST_JOB_ID}/results")
        assert response.status_code == 200
        data = response.json()
        assert "results" in data

    def test_start_walkforward_test(self, test_client, mock_db):
        """Test starting a walkforward test."""
        request_data = {
            "strategyId": TEST_STRATEGY_ID,
            "instrument": "EUR_USD",
            "timeframe": "H1",
            "start_date": (datetime.now() - timedelta(days=90)).isoformat(),
            "end_date": datetime.now().isoformat(),
            "parameters": {
                "fast_ma_period": 10,
                "slow_ma_period": 30,
            },
        }
        response = test_client.post("/api/v1/forex-optimizer/walkforward", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert "status" in data

    def test_get_walkforward_results(self, test_client, mock_db):
        """Test getting walkforward test results."""
        response = test_client.get(f"/api/v1/forex-optimizer/walkforward/{TEST_JOB_ID}/results")
        assert response.status_code == 200
        data = response.json()
        assert "results" in data

    def test_start_montecarlo_simulation(self, test_client, mock_db):
        """Test starting a Monte Carlo simulation."""
        request_data = {
            "strategyId": TEST_STRATEGY_ID,
            "instrument": "EUR_USD",
            "timeframe": "H1",
            "start_date": (datetime.now() - timedelta(days=30)).isoformat(),
            "end_date": datetime.now().isoformat(),
            "parameters": {
                "fast_ma_period": 10,
                "slow_ma_period": 30,
            },
        }
        response = test_client.post("/api/v1/forex-optimizer/montecarlo", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert "status" in data

    def test_get_montecarlo_results(self, test_client, mock_db):
        """Test getting Monte Carlo simulation results."""
        response = test_client.get(f"/api/v1/forex-optimizer/montecarlo/{TEST_JOB_ID}/results")
        assert response.status_code == 200
        data = response.json()
        assert "results" in data

class TestLegacyEndpoints:
    """Tests for legacy endpoints."""

    def test_legacy_get_broker_credentials(self, test_client):
        """Test legacy broker credentials endpoint."""
        response = test_client.get("/api/brokers/credentials")
        assert response.status_code in [200, 302, 307]

    def test_legacy_get_strategies(self, test_client):
        """Test legacy strategies endpoint."""
        response = test_client.get("/api/strategies")
        assert response.status_code in [200, 302, 307] 
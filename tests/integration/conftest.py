"""
Pytest configuration for integration tests.

This module contains fixtures and configuration for integration tests.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock
import pandas as pd
import asyncio
import os
import json
from datetime import datetime, timedelta
from fastapi.testclient import TestClient

from forex_ai.api.main import app
from forex_ai.auth.supabase import get_current_user
from forex_ai.backend_api.db import account_db, execution_db, strategy_db

# Test data
TEST_USER_ID = "test-user-id"
TEST_ACCOUNT_ID = "test-account-id"
TEST_STRATEGY_ID = "test-strategy-id"
TEST_SIGNAL_ID = "test-signal-id"
TEST_JOB_ID = "test-job-id"

# TODO: Add shared fixtures here, for example:
# - A fixture for a default AnalysisConfig object.
# - A fixture for a mock MarketDataFetcher.
# - A fixture for a sample AutoAgentOrchestrator instance. 

@pytest.fixture
def mock_analysis_config():
    """Provides a mock AnalysisConfig object."""
    config = MagicMock()
    config.timeframes = ["H1", "H4"]
    config.indicators = ["RSI", "MACD"]
    config.lookback_periods = 100
    return config

@pytest.fixture
def mock_data_fetcher():
    """Provides a mock MarketDataFetcher instance."""
    fetcher = MagicMock()
    
    # Create a realistic-looking mock candle
    mock_candle = {
        "time": "2024-01-01T00:00:00Z",
        "volume": 100,
        "mid": {"o": "1.1000", "h": "1.1050", "l": "1.0990", "c": "1.1030"}
    }
    
    # Mock the get_candles method to be async
    async def mock_get_candles(*args, **kwargs):
        return {
            "instrument": kwargs.get("instrument", "EUR_USD"),
            "granularity": kwargs.get("timeframe", "H1"),
            "candles": [mock_candle for _ in range(100)] # Return 100 candles
        }

    fetcher.get_candles = AsyncMock(side_effect=mock_get_candles)
    return fetcher

@pytest.fixture
def mock_orchestrator_config(mock_analysis_config, mock_data_fetcher):
    """Provides a configuration dictionary for the orchestrator."""
    return {
        "config": mock_analysis_config,
        "data_fetcher": mock_data_fetcher
    }

@pytest.fixture(scope='module')
def test_client():
    """Create a Flask test client for the TA-Lib server."""
    app.config['TESTING'] = True

    with app.test_client() as client:
        yield client 

@pytest.fixture
def test_client():
    """
    Create a test client for the FastAPI app.
    
    Returns:
        TestClient: A test client for the FastAPI app.
    """
    # Mock authentication
    def mock_get_current_user():
        """Mock authentication for testing."""
        return {
            "id": TEST_USER_ID,
            "email": "test@example.com",
            "name": "Test User",
        }
    
    # Override authentication dependency
    app.dependency_overrides[get_current_user] = mock_get_current_user
    
    # Create test client
    client = TestClient(app)
    
    return client

@pytest.fixture
def mock_account_data():
    """
    Mock account data for testing.
    
    Returns:
        dict: Mock account data.
    """
    return {
        "id": TEST_ACCOUNT_ID,
        "user_id": TEST_USER_ID,
        "broker": "oanda",
        "account_number": "101-001-12345678-001",
        "name": "Test Account",
        "currency": "USD",
        "balance": 10000.0,
        "unrealized_pl": 150.0,
        "margin_used": 500.0,
        "margin_available": 9500.0,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }

@pytest.fixture
def mock_trades_data():
    """
    Mock trades data for testing.
    
    Returns:
        list: Mock trades data.
    """
    trades = []
    now = datetime.now()
    
    # Generate some mock trades
    for i in range(20):
        trade_time = now - timedelta(days=i)
        profit = 50.0 if i % 2 == 0 else -30.0
        
        trades.append({
            "id": f"trade-{i}",
            "account_id": TEST_ACCOUNT_ID,
            "user_id": TEST_USER_ID,
            "instrument": "EUR_USD",
            "direction": "buy" if i % 2 == 0 else "sell",
            "size": 0.1,
            "entry_price": 1.1000 + (i * 0.0001),
            "exit_price": 1.1050 + (i * 0.0001) if i % 2 == 0 else 1.0970 + (i * 0.0001),
            "profit_loss": profit,
            "profit_loss_pips": profit * 10,
            "open_time": trade_time.isoformat(),
            "close_time": (trade_time + timedelta(hours=4)).isoformat(),
            "status": "closed",
        })
    
    return trades

@pytest.fixture
def mock_signals_data():
    """
    Mock signals data for testing.
    
    Returns:
        list: Mock signals data.
    """
    signals = []
    now = datetime.now()
    
    # Generate some mock signals
    for i in range(10):
        signal_time = now - timedelta(days=i)
        
        signals.append({
            "id": f"signal-{i}" if i != 0 else TEST_SIGNAL_ID,
            "user_id": TEST_USER_ID,
            "strategy_id": TEST_STRATEGY_ID,
            "strategy_name": "Test Strategy",
            "instrument": "EUR_USD",
            "timeframe": "H1",
            "direction": "buy" if i % 2 == 0 else "sell",
            "entry_price": 1.1000 + (i * 0.0001),
            "stop_loss": 1.0950 + (i * 0.0001),
            "take_profit": 1.1100 + (i * 0.0001),
            "risk_reward_ratio": 2.0,
            "confidence": 0.8,
            "signal_time": signal_time.isoformat(),
            "expiration_time": (signal_time + timedelta(hours=8)).isoformat(),
            "status": "active" if i == 0 else "closed",
            "notes": f"Test signal {i}",
            "executed": i != 0,
            "execution_time": (signal_time + timedelta(hours=1)).isoformat() if i != 0 else None,
            "execution_price": 1.1010 + (i * 0.0001) if i != 0 else None,
            "closed": i != 0,
            "close_time": (signal_time + timedelta(hours=5)).isoformat() if i != 0 else None,
            "close_price": 1.1060 + (i * 0.0001) if i != 0 else None,
            "profit_loss": 50.0 if i % 2 == 0 and i != 0 else -30.0 if i != 0 else None,
            "profit_loss_pips": 50.0 if i % 2 == 0 and i != 0 else -30.0 if i != 0 else None,
        })
    
    return signals

@pytest.fixture
def mock_optimization_job():
    """
    Mock optimization job data for testing.
    
    Returns:
        dict: Mock optimization job data.
    """
    now = datetime.now()
    
    return {
        "id": TEST_JOB_ID,
        "user_id": TEST_USER_ID,
        "strategy_id": TEST_STRATEGY_ID,
        "instrument": "EUR_USD",
        "timeframe": "H1",
        "start_date": (now - timedelta(days=30)).isoformat(),
        "end_date": now.isoformat(),
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
        "population_size": 50,
        "generations": 10,
        "parallel_jobs": 4,
        "status": "completed",
        "progress": 100.0,
        "message": "Optimization completed successfully",
        "created_at": (now - timedelta(hours=2)).isoformat(),
        "updated_at": (now - timedelta(minutes=30)).isoformat(),
        "completed_at": (now - timedelta(minutes=30)).isoformat(),
    }

@pytest.fixture
def mock_optimization_results():
    """
    Mock optimization results data for testing.
    
    Returns:
        list: Mock optimization results data.
    """
    results = []
    
    # Generate some mock optimization results
    for i in range(10):
        results.append({
            "job_id": TEST_JOB_ID,
            "parameters": {
                "fast_ma_period": 5 + i,
                "slow_ma_period": 20 + i * 2,
            },
            "profit_factor": 1.5 + (i * 0.1),
            "sharpe_ratio": 1.2 + (i * 0.05),
            "win_rate": 55.0 + (i * 1.0),
            "total_trades": 100 - i,
            "net_profit": 1000.0 + (i * 100.0),
            "max_drawdown": 5.0 - (i * 0.2),
        })
    
    return results

@pytest.fixture
def mock_walkforward_results():
    """
    Mock walkforward test results data for testing.
    
    Returns:
        list: Mock walkforward test results data.
    """
    results = []
    now = datetime.now()
    
    # Generate some mock walkforward test results
    for i in range(3):
        window_start = now - timedelta(days=90 - i * 30)
        window_end = window_start + timedelta(days=30)
        
        results.append({
            "job_id": TEST_JOB_ID,
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "profit_factor": 1.5 + (i * 0.1),
            "sharpe_ratio": 1.2 + (i * 0.05),
            "win_rate": 55.0 + (i * 1.0),
            "total_trades": 30 - i,
            "net_profit": 300.0 + (i * 50.0),
            "max_drawdown": 5.0 - (i * 0.2),
        })
    
    return results

@pytest.fixture
def mock_montecarlo_results():
    """
    Mock Monte Carlo simulation results data for testing.
    
    Returns:
        list: Mock Monte Carlo simulation results data.
    """
    results = []
    
    # Generate some mock Monte Carlo simulation results
    percentiles = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    
    for i, percentile in enumerate(percentiles):
        results.append({
            "job_id": TEST_JOB_ID,
            "percentile": percentile,
            "final_equity": 10000.0 + (i * 1000.0),
            "max_drawdown": 10.0 - (i * 1.0),
            "profit_factor": 1.0 + (i * 0.2),
            "sharpe_ratio": 0.8 + (i * 0.2),
        })
    
    return results

@pytest.fixture
def mock_db(monkeypatch, mock_account_data, mock_trades_data, mock_signals_data,
            mock_optimization_job, mock_optimization_results, mock_walkforward_results,
            mock_montecarlo_results):
    """
    Mock database functions for testing.
    
    Args:
        monkeypatch: Pytest monkeypatch fixture.
        mock_account_data: Mock account data.
        mock_trades_data: Mock trades data.
        mock_signals_data: Mock signals data.
        mock_optimization_job: Mock optimization job data.
        mock_optimization_results: Mock optimization results data.
        mock_walkforward_results: Mock walkforward test results data.
        mock_montecarlo_results: Mock Monte Carlo simulation results data.
    """
    # Mock account_db functions
    monkeypatch.setattr(account_db, "user_has_account_access", lambda user_id, account_id: True)
    monkeypatch.setattr(account_db, "get_account_by_id", lambda account_id: mock_account_data)
    
    def mock_get_account_metrics(account_id, period="1m"):
        return {
            "win_rate": 60.0,
            "profit_factor": 1.8,
            "sharpe_ratio": 1.5,
            "drawdown_max": 5.0,
            "drawdown_current": 2.0,
            "total_trades": 100,
            "profitable_trades": 60,
            "losing_trades": 40,
            "average_win": 50.0,
            "average_loss": -30.0,
            "largest_win": 200.0,
            "largest_loss": -100.0,
        }
    
    def mock_get_account_performance(account_id, period="1m"):
        now = datetime.now()
        dates = [(now - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(30)]
        
        daily_returns = {date: (i % 5) * 10.0 for i, date in enumerate(dates)}
        
        cumulative = 0
        cumulative_returns = {}
        for date in dates:
            cumulative += daily_returns[date]
            cumulative_returns[date] = cumulative
        
        months = list(set([date[:7] for date in dates]))
        monthly_returns = {month: 100.0 + (i * 50.0) for i, month in enumerate(months)}
        
        return {
            "daily_returns": daily_returns,
            "cumulative_returns": cumulative_returns,
            "monthly_returns": monthly_returns,
            "equity_curve": cumulative_returns,
            "drawdown_curve": {date: (i % 5) * 0.5 for i, date in enumerate(dates)},
            "start_date": (now - timedelta(days=30)).isoformat(),
        }
    
    monkeypatch.setattr(account_db, "get_account_metrics", mock_get_account_metrics)
    monkeypatch.setattr(account_db, "get_account_performance", mock_get_account_performance)
    
    # Mock execution_db functions
    def mock_get_auto_trading_preferences(user_id):
        return {
            "enabled": True,
            "risk_per_trade": 1.0,
            "max_daily_trades": 5,
            "max_open_trades": 3,
            "allowed_instruments": ["EUR_USD", "GBP_USD", "USD_JPY"],
            "trading_hours_start": "00:00",
            "trading_hours_end": "23:59",
            "trading_days": [0, 1, 2, 3, 4],
            "min_win_rate": 50.0,
            "min_profit_factor": 1.2,
            "stop_loss_required": True,
            "take_profit_required": True,
        }
    
    def mock_update_auto_trading_preferences(user_id, preferences):
        return preferences
    
    def mock_set_auto_trading_status(user_id, account_id, enabled):
        return True
    
    def mock_get_auto_trading_stats(user_id, account_id=None, period="1m"):
        return {
            "total_trades": 50,
            "successful_trades": 30,
            "failed_trades": 20,
            "win_rate": 60.0,
            "profit_loss": 1500.0,
            "daily_stats": {
                (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d"): {
                    "trades": 5,
                    "profit_loss": 100.0 - (i * 10.0),
                    "successful_trades": 3,
                    "failed_trades": 2,
                }
                for i in range(10)
            },
        }
    
    def mock_get_auto_trading_trades(user_id, account_id=None, limit=10):
        return mock_trades_data[:limit]
    
    def mock_get_signals(user_id, strategy_id=None, instrument=None, timeframe=None,
                        status=None, start_date=None, end_date=None, limit=50, offset=0):
        return mock_signals_data[:limit]
    
    def mock_get_signal_count(user_id, strategy_id=None, instrument=None, timeframe=None,
                             status=None, start_date=None, end_date=None):
        return len(mock_signals_data)
    
    def mock_get_signal_by_id(signal_id):
        for signal in mock_signals_data:
            if signal["id"] == signal_id:
                return signal
        return {}
    
    def mock_get_signal_performance(user_id, strategy_id=None, instrument=None,
                                   timeframe=None, start_date=None, end_date=None):
        return {
            "win_rate": 65.0,
            "profit_factor": 2.0,
            "average_win": 60.0,
            "average_loss": -30.0,
            "largest_win": 200.0,
            "largest_loss": -80.0,
            "average_holding_time": 4.5,
            "total_signals": 20,
            "executed_signals": 18,
            "profitable_signals": 12,
            "losing_signals": 6,
        }
    
    def mock_execute_signal(signal_id, account_id, size, stop_loss=None, take_profit=None):
        return {
            "success": True,
            "message": "Signal executed successfully",
            "order_id": "order-123456",
        }
    
    monkeypatch.setattr(execution_db, "get_auto_trading_preferences", mock_get_auto_trading_preferences)
    monkeypatch.setattr(execution_db, "update_auto_trading_preferences", mock_update_auto_trading_preferences)
    monkeypatch.setattr(execution_db, "set_auto_trading_status", mock_set_auto_trading_status)
    monkeypatch.setattr(execution_db, "get_auto_trading_stats", mock_get_auto_trading_stats)
    monkeypatch.setattr(execution_db, "get_auto_trading_trades", mock_get_auto_trading_trades)
    monkeypatch.setattr(execution_db, "get_signals", mock_get_signals)
    monkeypatch.setattr(execution_db, "get_signal_count", mock_get_signal_count)
    monkeypatch.setattr(execution_db, "get_signal_by_id", mock_get_signal_by_id)
    monkeypatch.setattr(execution_db, "get_signal_performance", mock_get_signal_performance)
    monkeypatch.setattr(execution_db, "execute_signal", mock_execute_signal)
    
    # Mock strategy_db functions
    monkeypatch.setattr(strategy_db, "user_has_strategy_access", lambda user_id, strategy_id: True)
    
    def mock_start_optimization_job(user_id, strategy_id, instrument, timeframe, start_date,
                                   end_date, parameters, optimization_metric="profit_factor",
                                   population_size=50, generations=10, parallel_jobs=4):
        return {
            "id": TEST_JOB_ID,
            "status": "pending",
            "progress": 0.0,
            "message": "Job created",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
    
    def mock_get_optimization_job(job_id):
        return mock_optimization_job
    
    def mock_get_optimization_results(job_id):
        return {
            "results": mock_optimization_results,
        }
    
    def mock_start_walkforward_test(user_id, strategy_id, instrument, timeframe, start_date,
                                   end_date, parameters, window_size=90, step_size=30):
        return {
            "id": TEST_JOB_ID,
            "status": "pending",
            "progress": 0.0,
            "message": "Job created",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
    
    def mock_get_walkforward_job(job_id):
        return {
            **mock_optimization_job,
            "window_size": 90,
            "step_size": 30,
        }
    
    def mock_get_walkforward_results(job_id):
        return {
            "results": mock_walkforward_results,
            "overall_profit_factor": 1.7,
            "overall_sharpe_ratio": 1.3,
            "overall_win_rate": 58.0,
            "overall_total_trades": 85,
            "overall_net_profit": 850.0,
            "overall_max_drawdown": 4.0,
        }
    
    def mock_start_montecarlo_simulation(user_id, strategy_id, instrument, timeframe, start_date,
                                        end_date, parameters, simulations=1000, confidence_level=0.95):
        return {
            "id": TEST_JOB_ID,
            "status": "pending",
            "progress": 0.0,
            "message": "Job created",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
    
    def mock_get_montecarlo_job(job_id):
        return {
            **mock_optimization_job,
            "simulations": 1000,
            "confidence_level": 0.95,
        }
    
    def mock_get_montecarlo_results(job_id):
        return {
            "results": mock_montecarlo_results,
        }
    
    monkeypatch.setattr(strategy_db, "start_optimization_job", mock_start_optimization_job)
    monkeypatch.setattr(strategy_db, "get_optimization_job", mock_get_optimization_job)
    monkeypatch.setattr(strategy_db, "get_optimization_results", mock_get_optimization_results)
    monkeypatch.setattr(strategy_db, "start_walkforward_test", mock_start_walkforward_test)
    monkeypatch.setattr(strategy_db, "get_walkforward_job", mock_get_walkforward_job)
    monkeypatch.setattr(strategy_db, "get_walkforward_results", mock_get_walkforward_results)
    monkeypatch.setattr(strategy_db, "start_montecarlo_simulation", mock_start_montecarlo_simulation)
    monkeypatch.setattr(strategy_db, "get_montecarlo_job", mock_get_montecarlo_job)
    monkeypatch.setattr(strategy_db, "get_montecarlo_results", mock_get_montecarlo_results) 
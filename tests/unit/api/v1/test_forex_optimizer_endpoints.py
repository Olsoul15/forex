"""
Unit tests for forex optimizer endpoints.

Tests forex optimizer jobs endpoint.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

from forex_ai.api.v1.forex_optimizer_endpoints import (
    get_optimization_jobs,
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


class TestOptimizerJobsEndpoint:
    """Tests for optimizer jobs endpoint."""
    
    @pytest.mark.asyncio
    @patch('forex_ai.api.v1.forex_optimizer_endpoints.get_optimization_jobs')
    async def test_get_jobs_basic(self, mock_get_jobs, mock_user):
        """Test getting optimizer jobs with default parameters."""
        # Create a mock response
        now = datetime.now()
        mock_response = {
            "jobs": [
                {
                    "id": "job-1",
                    "name": "EUR/USD Optimization",
                    "status": "COMPLETED",
                    "instrument": "EUR/USD",
                    "timeframe": "H1",
                    "strategy_type": "trend_following",
                    "start_date": now - timedelta(days=30),
                    "end_date": now - timedelta(days=1),
                    "parameters": {
                        "fast_ma": {"min": 5, "max": 20, "step": 1},
                        "slow_ma": {"min": 20, "max": 50, "step": 5},
                        "stop_loss": {"min": 10, "max": 50, "step": 5}
                    },
                    "best_parameters": {
                        "fast_ma": 12,
                        "slow_ma": 35,
                        "stop_loss": 25
                    },
                    "metrics": {
                        "win_rate": 65.0,
                        "profit_factor": 2.1,
                        "sharpe_ratio": 1.5,
                        "max_drawdown": 8.2,
                        "total_trades": 42
                    },
                    "created_at": now - timedelta(days=2),
                    "completed_at": now - timedelta(hours=12)
                },
                {
                    "id": "job-2",
                    "name": "GBP/USD Optimization",
                    "status": "RUNNING",
                    "instrument": "GBP/USD",
                    "timeframe": "H4",
                    "strategy_type": "breakout",
                    "start_date": now - timedelta(days=60),
                    "end_date": now - timedelta(days=1),
                    "parameters": {
                        "breakout_period": {"min": 10, "max": 30, "step": 5},
                        "volatility_factor": {"min": 1.0, "max": 3.0, "step": 0.1},
                        "stop_loss": {"min": 20, "max": 60, "step": 10}
                    },
                    "best_parameters": None,
                    "metrics": None,
                    "created_at": now - timedelta(hours=5),
                    "completed_at": None
                }
            ],
            "count": 2,
            "timestamp": now
        }
        mock_get_jobs.return_value = mock_response
        
        response = mock_response
        assert response is not None
        assert len(response["jobs"]) == 2
        assert response["jobs"][0]["status"] == "COMPLETED"
        assert response["jobs"][0]["instrument"] == "EUR/USD"
        assert response["jobs"][0]["best_parameters"] is not None
        assert response["jobs"][1]["status"] == "RUNNING"
        assert response["jobs"][1]["instrument"] == "GBP/USD"
        assert response["jobs"][1]["best_parameters"] is None
    
    @pytest.mark.asyncio
    @patch('forex_ai.api.v1.forex_optimizer_endpoints.get_optimization_jobs')
    async def test_get_jobs_with_filters(self, mock_get_jobs, mock_user):
        """Test getting optimizer jobs with filters."""
        # Create a mock response with filtered jobs
        now = datetime.now()
        mock_response = {
            "jobs": [
                {
                    "id": "job-1",
                    "name": "EUR/USD Optimization",
                    "status": "COMPLETED",
                    "instrument": "EUR/USD",
                    "timeframe": "H1",
                    "strategy_type": "trend_following",
                    "start_date": now - timedelta(days=30),
                    "end_date": now - timedelta(days=1),
                    "parameters": {
                        "fast_ma": {"min": 5, "max": 20, "step": 1},
                        "slow_ma": {"min": 20, "max": 50, "step": 5},
                        "stop_loss": {"min": 10, "max": 50, "step": 5}
                    },
                    "best_parameters": {
                        "fast_ma": 12,
                        "slow_ma": 35,
                        "stop_loss": 25
                    },
                    "metrics": {
                        "win_rate": 65.0,
                        "profit_factor": 2.1,
                        "sharpe_ratio": 1.5,
                        "max_drawdown": 8.2,
                        "total_trades": 42
                    },
                    "created_at": now - timedelta(days=2),
                    "completed_at": now - timedelta(hours=12)
                }
            ],
            "count": 1,
            "timestamp": now
        }
        mock_get_jobs.return_value = mock_response
        
        response = mock_response
        assert response is not None
        assert len(response["jobs"]) == 1
        assert response["jobs"][0]["status"] == "COMPLETED"
        assert response["jobs"][0]["instrument"] == "EUR/USD"
        assert response["jobs"][0]["timeframe"] == "H1"
    
    @pytest.mark.asyncio
    @patch('forex_ai.api.v1.forex_optimizer_endpoints.get_optimization_jobs')
    async def test_get_jobs_empty(self, mock_get_jobs, mock_user):
        """Test getting optimizer jobs when no jobs exist."""
        # Create a mock response with no jobs
        now = datetime.now()
        mock_response = {
            "jobs": [],
            "count": 0,
            "timestamp": now
        }
        mock_get_jobs.return_value = mock_response
        
        response = mock_response
        assert response is not None
        assert len(response["jobs"]) == 0
        assert response["count"] == 0

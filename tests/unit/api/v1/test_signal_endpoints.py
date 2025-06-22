"""
Unit tests for signal endpoints.

Tests signal history and performance endpoints.
"""

import pytest
from datetime import datetime, timedelta
from fastapi import HTTPException, status, Query
from unittest.mock import patch

from forex_ai.api.v1.signal_endpoints import (
    get_signal_history,
    get_signal_performance,
    SignalPerformanceResponse,
    SignalPerformanceMetrics,
    SignalHistoryResponse,
    SignalModel
)
from forex_ai.backend_api.db import execution_db


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
def mock_execution_db(monkeypatch):
    """Mock execution database functions."""
    
    def mock_get_signals(
        user_id,
        strategy_id=None,
        instrument=None,
        timeframe=None,
        status=None,
        start_date=None,
        end_date=None,
        limit=50,
        offset=0,
    ):
        """Mock get_signals function."""
        if user_id == "test-user-id":
            now = datetime.now()
            signals = []
            
            # Generate mock signals
            for i in range(min(limit, 10)):
                signal_time = now - timedelta(days=i)
                signals.append({
                    "id": f"signal-{i}",
                    "strategy_id": strategy_id or "default-strategy",
                    "strategy_name": "Mock Strategy",
                    "instrument": instrument or "EUR/USD",
                    "timeframe": timeframe or "H1",
                    "direction": "BUY" if i % 2 == 0 else "SELL",
                    "entry_price": 1.1000 + (i * 0.0010),
                    "stop_loss": 1.0950 + (i * 0.0010),
                    "take_profit": 1.1050 + (i * 0.0010),
                    "risk_reward_ratio": 2.0,
                    "confidence": 0.75,
                    "signal_time": signal_time,
                    "expiration_time": signal_time + timedelta(days=1),
                    "status": status or ("CLOSED" if i < 5 else "OPEN"),
                    "notes": "Mock signal",
                    "executed": i < 5,
                    "execution_time": signal_time + timedelta(hours=1) if i < 5 else None,
                    "execution_price": 1.1005 + (i * 0.0010) if i < 5 else None,
                    "closed": i < 3,
                    "close_time": signal_time + timedelta(hours=5) if i < 3 else None,
                    "close_price": 1.1020 + (i * 0.0010) if i < 3 else None,
                    "profit_loss": 15.0 if i < 3 else None,
                    "profit_loss_pips": 15.0 if i < 3 else None,
                })
            
            # Filter by date if provided
            if start_date:
                signals = [s for s in signals if s["signal_time"] >= start_date]
            if end_date:
                signals = [s for s in signals if s["signal_time"] <= end_date]
            
            return signals
        return []
    
    def mock_get_signal_count(
        user_id,
        strategy_id=None,
        instrument=None,
        timeframe=None,
        status=None,
        start_date=None,
        end_date=None,
    ):
        """Mock get_signal_count function."""
        if user_id == "test-user-id":
            return 10
        return 0
    
    def mock_get_signal_performance(
        user_id,
        strategy_id=None,
        instrument=None,
        timeframe=None,
        period="1m",
    ):
        """Mock get_signal_performance function."""
        if user_id == "test-user-id":
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
        return None
    
    # Apply monkeypatches
    monkeypatch.setattr(execution_db, "get_signals", mock_get_signals)
    monkeypatch.setattr(execution_db, "get_signal_count", mock_get_signal_count)
    monkeypatch.setattr(execution_db, "get_signal_performance", mock_get_signal_performance)
    
    return {
        "get_signals": mock_get_signals,
        "get_signal_count": mock_get_signal_count,
        "get_signal_performance": mock_get_signal_performance,
    }


class TestSignalHistoryEndpoint:
    """Tests for signal history endpoint."""
    
    @pytest.mark.asyncio
    @patch('forex_ai.api.v1.signal_endpoints.get_signal_history')
    async def test_get_history_basic(self, mock_get_history, mock_user, mock_execution_db):
        """Test getting signal history with default parameters."""
        # Create a mock response
        now = datetime.now()
        signals = []
        for i in range(10):
            signal_time = now - timedelta(days=i)
            signals.append(SignalModel(
                id=f"signal-{i}",
                strategyId="default-strategy",
                strategyName="Mock Strategy",
                instrument="EUR/USD",
                timeframe="H1",
                direction="BUY" if i % 2 == 0 else "SELL",
                entry_price=1.1000 + (i * 0.0010),
                stop_loss=1.0950 + (i * 0.0010),
                take_profit=1.1050 + (i * 0.0010),
                risk_reward_ratio=2.0,
                confidence=0.75,
                signal_time=signal_time,
                expiration_time=signal_time + timedelta(days=1),
                status="CLOSED" if i < 5 else "OPEN",
                notes="Mock signal",
                executed=i < 5,
                execution_time=signal_time + timedelta(hours=1) if i < 5 else None,
                execution_price=1.1005 + (i * 0.0010) if i < 5 else None,
                closed=i < 3,
                close_time=signal_time + timedelta(hours=5) if i < 3 else None,
                close_price=1.1020 + (i * 0.0010) if i < 3 else None,
                profit_loss=15.0 if i < 3 else None,
                profit_loss_pips=15.0 if i < 3 else None,
            ))
        
        mock_response = SignalHistoryResponse(
            signals=signals,
            count=10,
            timestamp=now
        )
        mock_get_history.return_value = mock_response
        
        response = mock_response
        assert response is not None
        assert response.count == 10
        assert len(response.signals) == 10
        assert response.signals[0].instrument == "EUR/USD"
        assert response.signals[0].strategyId == "default-strategy"
    
    @pytest.mark.asyncio
    @patch('forex_ai.api.v1.signal_endpoints.get_signal_history')
    async def test_get_history_with_filters(self, mock_get_history, mock_user, mock_execution_db):
        """Test getting signal history with filters."""
        # Create a mock response
        now = datetime.now()
        signals = []
        for i in range(5):
            signal_time = now - timedelta(days=i)
            signals.append(SignalModel(
                id=f"signal-{i}",
                strategyId="test-strategy",
                strategyName="Test Strategy",
                instrument="GBP/USD",
                timeframe="M5",
                direction="BUY" if i % 2 == 0 else "SELL",
                entry_price=1.2000 + (i * 0.0010),
                stop_loss=1.1950 + (i * 0.0010),
                take_profit=1.2050 + (i * 0.0010),
                risk_reward_ratio=2.0,
                confidence=0.75,
                signal_time=signal_time,
                expiration_time=signal_time + timedelta(days=1),
                status="CLOSED",
                notes="Mock signal with filters",
                executed=True,
                execution_time=signal_time + timedelta(hours=1),
                execution_price=1.2005 + (i * 0.0010),
                closed=True,
                close_time=signal_time + timedelta(hours=5),
                close_price=1.2020 + (i * 0.0010),
                profit_loss=15.0,
                profit_loss_pips=15.0,
            ))
        
        mock_response = SignalHistoryResponse(
            signals=signals,
            count=5,
            timestamp=now
        )
        mock_get_history.return_value = mock_response
        
        response = mock_response
        assert response is not None
        assert all(signal.strategyId == "test-strategy" for signal in response.signals)
        assert all(signal.instrument == "GBP/USD" for signal in response.signals)
        assert all(signal.timeframe == "M5" for signal in response.signals)
        assert all(signal.status == "CLOSED" for signal in response.signals)
    
    @pytest.mark.asyncio
    @patch('forex_ai.api.v1.signal_endpoints.get_signal_history')
    async def test_get_history_with_pagination(self, mock_get_history, mock_user, mock_execution_db):
        """Test getting signal history with pagination."""
        # Create a mock response
        now = datetime.now()
        signals = []
        for i in range(5):
            signal_time = now - timedelta(days=i+5)  # Offset to simulate pagination
            signals.append(SignalModel(
                id=f"signal-{i+5}",  # Offset IDs to simulate pagination
                strategyId="default-strategy",
                strategyName="Mock Strategy",
                instrument="EUR/USD",
                timeframe="H1",
                direction="BUY" if i % 2 == 0 else "SELL",
                entry_price=1.1000 + ((i+5) * 0.0010),
                stop_loss=1.0950 + ((i+5) * 0.0010),
                take_profit=1.1050 + ((i+5) * 0.0010),
                risk_reward_ratio=2.0,
                confidence=0.75,
                signal_time=signal_time,
                expiration_time=signal_time + timedelta(days=1),
                status="OPEN",
                notes="Mock signal with pagination",
                executed=False,
                execution_time=None,
                execution_price=None,
                closed=False,
                close_time=None,
                close_price=None,
                profit_loss=None,
                profit_loss_pips=None,
            ))
        
        mock_response = SignalHistoryResponse(
            signals=signals,
            count=10,  # Total count is still 10
            timestamp=now
        )
        mock_get_history.return_value = mock_response
        
        response = mock_response
        assert response is not None
        assert len(response.signals) == 5
    
    @pytest.mark.asyncio
    @patch('forex_ai.api.v1.signal_endpoints.get_signal_history')
    async def test_get_history_with_date_range(self, mock_get_history, mock_user, mock_execution_db):
        """Test getting signal history with date range."""
        now = datetime.now()
        start_date = now - timedelta(days=5)
        end_date = now - timedelta(days=1)
        
        # Create a mock response with signals in the date range
        signals = []
        for i in range(5):
            signal_time = now - timedelta(days=i+1)  # Between 1 and 5 days ago
            signals.append(SignalModel(
                id=f"signal-{i}",
                strategyId="default-strategy",
                strategyName="Mock Strategy",
                instrument="EUR/USD",
                timeframe="H1",
                direction="BUY" if i % 2 == 0 else "SELL",
                entry_price=1.1000 + (i * 0.0010),
                stop_loss=1.0950 + (i * 0.0010),
                take_profit=1.1050 + (i * 0.0010),
                risk_reward_ratio=2.0,
                confidence=0.75,
                signal_time=signal_time,
                expiration_time=signal_time + timedelta(days=1),
                status="CLOSED" if i < 3 else "OPEN",
                notes="Mock signal with date range",
                executed=i < 3,
                execution_time=signal_time + timedelta(hours=1) if i < 3 else None,
                execution_price=1.1005 + (i * 0.0010) if i < 3 else None,
                closed=i < 2,
                close_time=signal_time + timedelta(hours=5) if i < 2 else None,
                close_price=1.1020 + (i * 0.0010) if i < 2 else None,
                profit_loss=15.0 if i < 2 else None,
                profit_loss_pips=15.0 if i < 2 else None,
            ))
        
        mock_response = SignalHistoryResponse(
            signals=signals,
            count=5,
            timestamp=now
        )
        mock_get_history.return_value = mock_response
        
        response = mock_response
        assert response is not None
        assert all(signal.signal_time >= start_date for signal in response.signals)
        assert all(signal.signal_time <= end_date for signal in response.signals)


class TestSignalPerformanceEndpoint:
    """Tests for signal performance endpoint."""
    
    @pytest.mark.asyncio
    @patch('forex_ai.api.v1.signal_endpoints.get_signal_performance')
    async def test_get_performance_basic(self, mock_get_performance, mock_user, mock_execution_db):
        """Test getting signal performance with default parameters."""
        # Create a mock response
        now = datetime.now()
        mock_metrics = SignalPerformanceMetrics(
            win_rate=65.0,
            profit_factor=2.0,
            average_win=60.0,
            average_loss=-30.0,
            largest_win=200.0,
            largest_loss=-80.0,
            average_holding_time=4.5,
            total_signals=20,
            executed_signals=18,
            profitable_signals=12,
            losing_signals=6,
        )
        
        mock_response = SignalPerformanceResponse(
            strategyId=None,
            instrument=None,
            timeframe=None,
            period="1m",
            start_date=now - timedelta(days=30),
            end_date=now,
            metrics=mock_metrics,
            timestamp=now
        )
        mock_get_performance.return_value = mock_response
        
        response = mock_response
        assert response is not None
        assert response.metrics.win_rate == 65.0
        assert response.metrics.profit_factor == 2.0
        assert response.metrics.total_signals == 20
        assert response.metrics.profitable_signals == 12
        assert response.metrics.losing_signals == 6
    
    @pytest.mark.asyncio
    @patch('forex_ai.api.v1.signal_endpoints.get_signal_performance')
    async def test_get_performance_with_filters(self, mock_get_performance, mock_user, mock_execution_db):
        """Test getting signal performance with filters."""
        # Create a mock response
        now = datetime.now()
        mock_metrics = SignalPerformanceMetrics(
            win_rate=65.0,
            profit_factor=2.0,
            average_win=60.0,
            average_loss=-30.0,
            largest_win=200.0,
            largest_loss=-80.0,
            average_holding_time=4.5,
            total_signals=20,
            executed_signals=18,
            profitable_signals=12,
            losing_signals=6,
        )
        
        mock_response = SignalPerformanceResponse(
            strategyId="test-strategy",
            instrument="GBP/USD",
            timeframe="M5",
            period="1m",
            start_date=now - timedelta(days=30),
            end_date=now,
            metrics=mock_metrics,
            timestamp=now
        )
        mock_get_performance.return_value = mock_response
        
        response = mock_response
        assert response is not None
        assert response.metrics.win_rate == 65.0
        assert response.metrics.profit_factor == 2.0
        assert response.strategyId == "test-strategy"
        assert response.instrument == "GBP/USD"
        assert response.timeframe == "M5"
    
    @pytest.mark.asyncio
    @patch('forex_ai.api.v1.signal_endpoints.get_signal_performance')
    async def test_get_performance_different_periods(self, mock_get_performance, mock_user, mock_execution_db):
        """Test getting performance for different time periods."""
        periods = ["1d", "1w", "1m", "3m", "6m", "1y", "all"]
        
        for period in periods:
            # Create a mock response for each period
            now = datetime.now()
            
            # Determine start date based on period
            if period == "1d":
                start_date = now - timedelta(days=1)
            elif period == "1w":
                start_date = now - timedelta(weeks=1)
            elif period == "1m":
                start_date = now - timedelta(days=30)
            elif period == "3m":
                start_date = now - timedelta(days=90)
            elif period == "6m":
                start_date = now - timedelta(days=180)
            elif period == "1y":
                start_date = now - timedelta(days=365)
            else:
                start_date = now - timedelta(days=365)  # Default for "all"
            
            mock_metrics = SignalPerformanceMetrics(
                win_rate=65.0,
                profit_factor=2.0,
                average_win=60.0,
                average_loss=-30.0,
                largest_win=200.0,
                largest_loss=-80.0,
                average_holding_time=4.5,
                total_signals=20,
                executed_signals=18,
                profitable_signals=12,
                losing_signals=6,
            )
            
            mock_response = SignalPerformanceResponse(
                strategyId=None,
                instrument=None,
                timeframe=None,
                period=period,
                start_date=start_date,
                end_date=now,
                metrics=mock_metrics,
                timestamp=now
            )
            mock_get_performance.return_value = mock_response
            
            response = mock_response
            assert response is not None
            assert response.period == period

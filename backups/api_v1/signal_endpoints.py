"""
Signal API Endpoints for Forex AI Trading System.

This module provides FastAPI endpoints for managing trading signals,
including signal history, performance, and execution.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Query, Path, Body
from pydantic import BaseModel, Field

from forex_ai.auth.supabase import get_current_user
from forex_ai.backend_api.db import execution_db, strategy_db
from forex_ai.custom_types import TradingSignal

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/signals", tags=["signals"])

# Models
class SignalModel(BaseModel):
    """Trading signal model."""
    id: str
    strategyId: str
    strategyName: str
    instrument: str
    timeframe: str
    direction: str
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    confidence: Optional[float] = None
    signal_time: datetime
    expiration_time: Optional[datetime] = None
    status: str
    notes: Optional[str] = None
    executed: bool = False
    execution_time: Optional[datetime] = None
    execution_price: Optional[float] = None
    closed: bool = False
    close_time: Optional[datetime] = None
    close_price: Optional[float] = None
    profit_loss: Optional[float] = None
    profit_loss_pips: Optional[float] = None

class SignalHistoryResponse(BaseModel):
    """Signal history response model."""
    signals: List[SignalModel]
    count: int
    timestamp: datetime

class SignalPerformanceMetrics(BaseModel):
    """Signal performance metrics model."""
    win_rate: float
    profit_factor: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    average_holding_time: float
    total_signals: int
    executed_signals: int
    profitable_signals: int
    losing_signals: int

class SignalPerformanceResponse(BaseModel):
    """Signal performance response model."""
    strategyId: Optional[str] = None
    instrument: Optional[str] = None
    timeframe: Optional[str] = None
    period: str
    start_date: datetime
    end_date: datetime
    metrics: SignalPerformanceMetrics
    timestamp: datetime

class ExecuteSignalRequest(BaseModel):
    """Execute signal request model."""
    signalId: str = Field(..., description="Signal ID to execute")
    accountId: str = Field(..., description="Account ID to execute the signal on")
    size: float = Field(..., description="Position size to execute")
    stop_loss: Optional[float] = Field(None, description="Stop loss price (overrides signal stop loss)")
    take_profit: Optional[float] = Field(None, description="Take profit price (overrides signal take profit)")

class ExecuteSignalResponse(BaseModel):
    """Execute signal response model."""
    signalId: str
    accountId: str
    orderId: str
    status: str
    message: str
    timestamp: datetime

# Endpoints
@router.get("/history", response_model=SignalHistoryResponse)
async def get_signal_history(
    strategyId: Optional[str] = Query(None, description="Filter by strategy ID"),
    instrument: Optional[str] = Query(None, description="Filter by instrument"),
    timeframe: Optional[str] = Query(None, description="Filter by timeframe"),
    status: Optional[str] = Query(None, description="Filter by status"),
    start_date: Optional[datetime] = Query(None, description="Start date for filtering"),
    end_date: Optional[datetime] = Query(None, description="End date for filtering"),
    limit: int = Query(50, description="Maximum number of signals to return"),
    offset: int = Query(0, description="Offset for pagination"),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Get signal history.

    Returns a list of trading signals based on the specified filters.
    """
    logger.info(f"Processing signal history request for user {current_user['id']}")

    try:
        # Get signal history
        signals = execution_db.get_signals(
            user_id=current_user["id"],
            strategy_id=strategyId,
            instrument=instrument,
            timeframe=timeframe,
            status=status,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            offset=offset,
        )
        
        if not signals:
            # Return empty list
            return SignalHistoryResponse(
                signals=[],
                count=0,
                timestamp=datetime.now(),
            )
        
        # Convert signals to model
        signal_models = []
        for signal in signals:
            signal_models.append(
                SignalModel(
                    id=signal.get("id", ""),
                    strategyId=signal.get("strategy_id", ""),
                    strategyName=signal.get("strategy_name", ""),
                    instrument=signal.get("instrument", ""),
                    timeframe=signal.get("timeframe", ""),
                    direction=signal.get("direction", ""),
                    entry_price=signal.get("entry_price", 0.0),
                    stop_loss=signal.get("stop_loss"),
                    take_profit=signal.get("take_profit"),
                    risk_reward_ratio=signal.get("risk_reward_ratio"),
                    confidence=signal.get("confidence"),
                    signal_time=signal.get("signal_time", datetime.now()),
                    expiration_time=signal.get("expiration_time"),
                    status=signal.get("status", ""),
                    notes=signal.get("notes"),
                    executed=signal.get("executed", False),
                    execution_time=signal.get("execution_time"),
                    execution_price=signal.get("execution_price"),
                    closed=signal.get("closed", False),
                    close_time=signal.get("close_time"),
                    close_price=signal.get("close_price"),
                    profit_loss=signal.get("profit_loss"),
                    profit_loss_pips=signal.get("profit_loss_pips"),
                )
            )
        
        # Get total count
        total_count = execution_db.get_signal_count(
            user_id=current_user["id"],
            strategy_id=strategyId,
            instrument=instrument,
            timeframe=timeframe,
            status=status,
            start_date=start_date,
            end_date=end_date,
        )
        
        return SignalHistoryResponse(
            signals=signal_models,
            count=total_count,
            timestamp=datetime.now(),
        )
    except Exception as e:
        logger.error(f"Error getting signal history: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting signal history: {str(e)}",
        )

@router.get("/performance", response_model=SignalPerformanceResponse)
async def get_signal_performance(
    strategyId: Optional[str] = Query(None, description="Filter by strategy ID"),
    instrument: Optional[str] = Query(None, description="Filter by instrument"),
    timeframe: Optional[str] = Query(None, description="Filter by timeframe"),
    period: str = Query("1m", description="Time period (1d, 1w, 1m, 3m, 6m, 1y, all)"),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Get signal performance.

    Returns performance metrics for signals based on the specified filters.
    """
    logger.info(f"Processing signal performance request for user {current_user['id']}")

    try:
        # Calculate date range
        now = datetime.now()
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
            # Default to all available data
            start_date = None
        
        # Get signal performance
        performance = execution_db.get_signal_performance(
            user_id=current_user["id"],
            strategy_id=strategyId,
            instrument=instrument,
            timeframe=timeframe,
            start_date=start_date,
            end_date=now,
        )
        
        if not performance:
            # Return empty performance
            return SignalPerformanceResponse(
                strategyId=strategyId,
                instrument=instrument,
                timeframe=timeframe,
                period=period,
                start_date=start_date or datetime.now() - timedelta(days=365),
                end_date=now,
                metrics=SignalPerformanceMetrics(
                    win_rate=0.0,
                    profit_factor=0.0,
                    average_win=0.0,
                    average_loss=0.0,
                    largest_win=0.0,
                    largest_loss=0.0,
                    average_holding_time=0.0,
                    total_signals=0,
                    executed_signals=0,
                    profitable_signals=0,
                    losing_signals=0,
                ),
                timestamp=datetime.now(),
            )
        
        # Create metrics model
        metrics = SignalPerformanceMetrics(
            win_rate=performance.get("win_rate", 0.0),
            profit_factor=performance.get("profit_factor", 0.0),
            average_win=performance.get("average_win", 0.0),
            average_loss=performance.get("average_loss", 0.0),
            largest_win=performance.get("largest_win", 0.0),
            largest_loss=performance.get("largest_loss", 0.0),
            average_holding_time=performance.get("average_holding_time", 0.0),
            total_signals=performance.get("total_signals", 0),
            executed_signals=performance.get("executed_signals", 0),
            profitable_signals=performance.get("profitable_signals", 0),
            losing_signals=performance.get("losing_signals", 0),
        )
        
        return SignalPerformanceResponse(
            strategyId=strategyId,
            instrument=instrument,
            timeframe=timeframe,
            period=period,
            start_date=start_date or performance.get("start_date", now - timedelta(days=365)),
            end_date=now,
            metrics=metrics,
            timestamp=datetime.now(),
        )
    except Exception as e:
        logger.error(f"Error getting signal performance: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting signal performance: {str(e)}",
        )

@router.post("/execute", response_model=ExecuteSignalResponse)
async def execute_signal(
    request: ExecuteSignalRequest = Body(...),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Execute a trading signal.

    Executes a trading signal on the specified account.
    """
    logger.info(f"Processing execute signal request for signal {request.signalId}")

    try:
        # Get signal
        signal = execution_db.get_signal_by_id(request.signalId)
        
        if not signal:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Signal {request.signalId} not found",
            )
        
        # Verify user has access to this signal
        if signal["user_id"] != current_user["id"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to this signal",
            )
        
        # Verify signal is valid for execution
        if signal["status"] not in ["active", "pending"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Signal {request.signalId} is not valid for execution (status: {signal['status']})",
            )
        
        # Verify signal is not expired
        if signal.get("expiration_time"):
            # Parse the expiration time if it's a string
            expiration_time = signal["expiration_time"]
            if isinstance(expiration_time, str):
                try:
                    expiration_time = datetime.fromisoformat(expiration_time.replace('Z', '+00:00'))
                except ValueError:
                    # If parsing fails, use a different format
                    try:
                        expiration_time = datetime.strptime(expiration_time, "%Y-%m-%dT%H:%M:%S.%f")
                    except ValueError:
                        # If that fails too, just use the current time to avoid comparison errors
                        expiration_time = datetime.now() + timedelta(hours=1)  # Assume not expired
            
            if expiration_time < datetime.now():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Signal {request.signalId} has expired",
                )
        
        # Verify signal is not already executed
        if signal.get("executed", False):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Signal {request.signalId} has already been executed",
            )
        
        # Execute signal
        result = execution_db.execute_signal(
            signal_id=request.signalId,
            account_id=request.accountId,
            size=request.size,
            stop_loss=request.stop_loss,
            take_profit=request.take_profit,
        )
        
        if not result or not result.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to execute signal: {result.get('message', 'Unknown error')}",
            )
        
        return ExecuteSignalResponse(
            signalId=request.signalId,
            accountId=request.accountId,
            orderId=result.get("order_id", ""),
            status="success",
            message=result.get("message", "Signal executed successfully"),
            timestamp=datetime.now(),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing signal: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing signal: {str(e)}",
        ) 
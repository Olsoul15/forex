"""
Signal API Endpoints for Forex AI Trading System.

This module provides FastAPI endpoints for managing trading signals,
including signal history, performance, and execution.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends, HTTPException, status, Query, Path, Body
from pydantic import BaseModel, Field

from forex_ai.auth.supabase import get_current_user
from forex_ai.data.storage.signal_repository import SignalRepository, Signal
from forex_ai.custom_types import TradingSignal

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/signals", tags=["signals"])

# Initialize repositories
signal_repository = SignalRepository()

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
        # Get signal history using the repository
        signals = await signal_repository.get_signals_history(
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
            # Handle both dictionary and Signal model objects
            if hasattr(signal, "model_dump"):
                signal_dict = signal.model_dump()
            else:
                signal_dict = signal
                
            signal_models.append(
                SignalModel(
                    id=signal_dict.get("id", ""),
                    strategyId=signal_dict.get("strategy_id", ""),
                    strategyName=signal_dict.get("strategy_name", ""),
                    instrument=signal_dict.get("instrument", ""),
                    timeframe=signal_dict.get("timeframe", ""),
                    direction=signal_dict.get("direction", ""),
                    entry_price=signal_dict.get("entry_price", 0.0),
                    stop_loss=signal_dict.get("stop_loss"),
                    take_profit=signal_dict.get("take_profit"),
                    risk_reward_ratio=signal_dict.get("risk_reward_ratio"),
                    confidence=signal_dict.get("confidence"),
                    signal_time=signal_dict.get("signal_time", datetime.now()),
                    expiration_time=signal_dict.get("expiration_time"),
                    status=signal_dict.get("status", ""),
                    notes=signal_dict.get("notes"),
                    executed=signal_dict.get("executed", False),
                    execution_time=signal_dict.get("execution_time"),
                    execution_price=signal_dict.get("execution_price"),
                    closed=signal_dict.get("closed", False),
                    close_time=signal_dict.get("close_time"),
                    close_price=signal_dict.get("close_price"),
                    profit_loss=signal_dict.get("profit_loss"),
                    profit_loss_pips=signal_dict.get("profit_loss_pips"),
                )
            )
        
        # Get total count
        total_count = len(signals)
        
        return SignalHistoryResponse(
            signals=signal_models,
            count=total_count,
            timestamp=datetime.now(),
        )
    except Exception as e:
        logger.error(f"Error getting signal history: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
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
        
        # Get signal performance using the repository
        performance = await signal_repository.get_signals_performance(
            user_id=current_user["id"],
            strategy_id=strategyId,
            instrument=instrument,
            timeframe=timeframe,
            period=period,
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
        metrics = SignalPerformanceMetrics(**performance)
        
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
            status_code=500,
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
        # Get signal by ID
        signal = await signal_repository.get_by_id(request.signalId)
        
        if not signal:
            raise HTTPException(
                status_code=404,
                detail=f"Signal {request.signalId} not found",
            )
        
        # Convert to dictionary if it's a model
        if hasattr(signal, "model_dump"):
            signal_dict = signal.model_dump()
        else:
            signal_dict = signal
            
        # Verify signal is valid for execution
        if signal_dict["status"] not in ["active", "pending"]:
            raise HTTPException(
                status_code=400,
                detail=f"Signal {request.signalId} is not valid for execution (status: {signal_dict['status']})",
            )
        
        # Verify signal is not expired - FIXED TIMEZONE HANDLING
        if signal_dict.get("expiration_time"):
            # Always work with UTC timezone for consistency
            now = datetime.now(timezone.utc)
            
            # Parse and normalize expiration time to UTC
            expiration_time = signal_dict["expiration_time"]
            if isinstance(expiration_time, str):
                try:
                    # Handle ISO format with timezone
                    expiration_time = datetime.fromisoformat(expiration_time.replace('Z', '+00:00'))
                except ValueError:
                    try:
                        # Handle other datetime formats and assume UTC
                        expiration_time = datetime.strptime(expiration_time, "%Y-%m-%dT%H:%M:%S.%f")
                        expiration_time = expiration_time.replace(tzinfo=timezone.utc)
                    except ValueError:
                        # If all parsing fails, set a future time to allow execution
                        expiration_time = now + timedelta(hours=1)
            
            # Ensure timezone awareness - if naive datetime, assume UTC
            if isinstance(expiration_time, datetime) and expiration_time.tzinfo is None:
                expiration_time = expiration_time.replace(tzinfo=timezone.utc)
                
            # Only check expiration if we have a valid datetime
            if expiration_time < now:
                logger.warning(f"Signal {request.signalId} has expired: {expiration_time} < {now}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Signal {request.signalId} has expired",
                )
        
        # Update signal status
        update_data = {
            "status": "executed",
            "executed": True,
            "execution_time": datetime.now().isoformat(),
            "execution_price": signal_dict.get("entry_price"),  # Use current price in real implementation
        }
        
        updated = await signal_repository.update(request.signalId, update_data)
        
        if not updated:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to update signal status",
            )
        
        # In a real implementation, we would create an order with the broker here
        
        return ExecuteSignalResponse(
            signalId=request.signalId,
            accountId=request.accountId,
            orderId=f"order-{datetime.now().timestamp()}",  # Generate mock order ID
            status="success",
            message="Signal executed successfully",
            timestamp=datetime.now(),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing signal: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error executing signal: {str(e)}",
        ) 
"""
Auto-Trading API Endpoints for Forex AI Trading System.

This module provides FastAPI endpoints for managing auto-trading functionality,
including preferences, enabling/disabling auto-trading, and viewing stats.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Query, Path, Body
from pydantic import BaseModel, Field

from forex_ai.auth.supabase import get_current_user
from forex_ai.backend_api.db import account_db, execution_db

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/auto-trading", tags=["auto-trading"])

# Models
class AutoTradingPreference(BaseModel):
    """Auto-trading preference model."""
    enabled: bool = Field(False, description="Whether auto-trading is enabled")
    risk_per_trade: float = Field(1.0, description="Risk per trade as percentage of account balance")
    max_daily_trades: int = Field(5, description="Maximum number of trades per day")
    max_open_trades: int = Field(3, description="Maximum number of open trades at once")
    allowed_instruments: List[str] = Field([], description="List of allowed instruments (empty for all)")
    trading_hours_start: str = Field("00:00", description="Trading hours start time (UTC)")
    trading_hours_end: str = Field("23:59", description="Trading hours end time (UTC)")
    trading_days: List[int] = Field([0, 1, 2, 3, 4], description="Trading days (0=Monday, 6=Sunday)")
    min_win_rate: float = Field(50.0, description="Minimum win rate required for strategies")
    min_profit_factor: float = Field(1.2, description="Minimum profit factor required for strategies")
    stop_loss_required: bool = Field(True, description="Whether stop loss is required for trades")
    take_profit_required: bool = Field(True, description="Whether take profit is required for trades")

class AutoTradingPreferencesRequest(BaseModel):
    """Auto-trading preferences request model."""
    userId: str = Field(..., description="User ID")
    preferences: AutoTradingPreference

class AutoTradingPreferencesResponse(BaseModel):
    """Auto-trading preferences response model."""
    userId: str
    preferences: AutoTradingPreference
    timestamp: datetime

class AutoTradingEnableRequest(BaseModel):
    """Auto-trading enable request model."""
    userId: str = Field(..., description="User ID")
    accountId: str = Field(..., description="Account ID")

class AutoTradingDisableRequest(BaseModel):
    """Auto-trading disable request model."""
    userId: str = Field(..., description="User ID")
    accountId: str = Field(..., description="Account ID")

class AutoTradingStatusResponse(BaseModel):
    """Auto-trading status response model."""
    userId: str
    accountId: str
    enabled: bool
    message: str
    timestamp: datetime

class AutoTradingTrade(BaseModel):
    """Auto-trading trade model."""
    id: str
    accountId: str
    instrument: str
    direction: str
    size: float
    entry_price: float
    exit_price: Optional[float] = None
    profit_loss: Optional[float] = None
    status: str
    strategy_id: str
    strategy_name: str
    entry_time: datetime
    exit_time: Optional[datetime] = None

class AutoTradingStatsResponse(BaseModel):
    """Auto-trading stats response model."""
    userId: str
    total_trades: int
    successful_trades: int
    failed_trades: int
    win_rate: float
    profit_loss: float
    daily_stats: Dict[str, Any]
    timestamp: datetime

class AutoTradingTradesResponse(BaseModel):
    """Auto-trading trades response model."""
    userId: str
    trades: List[AutoTradingTrade]
    count: int
    timestamp: datetime

# Endpoints
@router.get("/preferences", response_model=AutoTradingPreferencesResponse)
async def get_auto_trading_preferences(
    userId: str = Query(..., description="User ID"),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Get auto-trading preferences.

    Returns the auto-trading preferences for the specified user.
    """
    logger.info(f"Processing auto-trading preferences request for user {userId}")

    # Verify user has access
    if current_user["id"] != userId:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this user's preferences",
        )
    
    try:
        # Get preferences
        preferences = execution_db.get_auto_trading_preferences(userId)
        
        if not preferences:
            # Return default preferences
            preferences = AutoTradingPreference().dict()
        
        return AutoTradingPreferencesResponse(
            userId=userId,
            preferences=AutoTradingPreference(**preferences),
            timestamp=datetime.now(),
        )
    except Exception as e:
        logger.error(f"Error getting auto-trading preferences: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting auto-trading preferences: {str(e)}",
        )

@router.put("/preferences", response_model=AutoTradingPreferencesResponse)
async def update_auto_trading_preferences(
    request: AutoTradingPreferencesRequest = Body(...),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Update auto-trading preferences.

    Updates the auto-trading preferences for the specified user.
    """
    logger.info(f"Processing update auto-trading preferences request for user {request.userId}")

    # Verify user has access
    if current_user["id"] != request.userId:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to update this user's preferences",
        )
    
    try:
        # Update preferences
        updated_preferences = execution_db.update_auto_trading_preferences(
            request.userId, request.preferences.dict()
        )
        
        return AutoTradingPreferencesResponse(
            userId=request.userId,
            preferences=AutoTradingPreference(**updated_preferences),
            timestamp=datetime.now(),
        )
    except Exception as e:
        logger.error(f"Error updating auto-trading preferences: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating auto-trading preferences: {str(e)}",
        )

@router.post("/enable", response_model=AutoTradingStatusResponse)
async def enable_auto_trading(
    request: AutoTradingEnableRequest = Body(...),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Enable auto-trading.

    Enables auto-trading for the specified user and account.
    """
    logger.info(f"Processing enable auto-trading request for user {request.userId}, account {request.accountId}")

    # Verify user has access
    if current_user["id"] != request.userId:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to enable auto-trading for this user",
        )
    
    # Verify user has access to this account
    if not account_db.user_has_account_access(request.userId, request.accountId):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this account",
        )
    
    try:
        # Enable auto-trading
        success = execution_db.set_auto_trading_status(request.userId, request.accountId, True)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to enable auto-trading",
            )
        
        return AutoTradingStatusResponse(
            userId=request.userId,
            accountId=request.accountId,
            enabled=True,
            message="Auto-trading enabled successfully",
            timestamp=datetime.now(),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error enabling auto-trading: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error enabling auto-trading: {str(e)}",
        )

@router.post("/disable", response_model=AutoTradingStatusResponse)
async def disable_auto_trading(
    request: AutoTradingDisableRequest = Body(...),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Disable auto-trading.

    Disables auto-trading for the specified user and account.
    """
    logger.info(f"Processing disable auto-trading request for user {request.userId}, account {request.accountId}")

    # Verify user has access
    if current_user["id"] != request.userId:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to disable auto-trading for this user",
        )
    
    # Verify user has access to this account
    if not account_db.user_has_account_access(request.userId, request.accountId):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this account",
        )
    
    try:
        # Disable auto-trading
        success = execution_db.set_auto_trading_status(request.userId, request.accountId, False)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to disable auto-trading",
            )
        
        return AutoTradingStatusResponse(
            userId=request.userId,
            accountId=request.accountId,
            enabled=False,
            message="Auto-trading disabled successfully",
            timestamp=datetime.now(),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error disabling auto-trading: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error disabling auto-trading: {str(e)}",
        )

@router.get("/stats", response_model=AutoTradingStatsResponse)
async def get_auto_trading_stats(
    userId: str = Query(..., description="User ID"),
    accountId: Optional[str] = Query(None, description="Account ID (optional)"),
    period: str = Query("1m", description="Time period (1d, 1w, 1m, 3m, 6m, 1y, all)"),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Get auto-trading stats.

    Returns statistics about auto-trading performance for the specified user.
    """
    logger.info(f"Processing auto-trading stats request for user {userId}")

    # Verify user has access
    if current_user["id"] != userId:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this user's stats",
        )
    
    try:
        # Get stats
        stats = execution_db.get_auto_trading_stats(userId, accountId, period)
        
        if not stats:
            # Return empty stats
            return AutoTradingStatsResponse(
                userId=userId,
                total_trades=0,
                successful_trades=0,
                failed_trades=0,
                win_rate=0.0,
                profit_loss=0.0,
                daily_stats={},
                timestamp=datetime.now(),
            )
        
        return AutoTradingStatsResponse(
            userId=userId,
            total_trades=stats.get("total_trades", 0),
            successful_trades=stats.get("successful_trades", 0),
            failed_trades=stats.get("failed_trades", 0),
            win_rate=stats.get("win_rate", 0.0),
            profit_loss=stats.get("profit_loss", 0.0),
            daily_stats=stats.get("daily_stats", {}),
            timestamp=datetime.now(),
        )
    except Exception as e:
        logger.error(f"Error getting auto-trading stats: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting auto-trading stats: {str(e)}",
        )

@router.get("/trades", response_model=AutoTradingTradesResponse)
async def get_recent_auto_trades(
    userId: str = Query(..., description="User ID"),
    accountId: Optional[str] = Query(None, description="Account ID (optional)"),
    limit: int = Query(10, description="Maximum number of trades to return"),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Get recent auto-trades.

    Returns recent trades executed by the auto-trading system for the specified user.
    """
    logger.info(f"Processing recent auto-trades request for user {userId}")

    # Verify user has access
    if current_user["id"] != userId:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this user's trades",
        )
    
    try:
        # Get trades
        trades = execution_db.get_auto_trading_trades(userId, accountId, limit)
        
        if not trades:
            # Return empty trades list
            return AutoTradingTradesResponse(
                userId=userId,
                trades=[],
                count=0,
                timestamp=datetime.now(),
            )
        
        # Convert trades to model
        trade_models = []
        for trade in trades:
            trade_models.append(
                AutoTradingTrade(
                    id=trade.get("id", ""),
                    accountId=trade.get("account_id", ""),
                    instrument=trade.get("instrument", ""),
                    direction=trade.get("direction", ""),
                    size=trade.get("size", 0.0),
                    entry_price=trade.get("entry_price", 0.0),
                    exit_price=trade.get("exit_price"),
                    profit_loss=trade.get("profit_loss"),
                    status=trade.get("status", ""),
                    strategy_id=trade.get("strategy_id", ""),
                    strategy_name=trade.get("strategy_name", ""),
                    entry_time=trade.get("entry_time", datetime.now()),
                    exit_time=trade.get("exit_time"),
                )
            )
        
        return AutoTradingTradesResponse(
            userId=userId,
            trades=trade_models,
            count=len(trade_models),
            timestamp=datetime.now(),
        )
    except Exception as e:
        logger.error(f"Error getting auto-trading trades: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting auto-trading trades: {str(e)}",
        ) 
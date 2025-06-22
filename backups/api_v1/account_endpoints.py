"""
Account API Endpoints for Forex AI Trading System.

This module provides FastAPI endpoints for accessing account information,
metrics, performance, balance, and equity.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Query, Path
from pydantic import BaseModel

from forex_ai.auth.supabase import get_current_user
from forex_ai.backend_api.db import account_db

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/account", tags=["account"])

# Models
class AccountMetricsResponse(BaseModel):
    """Account metrics response model."""
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    drawdown_max: float
    drawdown_current: float
    total_trades: int
    profitable_trades: int
    losing_trades: int
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    timestamp: datetime

class AccountPerformanceResponse(BaseModel):
    """Account performance response model."""
    daily_returns: Dict[str, float]
    cumulative_returns: Dict[str, float]
    monthly_returns: Dict[str, float]
    equity_curve: Dict[str, float]
    drawdown_curve: Dict[str, float]
    period: str
    start_date: datetime
    end_date: datetime
    timestamp: datetime

class AccountBalanceResponse(BaseModel):
    """Account balance response model."""
    balance: float
    currency: str
    timestamp: datetime

class AccountEquityResponse(BaseModel):
    """Account equity response model."""
    equity: float
    balance: float
    floating_pnl: float
    currency: str
    timestamp: datetime

# Endpoints
@router.get("/metrics", response_model=AccountMetricsResponse)
async def get_account_metrics(
    accountId: str = Query(..., description="Account ID"),
    period: str = Query("1m", description="Time period for metrics (1d, 1w, 1m, 3m, 6m, 1y, all)"),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Get account metrics.

    Returns performance metrics for the account including win rate, profit factor,
    Sharpe ratio, drawdown, and trade statistics.
    """
    logger.info(f"Processing account metrics request for account {accountId}, period {period}")

    try:
        # Verify user has access to this account
        if not account_db.user_has_account_access(current_user["id"], accountId):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to this account",
            )
        
        # Get account metrics
        metrics = account_db.get_account_metrics(accountId, period)
        
        if not metrics:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Metrics not found for account {accountId}",
            )
        
        # Create response
        return AccountMetricsResponse(
            win_rate=metrics.get("win_rate", 0.0),
            profit_factor=metrics.get("profit_factor", 0.0),
            sharpe_ratio=metrics.get("sharpe_ratio", 0.0),
            drawdown_max=metrics.get("drawdown_max", 0.0),
            drawdown_current=metrics.get("drawdown_current", 0.0),
            total_trades=metrics.get("total_trades", 0),
            profitable_trades=metrics.get("profitable_trades", 0),
            losing_trades=metrics.get("losing_trades", 0),
            average_win=metrics.get("average_win", 0.0),
            average_loss=metrics.get("average_loss", 0.0),
            largest_win=metrics.get("largest_win", 0.0),
            largest_loss=metrics.get("largest_loss", 0.0),
            timestamp=datetime.now(),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting account metrics: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting account metrics: {str(e)}",
        )

@router.get("/performance", response_model=AccountPerformanceResponse)
async def get_account_performance(
    accountId: str = Query(..., description="Account ID"),
    period: str = Query("1m", description="Time period for performance (1d, 1w, 1m, 3m, 6m, 1y, all)"),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Get account performance.

    Returns performance data for the account including daily returns, cumulative returns,
    monthly returns, equity curve, and drawdown curve.
    """
    logger.info(f"Processing account performance request for account {accountId}, period {period}")

    try:
        # Verify user has access to this account
        if not account_db.user_has_account_access(current_user["id"], accountId):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to this account",
            )
        
        # Get account performance
        performance = account_db.get_account_performance(accountId, period)
        
        if not performance:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Performance data not found for account {accountId}",
            )
        
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
            start_date = performance.get("start_date", now - timedelta(days=365))
        
        # Create response
        return AccountPerformanceResponse(
            daily_returns=performance.get("daily_returns", {}),
            cumulative_returns=performance.get("cumulative_returns", {}),
            monthly_returns=performance.get("monthly_returns", {}),
            equity_curve=performance.get("equity_curve", {}),
            drawdown_curve=performance.get("drawdown_curve", {}),
            period=period,
            start_date=start_date,
            end_date=now,
            timestamp=now,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting account performance: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting account performance: {str(e)}",
        )

@router.get("/balance", response_model=AccountBalanceResponse)
async def get_account_balance(
    accountId: str = Query(..., description="Account ID"),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Get account balance.

    Returns the current balance for the account.
    """
    logger.info(f"Processing account balance request for account {accountId}")

    try:
        # Verify user has access to this account
        if not account_db.user_has_account_access(current_user["id"], accountId):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to this account",
            )
        
        # Get account balance
        account = account_db.get_account_by_id(accountId)
        
        if not account:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Account {accountId} not found",
            )
        
        # Create response
        return AccountBalanceResponse(
            balance=account.get("balance", 0.0),
            currency=account.get("currency", "USD"),
            timestamp=datetime.now(),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting account balance: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting account balance: {str(e)}",
        )

@router.get("/equity", response_model=AccountEquityResponse)
async def get_account_equity(
    accountId: str = Query(..., description="Account ID"),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Get account equity.

    Returns the current equity for the account, which is the balance plus
    floating profit/loss from open positions.
    """
    logger.info(f"Processing account equity request for account {accountId}")

    try:
        # Verify user has access to this account
        if not account_db.user_has_account_access(current_user["id"], accountId):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to this account",
            )
        
        # Get account equity
        account = account_db.get_account_by_id(accountId)
        
        if not account:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Account {accountId} not found",
            )
        
        # Calculate floating P/L
        floating_pnl = account.get("unrealized_pl", 0.0)
        
        # Create response
        return AccountEquityResponse(
            equity=account.get("balance", 0.0) + floating_pnl,
            balance=account.get("balance", 0.0),
            floating_pnl=floating_pnl,
            currency=account.get("currency", "USD"),
            timestamp=datetime.now(),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting account equity: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting account equity: {str(e)}",
        ) 
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
from forex_ai.models.broker_models import BrokerType, OandaCredentials, SaveCredentialsRequest, SaveCredentialsResponse
from forex_ai.execution.oanda_api import OandaAPI
from forex_ai.data.connectors.oanda_handler import OandaDataHandler
from forex_ai.data.storage.account_repository import AccountRepository

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

    Returns the current balance for the specified account.
    """
    logger.info(f"Processing account balance request for account {accountId}")

    try:
        # Create repository
        account_repo = AccountRepository()
        
        # Get account
        account = await account_repo.get_by_id(accountId)
        
        # Check if account exists and user has access
        if not account:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Account {accountId} not found",
            )
            
        if account.user_id != current_user["id"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to this account",
            )
        
        # Create response
        return AccountBalanceResponse(
            balance=account.balance,
            currency=account.currency,
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

@router.post("/save-credentials", response_model=SaveCredentialsResponse)
async def save_broker_credentials(
    request: SaveCredentialsRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Save broker credentials.

    Returns a response indicating whether the credentials were saved successfully.
    """
    logger.info(f"Processing save broker credentials request for user {current_user['id']}")

    try:
        # Extract broker type from credentials
        if "broker_type" not in request.credentials:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Broker type not specified in credentials",
            )
        
        broker_type = request.credentials["broker_type"]
        
        # Validate required fields for OANDA
        if broker_type.lower() == "oanda":
            required_fields = ["access_token", "account_id"]
            missing_fields = [field for field in required_fields if field not in request.credentials]
            if missing_fields:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Missing required fields for OANDA: {', '.join(missing_fields)}",
                )
        
        # Save broker credentials
        result = account_db.save_broker_credentials(
            user_id=current_user["id"],
            broker_type=broker_type,
            credentials=request.credentials
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["message"],
            )
        
        # Create response
        return SaveCredentialsResponse(
            success=True,
            message="Broker credentials saved successfully",
            broker_type=broker_type,
            user_id=current_user["id"]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving broker credentials: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error saving broker credentials: {str(e)}",
        )

@router.get("/credentials/{broker_type}")
async def get_broker_credentials(
    broker_type: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Get broker credentials.

    Returns the broker credentials for the specified broker type.
    """
    logger.info(f"Processing get broker credentials request for user {current_user['id']}, broker {broker_type}")

    try:
        # Validate broker type
        try:
            broker_enum = BrokerType(broker_type.lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid broker type: {broker_type}",
            )
        
        # Get broker credentials
        credentials = account_db.get_broker_credentials(
            user_id=current_user["id"],
            broker_type=broker_enum
        )
        
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Credentials not found for broker type: {broker_type}",
            )
        
        # Create response
        return {
            "success": True,
            "broker_type": broker_type,
            "credentials": credentials,
            "timestamp": datetime.now()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting broker credentials: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting broker credentials: {str(e)}",
        )

def get_oanda_api_for_user(user_id: str) -> OandaAPI:
    """
    Create an OandaAPI instance with the user's credentials.
    
    Args:
        user_id: User ID to load credentials for
        
    Returns:
        OandaAPI instance
        
    Raises:
        HTTPException: If credentials are not found
    """
    try:
        # Get credentials from database
        credentials_dict = account_db.get_broker_credentials(user_id, "oanda")
        
        if not credentials_dict:
            logger.error(f"No OANDA credentials found for user {user_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="OANDA credentials not found for this user"
            )
            
        # Create OandaAPI instance
        return OandaAPI(user_id=user_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating OANDA API instance: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating OANDA API instance: {str(e)}"
        )

def get_oanda_data_handler_for_user(user_id: str) -> OandaDataHandler:
    """
    Create an OandaDataHandler instance with the user's credentials.
    
    Args:
        user_id: User ID to load credentials for
        
    Returns:
        OandaDataHandler instance
        
    Raises:
        HTTPException: If credentials are not found
    """
    try:
        # Get credentials from database
        credentials_dict = account_db.get_broker_credentials(user_id, "oanda")
        
        if not credentials_dict:
            logger.error(f"No OANDA credentials found for user {user_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="OANDA credentials not found for this user"
            )
            
        # Create OandaDataHandler instance
        return OandaDataHandler(user_id=user_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating OANDA data handler: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating OANDA data handler: {str(e)}"
        )

@router.get("/oanda/account-info")
async def get_oanda_account_info(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Get OANDA account information.

    Returns account information from OANDA for the current user.
    """
    logger.info(f"Processing OANDA account info request for user {current_user['id']}")

    try:
        # Create OANDA API instance with user credentials
        oanda_api = get_oanda_api_for_user(current_user["id"])
        
        # Get account info
        account_info = await oanda_api.get_account_info()
        
        # Create response
        return {
            "success": True,
            "account_info": account_info,
            "timestamp": datetime.now()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting OANDA account info: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting OANDA account info: {str(e)}",
        ) 
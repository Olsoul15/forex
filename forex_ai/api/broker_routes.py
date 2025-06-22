"""
Broker API routes for the Forex AI Trading System.

This module provides API endpoints for managing broker integrations,
including saving and retrieving broker credentials and account information.
"""

import logging
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Body, Path, Query

from forex_ai.models.broker_models import (
    BrokerType,
    OandaCredentials,
    BrokerAccountInfo,
    BrokerAccountList,
    SaveCredentialsRequest,
    SaveCredentialsResponse,
    DeleteCredentialsRequest,
    DeleteCredentialsResponse,
)
from forex_ai.execution.oanda_api import OandaAPI
from forex_ai.auth.supabase import get_current_user

# Configure logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/brokers", tags=["brokers"])

# In-memory storage for broker credentials (should be replaced with secure storage in production)
# This is just for development/demo purposes
_broker_credentials = {}


@router.post("/credentials", response_model=SaveCredentialsResponse)
async def save_broker_credentials(
    request: SaveCredentialsRequest = Body(...),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Save broker API credentials for a user.

    Args:
        request: Credentials data
        current_user: Current authenticated user

    Returns:
        SaveCredentialsResponse: Result of saving credentials
    """
    try:
        user_id = current_user.get("id")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid user information",
            )

        # Extract credentials from request
        creds_data = request.credentials
        broker_type = creds_data.get("broker_type")

        if not broker_type:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Broker type is required",
            )

        # Store credentials based on broker type
        if broker_type.lower() == BrokerType.OANDA:
            # Validate OANDA credentials
            try:
                oanda_creds = OandaCredentials(
                    broker_type=BrokerType.OANDA,
                    user_id=user_id,
                    access_token=creds_data.get("access_token"),
                    environment=creds_data.get("environment", "practice"),
                    default_account_id=creds_data.get("default_account_id"),
                )

                # Test the credentials by creating an API client and fetching accounts
                api = OandaAPI(oanda_creds)
                accounts = await api.get_accounts()
                
                if any("error" in account for account in accounts):
                    raise ValueError("Invalid OANDA credentials")

                # If no default account is specified, use the first one
                if not oanda_creds.default_account_id and accounts:
                    oanda_creds.default_account_id = accounts[0].get("id")

                # Store credentials (in a real app, encrypt these before storing)
                key = f"{user_id}:{broker_type}"
                _broker_credentials[key] = oanda_creds.dict()

                logger.info(f"Saved OANDA credentials for user {user_id}")

                return SaveCredentialsResponse(
                    success=True,
                    message="OANDA credentials saved successfully",
                    broker_type=BrokerType.OANDA,
                    user_id=user_id,
                )
            except Exception as e:
                logger.error(f"Error validating OANDA credentials: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid OANDA credentials: {str(e)}",
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported broker type: {broker_type}",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving broker credentials: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save broker credentials: {str(e)}",
        )


@router.get("/credentials", response_model=Dict[str, Any])
async def get_broker_credentials(
    broker_type: BrokerType = Query(..., description="Broker type"),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Get broker API credentials for the current user.

    Args:
        broker_type: Type of broker
        current_user: Current authenticated user

    Returns:
        Dict: Broker credentials (with sensitive data masked)
    """
    try:
        user_id = current_user.get("id")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid user information",
            )

        # Get credentials
        key = f"{user_id}:{broker_type}"
        creds = _broker_credentials.get(key)

        if not creds:
            return {"exists": False, "broker_type": broker_type}

        # Mask sensitive data
        if "access_token" in creds:
            token = creds["access_token"]
            if token and len(token) > 8:
                creds["access_token"] = token[:4] + "****" + token[-4:]

        return {"exists": True, "broker_type": broker_type, **creds}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving broker credentials: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve broker credentials: {str(e)}",
        )


@router.delete("/credentials", response_model=DeleteCredentialsResponse)
async def delete_broker_credentials(
    broker_type: BrokerType = Query(..., description="Broker type"),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Delete broker API credentials for the current user.

    Args:
        broker_type: Type of broker
        current_user: Current authenticated user

    Returns:
        DeleteCredentialsResponse: Result of deleting credentials
    """
    try:
        user_id = current_user.get("id")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid user information",
            )

        # Delete credentials
        key = f"{user_id}:{broker_type}"
        if key in _broker_credentials:
            del _broker_credentials[key]
            logger.info(f"Deleted {broker_type} credentials for user {user_id}")
            return DeleteCredentialsResponse(
                success=True,
                message=f"{broker_type} credentials deleted successfully",
            )
        else:
            return DeleteCredentialsResponse(
                success=False,
                message=f"No {broker_type} credentials found for this user",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting broker credentials: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete broker credentials: {str(e)}",
        )


@router.get("/accounts", response_model=BrokerAccountList)
async def get_broker_accounts(
    broker_type: BrokerType = Query(..., description="Broker type"),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Get accounts from the specified broker for the current user.

    Args:
        broker_type: Type of broker
        current_user: Current authenticated user

    Returns:
        BrokerAccountList: List of broker accounts
    """
    try:
        user_id = current_user.get("id")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid user information",
            )

        # Get credentials
        key = f"{user_id}:{broker_type}"
        creds_dict = _broker_credentials.get(key)

        if not creds_dict:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No {broker_type} credentials found for this user",
            )

        # Get accounts based on broker type
        if broker_type == BrokerType.OANDA:
            # Create OANDA credentials and API client
            oanda_creds = OandaCredentials(**creds_dict)
            api = OandaAPI(oanda_creds)
            
            # Get accounts
            accounts_data = await api.get_accounts()
            
            if any("error" in account for account in accounts_data):
                error_msg = next((account["error"] for account in accounts_data if "error" in account), "Unknown error")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Error retrieving OANDA accounts: {error_msg}",
                )
            
            # Format accounts
            accounts = []
            for account in accounts_data:
                accounts.append(
                    BrokerAccountInfo(
                        account_id=account.get("id", ""),
                        broker_type=BrokerType.OANDA,
                        name=account.get("name", ""),
                        currency=account.get("currency", "USD"),
                    )
                )
            
            return BrokerAccountList(accounts=accounts, count=len(accounts))
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported broker type: {broker_type}",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving broker accounts: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve broker accounts: {str(e)}",
        )


@router.get("/accounts/{account_id}", response_model=BrokerAccountInfo)
async def get_broker_account_info(
    account_id: str = Path(..., description="Account ID"),
    broker_type: BrokerType = Query(..., description="Broker type"),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Get detailed information about a specific broker account.

    Args:
        account_id: Account ID
        broker_type: Type of broker
        current_user: Current authenticated user

    Returns:
        BrokerAccountInfo: Account information
    """
    try:
        user_id = current_user.get("id")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid user information",
            )

        # Get credentials
        key = f"{user_id}:{broker_type}"
        creds_dict = _broker_credentials.get(key)

        if not creds_dict:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No {broker_type} credentials found for this user",
            )

        # Get account info based on broker type
        if broker_type == BrokerType.OANDA:
            # Create OANDA credentials and API client
            oanda_creds = OandaCredentials(**creds_dict)
            
            # Set the account ID we want to query
            oanda_creds.default_account_id = account_id
            
            api = OandaAPI(oanda_creds)
            
            # Get account info
            account_data = await api.get_account_info()
            
            if "error" in account_data:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Error retrieving OANDA account: {account_data['error']}",
                )
            
            # Format account info
            return BrokerAccountInfo(
                account_id=account_data.get("id", ""),
                broker_type=BrokerType.OANDA,
                balance=account_data.get("balance", 0.0),
                currency=account_data.get("currency", "USD"),
                margin_rate=account_data.get("margin_rate", 0.0),
                margin_used=account_data.get("margin_used", 0.0),
                margin_available=account_data.get("margin_available", 0.0),
                unrealized_pl=account_data.get("unrealized_pl", 0.0),
                realized_pl=account_data.get("realized_pl", 0.0),
                open_trade_count=account_data.get("open_trade_count", 0),
                pending_order_count=account_data.get("pending_order_count", 0),
                last_updated=account_data.get("last_updated", ""),
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported broker type: {broker_type}",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving broker account info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve broker account info: {str(e)}",
        ) 
"""
Broker API integration models for Forex AI Trading System.

This module defines the data models for broker API integration,
including credentials, account information, and broker-specific settings.
"""

from enum import Enum
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, field_validator


class BrokerType(str, Enum):
    """Supported broker types."""
    OANDA = "oanda"
    DEMO = "demo"
    # Add more brokers as they are supported


class OandaEnvironment(str, Enum):
    """OANDA API environments."""
    PRACTICE = "practice"
    LIVE = "live"


class BrokerCredentials(BaseModel):
    """Base model for broker API credentials."""
    broker_type: BrokerType
    user_id: str


class OandaCredentials(BrokerCredentials):
    """OANDA API credentials."""
    access_token: str
    environment: OandaEnvironment = OandaEnvironment.PRACTICE
    default_account_id: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "broker_type": "oanda",
                "user_id": "user-123",
                "access_token": "your-personal-access-token",
                "environment": "practice",
                "default_account_id": "123-456-789"
            }
        }
    
    @classmethod
    def from_env(cls, user_id: str = "env_user"):
        """
        Create OandaCredentials from environment variables.
        
        WARNING: This method is ONLY for direct development use via scripts.
        It should NEVER be used in API endpoints or as a fallback mechanism.
        In production, credentials must ALWAYS come from the database/API.
        
        Args:
            user_id: User ID to associate with the credentials
            
        Returns:
            OandaCredentials instance
            
        Raises:
            ValueError: If required environment variables are not set
        """
        from forex_ai.utils.config import get_env_var
        from forex_ai.config.settings import get_settings
        import logging
        import os
        
        logger = logging.getLogger(__name__)
        logger.warning("DEVELOPMENT ONLY: Using environment variables for OANDA credentials")
        
        # Check for required environment variables
        access_token = os.environ.get("OANDA_ACCESS_TOKEN") or os.environ.get("OANDA_API_KEY")
        account_id = os.environ.get("OANDA_ACCOUNT_ID")
        
        if not access_token:
            raise ValueError("OANDA_ACCESS_TOKEN or OANDA_API_KEY environment variable must be set")
        if not account_id:
            raise ValueError("OANDA_ACCOUNT_ID environment variable must be set")
        
        return cls(
            broker_type=BrokerType.OANDA,
            user_id=user_id,
            access_token=access_token,
            environment=OandaEnvironment.PRACTICE,  # Default to practice
            default_account_id=account_id
        )


class BrokerAccountInfo(BaseModel):
    """Broker account information."""
    account_id: str
    broker_type: BrokerType
    name: Optional[str] = None
    balance: float = 0.0
    currency: str = "USD"
    margin_rate: float = 0.0
    margin_used: float = 0.0
    margin_available: float = 0.0
    unrealized_pl: float = 0.0
    realized_pl: float = 0.0
    open_trade_count: int = 0
    pending_order_count: int = 0
    last_updated: str = ""
    

class BrokerAccountList(BaseModel):
    """List of broker accounts."""
    accounts: List[BrokerAccountInfo]
    count: int


class SaveCredentialsRequest(BaseModel):
    """Request to save broker credentials."""
    credentials: Dict[str, Any]
    
    @field_validator("credentials")
    @classmethod
    def validate_credentials(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that the credentials contain required fields based on broker type."""
        if "broker_type" not in v:
            raise ValueError("broker_type is required")
            
        broker_type = v["broker_type"]
        
        if broker_type == BrokerType.OANDA:
            required_fields = ["access_token", "account_id"]
            missing_fields = [field for field in required_fields if field not in v]
            if missing_fields:
                raise ValueError(f"Missing required fields for OANDA: {', '.join(missing_fields)}")
        
        return v


class SaveCredentialsResponse(BaseModel):
    """Response after saving broker credentials."""
    success: bool
    message: str
    broker_type: Optional[BrokerType] = None
    user_id: Optional[str] = None


class DeleteCredentialsRequest(BaseModel):
    """Request to delete broker credentials."""
    broker_type: BrokerType
    user_id: str


class DeleteCredentialsResponse(BaseModel):
    """Response after deleting broker credentials."""
    success: bool
    message: str 
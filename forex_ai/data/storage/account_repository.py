"""
Account repository for Forex AI Trading System.

This module provides a repository for account data using Supabase.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from forex_ai.data.storage.supabase_repository import SupabaseRepository

logger = logging.getLogger(__name__)


class Account(BaseModel):
    """Account model."""
    
    id: str
    user_id: str
    name: str
    balance: float
    currency: str
    broker: Optional[str] = None
    broker_account_id: Optional[str] = None
    is_demo: bool = True
    created_at: datetime
    updated_at: datetime


class AccountRepository(SupabaseRepository[Account]):
    """Repository for account data."""
    
    def __init__(self):
        """Initialize the repository."""
        super().__init__("accounts", Account)
    
    async def get_user_accounts(self, user_id: str) -> List[Account]:
        """
        Get accounts for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            List of accounts
        """
        return await self.get_by_user_id(user_id)
    
    async def get_account_by_id(self, account_id: str) -> Optional[Account]:
        """
        Get account by ID.
        
        Args:
            account_id: Account ID
            
        Returns:
            Account if found, None otherwise
        """
        return await self.get_by_id(account_id)
    
    async def get_account_balance(self, account_id: str) -> Optional[float]:
        """
        Get account balance.
        
        Args:
            account_id: Account ID
            
        Returns:
            Account balance if found, None otherwise
        """
        account = await self.get_by_id(account_id)
        return account.balance if account else None
    
    async def update_balance(self, account_id: str, balance: float) -> bool:
        """
        Update account balance.
        
        Args:
            account_id: Account ID
            balance: New balance
            
        Returns:
            True if successful, False otherwise
        """
        try:
            await self.update(account_id, {"balance": balance, "updated_at": datetime.now().isoformat()})
            return True
        except Exception as e:
            logger.error(f"Error updating account balance: {str(e)}", exc_info=True)
            return False
    
    async def create_account(self, user_id: str, name: str, balance: float, currency: str, 
                             broker: Optional[str] = None, broker_account_id: Optional[str] = None,
                             is_demo: bool = True) -> Optional[Account]:
        """
        Create a new account.
        
        Args:
            user_id: User ID
            name: Account name
            balance: Initial balance
            currency: Currency code
            broker: Broker name
            broker_account_id: Broker account ID
            is_demo: Whether this is a demo account
            
        Returns:
            Created account if successful, None otherwise
        """
        account_data = {
            "user_id": user_id,
            "name": name,
            "balance": balance,
            "currency": currency,
            "broker": broker,
            "broker_account_id": broker_account_id,
            "is_demo": is_demo,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        return await self.create(account_data)
    
    async def get_account_metrics(self, account_id: str, period: str = "1m") -> Dict[str, Any]:
        """
        Get account metrics.
        
        Args:
            account_id: Account ID
            period: Time period (1d, 1w, 1m, 3m, 6m, 1y, all)
            
        Returns:
            Dictionary of account metrics
        """
        # This would normally query trade history to calculate metrics
        # For now, we'll return mock data
        return {
            "win_rate": 65.0,
            "profit_factor": 2.1,
            "sharpe_ratio": 1.5,
            "drawdown_max": 5.2,
            "drawdown_current": 1.8,
            "total_trades": 25,
            "profitable_trades": 16,
            "losing_trades": 9,
            "average_win": 45.0,
            "average_loss": -22.0,
            "largest_win": 120.0,
            "largest_loss": -60.0
        }
    
    async def get_account_performance(self, account_id: str, period: str = "1m") -> Dict[str, Any]:
        """
        Get account performance.
        
        Args:
            account_id: Account ID
            period: Time period (1d, 1w, 1m, 3m, 6m, 1y, all)
            
        Returns:
            Dictionary of account performance data
        """
        # This would normally query trade history to calculate performance data
        # For now, we'll return mock data
        return {
            "daily_returns": {
                "2025-06-01": 0.5,
                "2025-06-02": -0.2,
                "2025-06-03": 0.8,
                "2025-06-04": 0.3,
                "2025-06-05": -0.1
            },
            "cumulative_returns": {
                "2025-06-01": 0.5,
                "2025-06-02": 0.3,
                "2025-06-03": 1.1,
                "2025-06-04": 1.4,
                "2025-06-05": 1.3
            },
            "monthly_returns": {
                "2025-01": 2.5,
                "2025-02": 1.8,
                "2025-03": -0.7,
                "2025-04": 1.2,
                "2025-05": 2.1,
                "2025-06": 1.3
            },
            "equity_curve": {
                "2025-06-01": 10050.0,
                "2025-06-02": 10030.0,
                "2025-06-03": 10110.0,
                "2025-06-04": 10140.0,
                "2025-06-05": 10130.0
            },
            "drawdown_curve": {
                "2025-06-01": 0.0,
                "2025-06-02": 0.2,
                "2025-06-03": 0.0,
                "2025-06-04": 0.0,
                "2025-06-05": 0.1
            }
        } 
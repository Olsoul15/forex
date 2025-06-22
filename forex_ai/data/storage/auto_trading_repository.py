"""
Auto-trading repository for Forex AI Trading System.

This module provides a repository for auto-trading preferences using Supabase.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field

from forex_ai.data.storage.supabase_repository import SupabaseRepository

logger = logging.getLogger(__name__)


class AutoTradingPreferences(BaseModel):
    """Auto-trading preferences model."""
    
    id: str
    user_id: str
    enabled: bool = False
    risk_per_trade: float = 1.0
    max_daily_trades: int = 5
    max_open_trades: int = 3
    allowed_instruments: List[str] = Field(default_factory=lambda: ["EUR_USD", "GBP_USD", "USD_JPY"])
    trading_hours_start: str = "08:00"
    trading_hours_end: str = "16:00"
    trading_days: List[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])  # Monday to Friday
    min_win_rate: float = 55.0
    min_profit_factor: float = 1.5
    stop_loss_required: bool = True
    take_profit_required: bool = True
    created_at: datetime
    updated_at: datetime


class AutoTradingRepository(SupabaseRepository[AutoTradingPreferences]):
    """Repository for auto-trading preferences."""
    
    def __init__(self):
        """Initialize the repository."""
        super().__init__("auto_trading_preferences", AutoTradingPreferences)
    
    async def get_preferences(self, user_id: str) -> Optional[AutoTradingPreferences]:
        """
        Get auto-trading preferences for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Auto-trading preferences if found, None otherwise
        """
        try:
            def build_query(table_ref):
                return table_ref.select("*").eq("user_id", user_id).limit(1)
            
            results = await self.query(build_query)
            
            if results and len(results) > 0:
                return results[0]
            
            return None
        except Exception as e:
            logger.error(f"Error getting auto-trading preferences: {str(e)}", exc_info=True)
            return None
    
    async def update_preferences(self, user_id: str, preferences: Dict[str, Any]) -> Optional[AutoTradingPreferences]:
        """
        Update auto-trading preferences for a user.
        
        Args:
            user_id: User ID
            preferences: Dictionary of preferences to update
            
        Returns:
            Updated preferences if successful, None otherwise
        """
        try:
            # First, get the existing preferences
            existing = await self.get_preferences(user_id)
            
            if not existing:
                # Create new preferences if they don't exist
                # Ensure all required fields are present
                new_preferences = {
                    "user_id": user_id,
                    "enabled": preferences.get("enabled", False),
                    "risk_per_trade": preferences.get("risk_per_trade", 1.0),
                    "max_daily_trades": preferences.get("max_daily_trades", 5),
                    "max_open_trades": preferences.get("max_open_trades", 3),
                    "allowed_instruments": preferences.get("allowed_instruments", ["EUR_USD", "GBP_USD", "USD_JPY"]),
                    "trading_hours_start": preferences.get("trading_hours_start", "08:00"),
                    "trading_hours_end": preferences.get("trading_hours_end", "16:00"),
                    "trading_days": preferences.get("trading_days", [0, 1, 2, 3, 4]),
                    "min_win_rate": preferences.get("min_win_rate", 55.0),
                    "min_profit_factor": preferences.get("min_profit_factor", 1.5),
                    "stop_loss_required": preferences.get("stop_loss_required", True),
                    "take_profit_required": preferences.get("take_profit_required", True),
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat()
                }
                
                return await self.create(new_preferences)
            
            # Update existing preferences
            # Start with existing values and update with new ones
            if hasattr(existing, "model_dump"):
                existing_dict = existing.model_dump()
            elif hasattr(existing, "dict"):
                existing_dict = existing.dict()
            else:
                existing_dict = dict(existing)
                
            # Remove id and created_at from existing dict to avoid overwriting
            if "id" in existing_dict:
                existing_id = existing_dict["id"]
            else:
                logger.error(f"Error updating preferences: existing preferences has no id")
                return None
                
            # Update with new preferences
            update_data = {**existing_dict, **preferences}
            
            # Ensure updated_at is set
            update_data["updated_at"] = datetime.now().isoformat()
            
            # Don't modify user_id
            update_data["user_id"] = user_id
            
            return await self.update(existing_id, update_data)
        except Exception as e:
            logger.error(f"Error updating auto-trading preferences: {str(e)}", exc_info=True)
            return None
    
    async def get_statistics(self, user_id: str) -> Dict[str, Any]:
        """
        Get auto-trading statistics for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Dictionary of statistics
        """
        # This would normally query the database for statistics
        # For now, we'll return mock data
        return {
            "total_trades": 25,
            "successful_trades": 15,
            "failed_trades": 5,
            "pending_trades": 5,
            "win_rate": 75.0,
            "profit_factor": 2.5,
            "total_profit_loss": 450.0,
            "average_profit_per_trade": 30.0,
            "average_loss_per_trade": -15.0,
            "largest_profit": 120.0,
            "largest_loss": -45.0,
            "active_since": (datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - 
                           timedelta(days=30)).isoformat(),
            "last_trade_time": (datetime.now() - timedelta(hours=4)).isoformat(),
        }

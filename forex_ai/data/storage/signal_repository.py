"""
Signal repository for Forex AI Trading System.

This module provides a repository for signal data using Supabase.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel

from forex_ai.data.storage.supabase_repository import SupabaseRepository

logger = logging.getLogger(__name__)


class Signal(BaseModel):
    """Signal model."""
    
    id: str
    strategy_id: str
    strategy_name: str
    instrument: str
    timeframe: str
    direction: str
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    confidence: float
    signal_time: datetime
    expiration_time: datetime
    status: str
    notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class SignalRepository(SupabaseRepository[Signal]):
    """Repository for signal data."""
    
    def __init__(self):
        """Initialize the repository."""
        super().__init__("signals", Signal)
    
    async def get_signals_history(self, 
                                 user_id: Optional[str] = None,
                                 instrument: Optional[str] = None, 
                                 timeframe: Optional[str] = None,
                                 direction: Optional[str] = None,
                                 strategy_id: Optional[str] = None,
                                 status: Optional[str] = None,
                                 start_date: Optional[datetime] = None,
                                 end_date: Optional[datetime] = None,
                                 limit: int = 100,
                                 offset: int = 0) -> List[Signal]:
        """
        Get signal history with filters.
        
        Args:
            user_id: Filter by user ID
            instrument: Filter by instrument
            timeframe: Filter by timeframe
            direction: Filter by direction
            strategy_id: Filter by strategy ID
            status: Filter by status
            start_date: Filter by start date
            end_date: Filter by end date
            limit: Maximum number of signals to return
            offset: Number of signals to skip
            
        Returns:
            List of signals
        """
        try:
            # Build query
            def build_query(table_ref):
                query = table_ref.select("*")
                
                # Apply filters
                if instrument:
                    query = query.eq("instrument", instrument)
                    
                if timeframe:
                    query = query.eq("timeframe", timeframe)
                    
                if direction:
                    query = query.eq("direction", direction)
                    
                if strategy_id:
                    query = query.eq("strategy_id", strategy_id)
                    
                if status:
                    query = query.eq("status", status)
                    
                if start_date:
                    query = query.gte("signal_time", start_date.isoformat())
                    
                if end_date:
                    query = query.lte("signal_time", end_date.isoformat())
                    
                # Apply pagination
                query = query.order("signal_time", desc=True).limit(limit).offset(offset)
                
                return query
            
            # Execute query
            return await self.query(build_query)
        except Exception as e:
            logger.error(f"Error getting signal history: {str(e)}", exc_info=True)
            return []
    
    async def get_signals_performance(self,
                                     user_id: Optional[str] = None,
                                     instrument: Optional[str] = None,
                                     timeframe: Optional[str] = None,
                                     strategy_id: Optional[str] = None,
                                     period: str = "1m") -> Dict[str, Any]:
        """
        Get signal performance metrics.
        
        Args:
            user_id: Filter by user ID
            instrument: Filter by instrument
            timeframe: Filter by timeframe
            strategy_id: Filter by strategy ID
            period: Time period (1d, 1w, 1m, 3m, 6m, 1y, all)
            
        Returns:
            Dictionary of performance metrics
        """
        # This would normally query signal history to calculate performance metrics
        # For now, we'll return mock data
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
            "losing_signals": 6
        }
    
    async def create_signal(self,
                           strategy_id: str,
                           strategy_name: str,
                           instrument: str,
                           timeframe: str,
                           direction: str,
                           entry_price: float,
                           stop_loss: Optional[float] = None,
                           take_profit: Optional[float] = None,
                           confidence: float = 0.5,
                           expiry_hours: int = 8,
                           notes: Optional[str] = None) -> Optional[Signal]:
        """
        Create a new signal.
        
        Args:
            strategy_id: Strategy ID
            strategy_name: Strategy name
            instrument: Instrument
            timeframe: Timeframe
            direction: Direction (buy/sell)
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            confidence: Confidence level (0-1)
            expiry_hours: Hours until expiration
            notes: Additional notes
            
        Returns:
            Created signal if successful, None otherwise
        """
        now = datetime.now()
        expiration_time = now + timedelta(hours=expiry_hours)
        
        signal_data = {
            "strategy_id": strategy_id,
            "strategy_name": strategy_name,
            "instrument": instrument,
            "timeframe": timeframe,
            "direction": direction,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "confidence": confidence,
            "signal_time": now.isoformat(),
            "expiration_time": expiration_time.isoformat(),
            "status": "active",
            "notes": notes,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat()
        }
        
        return await self.create(signal_data)
    
    async def update_signal_status(self, signal_id: str, status: str, notes: Optional[str] = None) -> bool:
        """
        Update signal status.
        
        Args:
            signal_id: Signal ID
            status: New status
            notes: Additional notes
            
        Returns:
            True if successful, False otherwise
        """
        update_data = {
            "status": status,
            "updated_at": datetime.now().isoformat()
        }
        
        if notes:
            update_data["notes"] = notes
            
        result = await self.update(signal_id, update_data)
        return result is not None

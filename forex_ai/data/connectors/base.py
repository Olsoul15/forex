"""
Base connector for data sources.

This module provides the base class for all data connectors.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import pandas as pd
from forex_ai.custom_types import CurrencyPair, TimeFrame

class BaseConnector(ABC):
    """Base class for data connectors."""
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to the data source.
        
        Returns:
            bool: True if connection is successful, False otherwise.
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """
        Disconnect from the data source.
        
        Returns:
            bool: True if disconnection is successful, False otherwise.
        """
        pass
    
    @abstractmethod
    async def is_connected(self) -> bool:
        """
        Check if connected to the data source.
        
        Returns:
            bool: True if connected, False otherwise.
        """
        pass
    
    @abstractmethod
    async def get_data(self, **kwargs) -> Any:
        """
        Get data from the data source.
        
        Args:
            **kwargs: Keyword arguments for the data request.
            
        Returns:
            Any: Data from the data source.
        """
        pass

class DataConnector(ABC):
    """Abstract base class for all data connectors."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @abstractmethod
    async def connect(self):
        """Establish connection to the data source."""
        pass

    @abstractmethod
    async def disconnect(self):
        """Disconnect from the data source."""
        pass

    @abstractmethod
    async def fetch_historical_data(
        self,
        currency_pair: CurrencyPair,
        timeframe: TimeFrame,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data."""
        pass

    # Add other common methods needed by connectors if known
    # e.g., fetch_realtime_tick, subscribe_to_stream, etc. 
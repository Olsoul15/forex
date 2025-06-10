"""
Market data service for the Forex AI Trading System.

This module provides a unified interface for market data access,
acting as a facade for various data sources and pipelines.
"""

from functools import lru_cache
from typing import Dict, List, Any, Optional

from forex_ai.data.pipelines.market_data import (
    fetch_market_data,
    import_from_csv,
    export_to_csv,
    export_to_json,
    export_to_excel,
    convert_timeframe,
    merge_data_sources,
    normalize_data
)
from forex_ai.utils.logging import get_logger

logger = get_logger(__name__)

class MarketDataService:
    """
    Service for accessing and manipulating market data.
    
    This class provides a unified interface for market data operations,
    delegating to specialized components for implementation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the market data service.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        logger.info("Initialized market data service")
        
    def fetch_data(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Fetch market data for a symbol and timeframe.
        
        Args:
            symbol: Trading symbol (e.g., 'EUR/USD')
            timeframe: Timeframe (e.g., '1h', '4h', '1d')
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            
        Returns:
            Market data
        """
        logger.info(f"Fetching market data for {symbol} ({timeframe}) from {start_date} to {end_date}")
        return fetch_market_data(symbol, timeframe, start_date, end_date)
    
    def import_csv(self, file_path: str) -> Dict[str, Any]:
        """
        Import market data from a CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Imported market data
        """
        logger.info(f"Importing market data from {file_path}")
        return import_from_csv(file_path)
    
    def export_csv(self, data: Dict[str, Any], file_path: str) -> bool:
        """
        Export market data to a CSV file.
        
        Args:
            data: Market data
            file_path: Path to the output CSV file
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Exporting market data to {file_path}")
        return export_to_csv(data, file_path)
    
    def export_json(self, data: Dict[str, Any], file_path: str) -> bool:
        """
        Export market data to a JSON file.
        
        Args:
            data: Market data
            file_path: Path to the output JSON file
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Exporting market data to {file_path}")
        return export_to_json(data, file_path)
    
    def export_excel(self, data: Dict[str, Any], file_path: str) -> bool:
        """
        Export market data to an Excel file.
        
        Args:
            data: Market data
            file_path: Path to the output Excel file
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Exporting market data to {file_path}")
        return export_to_excel(data, file_path)
    
    def convert_timeframe(self, data: Dict[str, Any], target_timeframe: str) -> Dict[str, Any]:
        """
        Convert market data to a different timeframe.
        
        Args:
            data: Market data
            target_timeframe: Target timeframe (e.g., '1h', '4h', '1d')
            
        Returns:
            Converted market data
        """
        logger.info(f"Converting market data to {target_timeframe}")
        return convert_timeframe(data, target_timeframe)
    
    def merge_sources(self, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge data from multiple sources.
        
        Args:
            sources: List of market data sources
            
        Returns:
            Merged market data
        """
        logger.info(f"Merging {len(sources)} market data sources")
        return merge_data_sources(sources)
    
    def normalize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize market data.
        
        Args:
            data: Market data
            
        Returns:
            Normalized market data
        """
        logger.info("Normalizing market data")
        return normalize_data(data)


@lru_cache()
def get_market_data_service(config: Optional[Dict[str, Any]] = None) -> MarketDataService:
    """
    Get the market data service.
    
    This function returns a singleton instance of the market data service,
    creating it if it doesn't exist.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        MarketDataService instance
    """
    logger.debug("Getting market data service")
    return MarketDataService(config) 
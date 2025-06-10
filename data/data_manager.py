"""
Data Manager for Forex AI.

This module provides data retrieval, storage, and caching capabilities for the Forex AI system.
It handles market data requests, efficient caching, and data validation.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
import json
import hashlib
import pandas as pd

# from forex_ai.core.orchestrator import get_orchestrator
from forex_ai.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

# Cache expiration times by timeframe (in seconds)
CACHE_EXPIRATION = {
    "M1": 60,  # 1 minute
    "M5": 300,  # 5 minutes
    "M15": 900,  # 15 minutes
    "M30": 1800,  # 30 minutes
    "H1": 3600,  # 1 hour
    "H4": 14400,  # 4 hours
    "D": 86400,  # 1 day
    "W": 604800,  # 1 week
    "MN": 2592000,  # 1 month (30 days)
}

# Default expiration time if timeframe not in the map
DEFAULT_CACHE_EXPIRATION = 3600  # 1 hour


class DataManager:
    """
    Data management component for the Forex AI system.

    Responsible for:
    1. Retrieving and caching market data
    2. Validating data integrity
    3. Providing data access to other components
    4. Managing data storage and retrieval
    """

    def __init__(self, forex_settings: Optional[Settings] = None):
        """Initialize the data manager."""
        self.forex_settings = forex_settings
        self.data_cache = {}  # In-memory cache
        self.cache_timestamps = {}  # Track when data was cached
        self.data_provider = None  # Will be initialized later
        self.cache_lock = asyncio.Lock()  # Lock for thread-safe cache access

        logger.info("Data Manager initialized")

    def set_data_provider(self, provider: Any):
        """
        Set the data provider for retrieving market data.

        Args:
            provider: The data provider instance
        """
        self.data_provider = provider
        logger.info(f"Data provider set: {provider.__class__.__name__}")

    async def update_market_data(
        self,
        pair: str,
        timeframe: str,
        count: int = 100,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """
        Retrieve the latest market data for a specified pair and timeframe.

        Args:
            pair: Currency pair (e.g., 'EUR_USD')
            timeframe: Timeframe (e.g., 'M5', 'H1', 'D')
            count: Number of candles to retrieve
            force_refresh: Force refresh even if cache is valid

        Returns:
            Dictionary containing OHLCV data
        """
        # Create cache key
        cache_key = self._create_cache_key(pair, timeframe, count)

        # Check if we have cached data and it's still valid
        if not force_refresh and await self._is_cache_valid(cache_key, timeframe):
            logger.debug(f"Using cached data for {pair} {timeframe}")
            return await self._get_from_cache(cache_key)

        # If we reach here, we need to fetch fresh data
        try:
            if self.data_provider is None:
                logger.error("No data provider configured")
                # Return empty or sample data if no provider
                return self._get_empty_data()

            # Fetch data from provider
            logger.debug(f"Fetching fresh data for {pair} {timeframe}")
            data = await self.data_provider.get_candles(pair, timeframe, count)

            # Validate the data
            if not self._validate_data(data):
                logger.warning(f"Invalid data received for {pair} {timeframe}")
                return self._get_empty_data()

            # Cache the valid data
            await self._cache_data(cache_key, timeframe, data)

            return data

        except Exception as e:
            logger.error(f"Error fetching market data for {pair} {timeframe}: {e}")

            # Try to use cached data even if expired
            cached_data = await self._get_from_cache(cache_key)
            if cached_data:
                logger.warning(
                    f"Using expired cache data for {pair} {timeframe} due to fetch error"
                )
                return cached_data

            # If no cached data available, return empty data
            return self._get_empty_data()

    async def get_historical_data(
        self,
        pair: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        count: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Get historical price data for backtesting or analysis.

        Args:
            pair: Currency pair
            timeframe: Timeframe
            start_time: Start datetime (optional)
            end_time: End datetime (optional)
            count: Number of candles (optional, used if start_time and end_time not provided)

        Returns:
            Dictionary containing historical OHLCV data
        """
        try:
            if self.data_provider is None:
                logger.error("No data provider configured")
                return self._get_empty_data()

            # Default to recent data if no time range specified
            if start_time is None and end_time is None and count is None:
                count = 100  # Default count

            # Create a specific cache key for historical data
            cache_key = self._create_historical_cache_key(
                pair, timeframe, start_time, end_time, count
            )

            # Check for cached historical data (shorter expiration for historical)
            cache_expiration = (
                CACHE_EXPIRATION.get(timeframe, DEFAULT_CACHE_EXPIRATION) * 5
            )
            if await self._is_cache_valid(cache_key, timeframe, cache_expiration):
                logger.debug(f"Using cached historical data for {pair} {timeframe}")
                return await self._get_from_cache(cache_key)

            # Fetch historical data from provider
            logger.debug(f"Fetching historical data for {pair} {timeframe}")
            data = await self.data_provider.get_historical_candles(
                pair, timeframe, start_time, end_time, count
            )

            # Validate and cache the data
            if self._validate_data(data):
                await self._cache_data(cache_key, timeframe, data, cache_expiration)
                return data
            else:
                logger.warning(
                    f"Invalid historical data received for {pair} {timeframe}"
                )
                return self._get_empty_data()

        except Exception as e:
            logger.error(f"Error fetching historical data for {pair} {timeframe}: {e}")
            return self._get_empty_data()

    async def _get_from_cache(self, cache_key: str) -> Dict[str, Any]:
        """
        Get data from cache by key.

        Args:
            cache_key: Cache key

        Returns:
            Cached data or None if not found
        """
        async with self.cache_lock:
            return self.data_cache.get(cache_key, self._get_empty_data())

    async def _is_cache_valid(
        self, cache_key: str, timeframe: str, expiration: Optional[int] = None
    ) -> bool:
        """
        Check if cached data is still valid.

        Args:
            cache_key: Cache key
            timeframe: Timeframe to determine cache expiration
            expiration: Optional custom expiration time in seconds

        Returns:
            True if cache is valid, False otherwise
        """
        async with self.cache_lock:
            # Check if data exists in cache
            if cache_key not in self.data_cache:
                return False

            # Check if timestamp exists
            if cache_key not in self.cache_timestamps:
                return False

            # Get cache timestamp
            timestamp = self.cache_timestamps[cache_key]

            # Determine expiration time
            if expiration is None:
                expiration = CACHE_EXPIRATION.get(timeframe, DEFAULT_CACHE_EXPIRATION)

            # Check if cache has expired
            now = datetime.now().timestamp()
            return (now - timestamp) < expiration

    async def _cache_data(
        self,
        cache_key: str,
        timeframe: str,
        data: Dict[str, Any],
        expiration: Optional[int] = None,
    ):
        """
        Cache data with timestamp.

        Args:
            cache_key: Cache key
            timeframe: Timeframe (used for default expiration)
            data: Data to cache
            expiration: Optional custom expiration time in seconds
        """
        if not data:
            return
        
        async with self.cache_lock:
            self.data_cache[cache_key] = data
            self.cache_timestamps[cache_key] = datetime.now().timestamp()

    def _create_cache_key(self, pair: str, timeframe: str, count: int) -> str:
        """
        Create a unique cache key for market data requests.

        Args:
            pair: Currency pair
            timeframe: Timeframe
            count: Number of candles

        Returns:
            Unique cache key string
        """
        key_str = f"market:{pair}:{timeframe}:{count}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _create_historical_cache_key(
        self,
        pair: str,
        timeframe: str,
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        count: Optional[int],
    ) -> str:
        """
        Create a unique cache key for historical data requests.

        Args:
            pair: Currency pair
            timeframe: Timeframe
            start_time: Start time
            end_time: End time
            count: Count

        Returns:
            Unique cache key string
        """
        start_str = start_time.isoformat() if start_time else "None"
        end_str = end_time.isoformat() if end_time else "None"
        count_str = str(count) if count is not None else "None"
        
        key_str = f"hist:{pair}:{timeframe}:{start_str}:{end_str}:{count_str}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _validate_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate the structure and content of market data dictionary 
        containing a DataFrame under the 'candles' key.

        Args:
            data: Market data dictionary, expected: {'candles': pd.DataFrame}.

        Returns:
            True if data is valid, False otherwise.
        """
        if not isinstance(data, dict):
            logger.warning("Data validation failed: Input is not a dictionary.")
            return False
            
        if 'candles' not in data:
            logger.warning("Data validation failed: Dictionary missing 'candles' key.")
            return False

        df = data['candles']
        if not isinstance(df, pd.DataFrame):
            logger.warning("Data validation failed: Value under 'candles' key is not a DataFrame.")
            return False

        if df.empty:
            # Allow empty dataframes through validation, let caller handle empty data
            logger.debug("Data validation: DataFrame is empty, passing validation.")
            return True 

        # Check for required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Data validation failed: DataFrame missing required columns: {missing_cols}")
            return False

        # Check if index is DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
             logger.warning("Data validation failed: DataFrame index is not a DatetimeIndex.")
             return False

        # Check data types (optional, but good practice)
        for col in required_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                 logger.warning(f"Data validation failed: Column '{col}' is not numeric.")
                 # Optionally return False here, or just log
                 # return False 
        
        # Add more specific validation checks if needed (e.g., price logic H >= L etc.)
        
        logger.debug("Data validation successful for DataFrame structure.")
        return True

    def _get_empty_data(self) -> Dict[str, Any]:
        """Return an empty data dictionary structure.
        
        Returns:
            Empty data dictionary.
        """
        return {
            "timestamp": [],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": [],
        }

    async def get_cached_pairs_and_timeframes(self) -> List[Tuple[str, str]]:
        """
        Get a list of pairs and timeframes currently in the cache.

        Returns:
            List of (pair, timeframe) tuples
        """
        pairs_tf = set()
        async with self.cache_lock:
            for key in self.data_cache.keys():
                try:
                    # Attempt to parse key structure
                    parts = key.split(":")
                    if parts[0] in ["market", "hist"] and len(parts) >= 3:
                        pairs_tf.add((parts[1], parts[2]))
                except Exception:
                    continue # Ignore keys that don't match expected format
        return list(pairs_tf)

    async def clear_cache(self):
        """Clear the entire data cache."""
        async with self.cache_lock:
            self.data_cache.clear()
            self.cache_timestamps.clear()
        logger.info("Data cache cleared")

# Commenting out this method as the import path is incorrect and causing issues
#    def register_with_orchestrator(self):
#        """Register this component with the orchestrator."""
#        try:
#            # Get the orchestrator instance (import is now at top level)
#            orchestrator = get_orchestrator()
#            # Register this DataManager instance
#            orchestrator.register_data_manager(self)
#            logger.info("DataManager registered with orchestrator.") # Optional: Log success
#        except ImportError:
#            # Handle case where orchestrator module might not be available
#            logger.warning("Orchestrator module not found, cannot register DataManager.")
#        except Exception as e:
#            # Catch any other exceptions during registration
#            logger.error(f"Failed to register DataManager with orchestrator: {e}", exc_info=True)

# Singleton instance
_data_manager = None

def get_data_manager(forex_settings: Optional[Settings] = None) -> DataManager:
    """Get the global DataManager instance."""
    global _data_manager
    if _data_manager is None:
        # Use get_settings() if None is provided
        settings_to_use = forex_settings if forex_settings is not None else get_settings()
        _data_manager = DataManager(settings_to_use)
    elif forex_settings is not None and _data_manager.forex_settings is not forex_settings:
        # Log warning if trying to re-init with different settings?
        logger.warning("Attempting to reinitialize DataManager with different settings. Using existing instance.")
        
    return _data_manager
# Ensure nothing follows this line except a single newline character 
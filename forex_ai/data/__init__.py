"""
Data module for the Forex AI Trading System.

This module provides data management functionality for the Forex AI Trading System,
including market data acquisition, storage, and preprocessing.
"""

from forex_ai.data.data_manager import DataManager
# Import what's available from market_data
# from forex_ai.data.market_data import (
#     get_market_data,
#     save_market_data,
#     get_historical_data,
# )
from forex_ai.data.storage.supabase_client import (
    SupabaseClient,
    get_supabase_db_client,
)
from forex_ai.data.storage.redis_client import (
    RedisClient,
    get_redis_client,
)
from forex_ai.data.connectors.base import BaseConnector
from forex_ai.data.connectors.alpha_vantage import AlphaVantageConnector
from forex_ai.data.connectors.news_api import NewsApiConnector
from forex_ai.data.connectors.oanda_handler import OandaDataHandler
from forex_ai.data.connectors.realtime_data import RealtimeDataConnector

# Export public interfaces
__all__ = [
    'DataManager',
    # 'get_market_data',
    # 'save_market_data',
    # 'get_historical_data',
    'SupabaseClient',
    'get_supabase_db_client',
    'RedisClient',
    'get_redis_client',
    'BaseConnector',
    'AlphaVantageConnector',
    'NewsApiConnector',
    'OandaDataHandler',
    'RealtimeDataConnector',
] 
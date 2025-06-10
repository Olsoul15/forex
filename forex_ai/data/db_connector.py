"""
Database connector for the Forex AI Trading System.

This module provides a unified interface for database access,
delegating to the Supabase client for actual database operations.
"""

from functools import lru_cache
from forex_ai.data.storage.supabase_client import get_supabase_db_client, SupabaseClient
from forex_ai.utils.logging import get_logger

logger = get_logger(__name__)

@lru_cache()
def get_db_client() -> SupabaseClient:
    """
    Get the database client.
    
    This is a convenience function that delegates to the Supabase client.
    It's cached so that repeated calls return the same instance.
    
    Returns:
        SupabaseClient: The Supabase client
    """
    logger.debug("Getting database client")
    return get_supabase_db_client() 
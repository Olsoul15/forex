"""
Base Supabase client initialization for Forex AI Trading System.
"""

import logging
from functools import lru_cache
from supabase import create_client, Client

from forex_ai.config.settings import get_settings

logger = logging.getLogger(__name__)

@lru_cache()
def get_base_supabase_client() -> Client:
    """
    Get a base Supabase client instance.

    Returns:
        A Supabase client instance

    Raises:
        ValueError: If Supabase credentials are invalid or connection fails
    """
    try:
        settings = get_settings()
        client = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_KEY
        )
        return client
    except Exception as e:
        error_msg = f"Failed to get Supabase client: {str(e)}"
        logger.critical(error_msg)
        raise ValueError(error_msg) 
"""
Storage clients for the Forex AI Trading System.

This package contains clients for various storage systems,
including PostgreSQL for persistent storage, Supabase for cloud storage,
and Redis for caching.
"""

from forex_ai.data.storage.postgres_client import PostgresClient, get_postgres_client
from forex_ai.data.storage.redis_client import RedisClient, get_redis_client
from forex_ai.data.storage.supabase_client import SupabaseClient, get_supabase_db_client

__all__ = [
    'PostgresClient',
    'get_postgres_client',
    'RedisClient',
    'get_redis_client',
    'SupabaseClient',
    'get_supabase_db_client',
] 
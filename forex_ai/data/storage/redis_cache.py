"""
Redis cache module for Forex AI Trading System.
"""

from forex_ai.data.storage.redis_client import get_redis_client

# Create a global Redis client instance for caching
redis_cache = get_redis_client() 
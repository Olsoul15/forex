"""
Redis client for Forex AI Trading System.
"""

import logging
import json
from typing import Dict, Any, Optional, List, Union
from functools import lru_cache

import redis

from forex_ai.config.settings import get_settings

logger = logging.getLogger(__name__)

class RedisClient:
    """Redis client for caching and data storage."""
    
    def __init__(self):
        """Initialize Redis client."""
        try:
            settings = get_settings()
            self.client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                password=settings.REDIS_PASSWORD,
                decode_responses=True
            )
            logger.info("Redis client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Redis client: {str(e)}")
            raise
    
    def set(self, key: str, value: Any, expiry: Optional[int] = None) -> bool:
        """
        Set a key-value pair in Redis.
        
        Args:
            key: Key to set.
            value: Value to set.
            expiry: Optional expiry time in seconds.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            result = self.client.set(key, value, ex=expiry)
            return bool(result)
        except Exception as e:
            logger.error(f"Failed to set key {key}: {str(e)}")
            return False
    
    def get(self, key: str, as_json: bool = False) -> Optional[Any]:
        """
        Get a value from Redis.
        
        Args:
            key: Key to get.
            as_json: Whether to parse the value as JSON.
            
        Returns:
            Value if found, None otherwise.
        """
        try:
            value = self.client.get(key)
            if value and as_json:
                return json.loads(value)
            return value
        except Exception as e:
            logger.error(f"Failed to get key {key}: {str(e)}")
            return None
    
    def delete(self, key: str) -> bool:
        """
        Delete a key from Redis.
        
        Args:
            key: Key to delete.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            result = self.client.delete(key)
            return bool(result)
        except Exception as e:
            logger.error(f"Failed to delete key {key}: {str(e)}")
            return False
    
    def exists(self, key: str) -> bool:
        """
        Check if a key exists in Redis.
        
        Args:
            key: Key to check.
            
        Returns:
            True if exists, False otherwise.
        """
        try:
            return bool(self.client.exists(key))
        except Exception as e:
            logger.error(f"Failed to check key {key}: {str(e)}")
            return False
    
    def expire(self, key: str, seconds: int) -> bool:
        """
        Set expiry time for a key.
        
        Args:
            key: Key to set expiry for.
            seconds: Expiry time in seconds.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            return bool(self.client.expire(key, seconds))
        except Exception as e:
            logger.error(f"Failed to set expiry for key {key}: {str(e)}")
            return False
    
    def ttl(self, key: str) -> int:
        """
        Get time to live for a key.
        
        Args:
            key: Key to get TTL for.
            
        Returns:
            TTL in seconds, -2 if key doesn't exist, -1 if no expiry.
        """
        try:
            return self.client.ttl(key)
        except Exception as e:
            logger.error(f"Failed to get TTL for key {key}: {str(e)}")
            return -2
    
    def ping(self) -> bool:
        """
        Check if Redis is responsive.
        
        Returns:
            True if responsive, False otherwise.
        """
        try:
            return self.client.ping()
        except Exception as e:
            logger.error(f"Failed to ping Redis: {str(e)}")
            return False
    
    def close(self):
        """Close the Redis connection."""
        try:
            self.client.close()
            logger.info("Redis connection closed")
        except Exception as e:
            logger.error(f"Failed to close Redis connection: {str(e)}")


@lru_cache()
def get_redis_client() -> RedisClient:
    """
    Get a Redis client instance.
    
    Returns:
        A Redis client instance.
    """
    return RedisClient() 
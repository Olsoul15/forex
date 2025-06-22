"""
Redis storage implementation.
"""

import os
import time
import logging
import json
import asyncio
from typing import Any, Dict, List, Optional, Callable, AsyncContextManager
from contextlib import asynccontextmanager
import redis
from redis.client import Redis, Pipeline
from redis.exceptions import ConnectionError, TimeoutError

from forex_ai.data.storage.base import CompleteStorage
from forex_ai.exceptions import CacheError

logger = logging.getLogger(__name__)

class RedisStorage(CompleteStorage):
    """Redis storage implementation with improved connection handling and transactions."""
    
    def __init__(self):
        """Initialize Redis storage with connection retry logic."""
        self._connection_retries = 3
        self._retry_delay = 1.0  # seconds
        self._initialize_connection()
        self._closed = False
    
    def _initialize_connection(self):
        """Initialize Redis connection with retry logic."""
        host = os.environ.get("REDIS_HOST", "localhost")
        port = int(os.environ.get("REDIS_PORT", "6379"))
        db = int(os.environ.get("REDIS_DB", "0"))
        password = os.environ.get("REDIS_PASSWORD")
        
        logger.info(f"Initializing Redis client with host: {host}, port: {port}")
        
        last_error = None
        for attempt in range(self._connection_retries):
            try:
                self._client = Redis(
                    host=host,
                    port=port,
                    db=db,
                    password=password,
                    decode_responses=True,
                    socket_timeout=5,
                    socket_connect_timeout=5,
                    retry_on_timeout=True,
                    health_check_interval=30
                )
                self._pubsub = self._client.pubsub()
                self._client.ping()  # Test connection
                return
            except Exception as e:
                last_error = e
                if attempt < self._connection_retries - 1:
                    logger.warning(f"Redis connection attempt {attempt + 1} failed: {str(e)}")
                    time.sleep(self._retry_delay * (attempt + 1))  # Exponential backoff
                
        logger.error(f"Failed to initialize Redis storage after {self._connection_retries} attempts")
        raise CacheError(f"Failed to initialize Redis storage: {str(last_error)}") from last_error
    
    async def _ensure_connection(self):
        """Ensure Redis connection is alive, reconnect if needed."""
        try:
            if not self._client.ping():
                self._initialize_connection()
        except Exception:
            self._initialize_connection()
    
    @asynccontextmanager
    async def transaction(self) -> AsyncContextManager[Pipeline]:
        """Create a Redis transaction context.
        
        Usage:
            async with storage.transaction() as tr:
                tr.set("key1", "value1")
                tr.set("key2", "value2")
                await tr.execute()
        """
        await self._ensure_connection()
        pipeline = self._client.pipeline(transaction=True)
        try:
            yield pipeline
            await pipeline.execute()
        except Exception as e:
            logger.error(f"Transaction failed: {str(e)}")
            raise
        finally:
            await pipeline.reset()
    
    async def close(self):
        """Close Redis connection."""
        if not self._closed:
            self._client.close()
            self._closed = True
    
    async def ping(self) -> bool:
        """Test Redis connection."""
        try:
            await self._ensure_connection()
            return True
        except Exception:
            return False
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set key-value pair with improved error handling."""
        try:
            await self._ensure_connection()
            value_str = json.dumps(value)
            if ttl is not None:
                return self._client.setex(key, ttl, value_str)
            return self._client.set(key, value_str)
        except Exception as e:
            logger.error(f"Failed to set key {key}: {str(e)}")
            return False
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value by key with improved error handling."""
        try:
            await self._ensure_connection()
            value = self._client.get(key)
            if value is not None:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Failed to get key {key}: {str(e)}")
            return None
    
    async def delete(self, key: str) -> bool:
        """Delete key with improved error handling."""
        try:
            await self._ensure_connection()
            return bool(self._client.delete(key))
        except Exception as e:
            logger.error(f"Failed to delete key {key}: {str(e)}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists with improved error handling."""
        try:
            await self._ensure_connection()
            return bool(self._client.exists(key))
        except Exception as e:
            logger.error(f"Failed to check key {key}: {str(e)}")
            return False
    
    async def increment(self, key: str) -> int:
        """Increment counter with improved error handling."""
        try:
            await self._ensure_connection()
            return self._client.incr(key)
        except Exception as e:
            logger.error(f"Failed to increment key {key}: {str(e)}")
            return 0
    
    async def hset(self, key: str, field: Optional[str] = None, value: Optional[Any] = None, mapping: Optional[Dict[str, Any]] = None) -> int:
        """Set hash field."""
        try:
            if mapping is not None:
                mapping_str = {k: json.dumps(v) for k, v in mapping.items()}
                return self._client.hset(key, mapping=mapping_str)
            if field is not None and value is not None:
                value_str = json.dumps(value)
                return self._client.hset(key, field, value_str)
            raise ValueError("Either field and value or mapping must be provided")
        except Exception as e:
            logger.error(f"Failed to set hash field {key}.{field}: {str(e)}")
            return 0
    
    async def hget(self, key: str, field: str) -> Optional[Any]:
        """Get hash field."""
        try:
            value = self._client.hget(key, field)
            if value is not None:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Failed to get hash field {key}.{field}: {str(e)}")
            return None
    
    async def hdel(self, key: str, *fields: str) -> int:
        """Delete hash fields."""
        try:
            return self._client.hdel(key, *fields)
        except Exception as e:
            logger.error(f"Failed to delete hash fields {key}.{fields}: {str(e)}")
            return 0
    
    async def hgetall(self, key: str) -> Dict[str, Any]:
        """Get all hash fields."""
        try:
            result = self._client.hgetall(key)
            return {k: json.loads(v) for k, v in result.items()}
        except Exception as e:
            logger.error(f"Failed to get all hash fields {key}: {str(e)}")
            return {}
    
    async def lpush(self, key: str, *values: Any) -> int:
        """Push values to list head."""
        try:
            values_str = [json.dumps(v) for v in values]
            return self._client.lpush(key, *values_str)
        except Exception as e:
            logger.error(f"Failed to push to list {key}: {str(e)}")
            return 0
    
    async def rpush(self, key: str, *values: Any) -> int:
        """Push values to list tail."""
        try:
            values_str = [json.dumps(v) for v in values]
            return self._client.rpush(key, *values_str)
        except Exception as e:
            logger.error(f"Failed to push to list {key}: {str(e)}")
            return 0
    
    async def lpop(self, key: str) -> Optional[Any]:
        """Pop value from list head."""
        try:
            value = self._client.lpop(key)
            if value is not None:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Failed to pop from list {key}: {str(e)}")
            return None
    
    async def rpop(self, key: str) -> Optional[Any]:
        """Pop value from list tail."""
        try:
            value = self._client.rpop(key)
            if value is not None:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Failed to pop from list {key}: {str(e)}")
            return None
    
    async def lrange(self, key: str, start: int, stop: int) -> List[Any]:
        """Get list range."""
        try:
            values = self._client.lrange(key, start, stop)
            return [json.loads(v) for v in values]
        except Exception as e:
            logger.error(f"Failed to get list range {key}[{start}:{stop}]: {str(e)}")
            return []
    
    async def clear_pattern(self, pattern: str) -> int:
        """Clear keys matching pattern."""
        try:
            keys = self._client.keys(pattern)
            if keys:
                return self._client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Failed to clear pattern {pattern}: {str(e)}")
            return 0
    
    async def subscribe(self, channel: str, callback: Callable[[str, Any], None]):
        """Subscribe to channel."""
        try:
            async def message_handler(message):
                try:
                    if message["type"] == "message":
                        data = json.loads(message["data"])
                        await callback(message["channel"], data)
                except Exception as e:
                    logger.error(f"Failed to handle message: {str(e)}")
            
            self._pubsub.subscribe(**{channel: message_handler})
            return True
        except Exception as e:
            logger.error(f"Failed to subscribe to channel {channel}: {str(e)}")
            return False
    
    async def unsubscribe(self, channel: str, callback: Callable[[str, Any], None]):
        """Unsubscribe from channel."""
        try:
            self._pubsub.unsubscribe(channel)
            return True
        except Exception as e:
            logger.error(f"Failed to unsubscribe from channel {channel}: {str(e)}")
            return False
    
    async def publish(self, channel: str, message: Any) -> int:
        """Publish message to channel."""
        try:
            message_str = json.dumps(message)
            return self._client.publish(channel, message_str)
        except Exception as e:
            logger.error(f"Failed to publish to channel {channel}: {str(e)}")
            return 0
    
    async def acquire_lock(self, name: str, timeout: Optional[int] = None, blocking_timeout: Optional[float] = None) -> bool:
        """Acquire distributed lock."""
        try:
            return bool(self._client.set(
                f"lock:{name}",
                "1",
                nx=True,
                ex=timeout,
                px=None if blocking_timeout is None else int(blocking_timeout * 1000)
            ))
        except Exception as e:
            logger.error(f"Failed to acquire lock {name}: {str(e)}")
            return False
    
    async def release_lock(self, name: str) -> bool:
        """Release distributed lock."""
        try:
            return bool(self._client.delete(f"lock:{name}"))
        except Exception as e:
            logger.error(f"Failed to release lock {name}: {str(e)}")
            return False 
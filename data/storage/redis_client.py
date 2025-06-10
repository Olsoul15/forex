"""
Redis client for the Forex AI Trading System.

This module provides caching and message broker capabilities using Redis.
NOTE: This is a placeholder with basic structure to be implemented in future.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Union, Any, Callable, Mapping
from datetime import timedelta
from urllib.parse import urlunparse
import os

import redis
from redis.exceptions import RedisError

from forex_ai.config.settings import get_settings
from forex_ai.exceptions import CacheError, ForexAiError

# Define a local MessageBrokerError class until it's added to exceptions.py
class MessageBrokerError(ForexAiError):
    """Base exception for message broker-related errors."""
    pass

logger = logging.getLogger(__name__)

# Global variable to hold the Redis client instance
_redis_client_instance: Optional['RedisClient'] = None
_redis_pool_instance: Optional[redis.BlockingConnectionPool] = None # For synchronous client


class RedisClient:
    """
    Client for Redis operations.
    
    This class provides methods for caching data, pub/sub messaging,
    and distributed locking using Redis.
    
    Note: Uses synchronous redis-py. Eventlet monkey-patching will make it non-blocking.
    """
    
    def __init__(
        self,
        pool: redis.BlockingConnectionPool
    ):
        """
        Initialize the Redis client with a connection pool.
        
        Args:
            pool: A redis.BlockingConnectionPool instance.
        """
        try:
            self._client = redis.Redis(connection_pool=pool, decode_responses=True)
            self._pool = pool # Keep a reference to the pool
            self._pubsub = None # Pub/sub client initialized on demand

            logger.info(
                f"Synchronous Redis client initialized with pool."
            )

        except RedisError as e:
            logger.error(f"Failed to initialize synchronous Redis client with pool: {e}")
            raise CacheError(f"Failed to initialize Redis client: {e}") from e
        except Exception as e: # Catch other potential errors during init
             logger.error(f"Unexpected error initializing Redis client: {e}")
             raise CacheError(f"Unexpected error initializing Redis client: {e}") from e

    @property
    def client(self) -> redis.Redis:
        """Provides access to the underlying redis-py synchronous client."""
        if self._client is None:
            logger.error("Redis client accessed before initialization.")
            raise CacheError("Redis client not initialized.")
        return self._client

    #
    # Cache operations
    #

    def get(self, key: str) -> Optional[str]:
        """
        Get a value from the cache.
        """
        try:
            return self.client.get(key)
        except RedisError as e:
            logger.error(f"Redis GET failed for key '{key}': {e}")
            raise CacheError(f"Failed to get key '{key}'") from e
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None, # ttl in seconds
        nx: bool = False,
        xx: bool = False,
    ) -> bool:
        """
        Set a value in the cache.
        """
        try:
            result = self.client.set(key, value, ex=ttl, nx=nx, xx=xx)
            return result is True
        except RedisError as e:
            logger.error(f"Redis SET failed for key '{key}': {e}")
            raise CacheError(f"Failed to set key '{key}'") from e
    
    def delete(self, key: str) -> bool:
        """
        Delete a value from the cache.
        """
        try:
            deleted_count = self.client.delete(key)
            return deleted_count > 0
        except RedisError as e:
            logger.error(f"Redis DELETE failed for key '{key}': {e}")
            raise CacheError(f"Failed to delete key '{key}'") from e
    
    def exists(self, key: str) -> bool:
        """
        Check if a key exists in the cache.
        """
        try:
            num_keys = self.client.exists(key)
            return num_keys > 0
        except RedisError as e:
            logger.error(f"Redis EXISTS failed for key '{key}': {e}")
            raise CacheError(f"Failed to check existence for key '{key}'") from e
    
    def expire(self, key: str, ttl: int) -> bool:
        """
        Set a key's time to live.
        """
        try:
            return self.client.expire(key, ttl)
        except RedisError as e:
            logger.error(f"Redis EXPIRE failed for key '{key}': {e}")
            raise CacheError(f"Failed to set TTL for key '{key}'") from e
    
    def keys(self, pattern: str) -> List[str]:
        """
        Get keys matching a pattern.
        Warning: Use with caution in production, can be slow.
        """
        try:
            # .keys() in redis-py with decode_responses=True returns list of strings
            return self.client.keys(pattern)
        except RedisError as e:
            logger.error(f"Redis KEYS failed for pattern '{pattern}': {e}")
            raise CacheError(f"Failed to get keys for pattern '{pattern}'") from e

    def hget(self, name: str, key: str) -> Optional[str]:
        try:
            return self.client.hget(name, key)
        except RedisError as e:
            logger.error(f"Redis HGET failed for name '{name}' key '{key}': {e}")
            raise CacheError(f"Failed to HGET for name '{name}' key '{key}'") from e

    def hset(self, name: str, key: str = None, value: Any = None, mapping: Mapping[str, Any] = None) -> int:
        try:
            if mapping is not None:
                return self.client.hset(name, mapping=mapping)
            elif key is not None:
                return self.client.hset(name, key, value)
            else:
                raise ValueError("hset requires either key/value or mapping")
        except RedisError as e:
            logger.error(f"Redis HSET failed for name '{name}': {e}")
            raise CacheError(f"Failed to HSET for name '{name}'") from e

    def hmset(self, name: str, mapping: Mapping[str, Any]) -> bool:
        # hmset is deprecated, use hset with mapping
        try:
            return self.client.hset(name, mapping=mapping) is not None # hset returns num fields added
        except RedisError as e:
            logger.error(f"Redis HMSET (using HSET) failed for name '{name}': {e}")
            raise CacheError(f"Failed to HMSET for name '{name}'") from e
            
    def hgetall(self, name: str) -> Dict[str, str]:
        try:
            return self.client.hgetall(name)
        except RedisError as e:
            logger.error(f"Redis HGETALL failed for name '{name}': {e}")
            raise CacheError(f"Failed to HGETALL for name '{name}'") from e

    def hdel(self, name: str, *keys: str) -> int:
        try:
            return self.client.hdel(name, *keys)
        except RedisError as e:
            logger.error(f"Redis HDEL failed for name '{name}': {e}")
            raise CacheError(f"Failed to HDEL for name '{name}'") from e
            
    def lpush(self, name: str, *values: Any) -> int:
        try:
            return self.client.lpush(name, *values)
        except RedisError as e:
            logger.error(f"Redis LPUSH failed for name '{name}': {e}")
            raise CacheError(f"Failed to LPUSH for name '{name}'") from e

    def rpush(self, name: str, *values: Any) -> int:
        try:
            return self.client.rpush(name, *values)
        except RedisError as e:
            logger.error(f"Redis RPUSH failed for name '{name}': {e}")
            raise CacheError(f"Failed to RPUSH for name '{name}'") from e

    def lpop(self, name: str) -> Optional[str]:
        try:
            return self.client.lpop(name)
        except RedisError as e:
            logger.error(f"Redis LPOP failed for name '{name}': {e}")
            raise CacheError(f"Failed to LPOP for name '{name}'") from e

    def rpop(self, name: str) -> Optional[str]:
        try:
            return self.client.rpop(name)
        except RedisError as e:
            logger.error(f"Redis RPOP failed for name '{name}': {e}")
            raise CacheError(f"Failed to RPOP for name '{name}'") from e

    def lrange(self, name: str, start: int, end: int) -> List[str]:
        try:
            return self.client.lrange(name, start, end)
        except RedisError as e:
            logger.error(f"Redis LRANGE failed for name '{name}': {e}")
            raise CacheError(f"Failed to LRANGE for name '{name}'") from e
            
    def publish(self, channel: str, message: Any) -> int:
        """
        Publish a message to a channel.
        The message is serialized to JSON.
        """
        try:
            if not isinstance(message, str):
                message = json.dumps(message)
            return self.client.publish(channel, message)
        except RedisError as e:
            logger.error(f"Redis PUBLISH failed for channel '{channel}': {e}")
            raise MessageBrokerError(f"Failed to publish to channel '{channel}'") from e
        except TypeError as e: # For json.dumps errors
            logger.error(f"Failed to serialize message for channel '{channel}': {e}")
            raise MessageBrokerError(f"Failed to serialize message for channel '{channel}': {e}") from e
            
    # Synchronous pubsub is different. Typically you get a pubsub object,
    # subscribe, and then run a listen() loop in a separate thread/greenlet.
    # For simplicity here, we might not fully replicate the async handler pattern
    # directly, or assume eventlet handles the blocking listen().

    def get_pubsub(self):
        if self._pubsub is None:
            self._pubsub = self.client.pubsub(ignore_subscribe_messages=True)
        return self._pubsub

    def subscribe(self, channel: str) -> None: # Handler logic needs to be external
        ps = self.get_pubsub()
        try:
            ps.subscribe(channel)
            # The connection from the pool should now have socket_timeout=None by default.
            logger.info(f"Subscribed to channel '{channel}' (synchronous). Listening loop needed separately.")
        except RedisError as e:
            logger.error(f"Redis SUBSCRIBE failed for channel '{channel}': {e}")
            raise MessageBrokerError(f"Failed to subscribe to channel '{channel}'") from e

    def unsubscribe(self, channel: str) -> None:
        ps = self.get_pubsub()
        try:
            ps.unsubscribe(channel)
            logger.info(f"Unsubscribed from channel '{channel}' (synchronous).")
        except RedisError as e:
            logger.error(f"Redis UNSUBSCRIBE failed for channel '{channel}': {e}")
            raise MessageBrokerError(f"Failed to unsubscribe from channel '{channel}'") from e
            
    # listen() would be a blocking call, intended to be run in a green thread by eventlet
    def listen_on_channel(self, channel: str, handler: Callable[[Dict[str, Any]], None]):
        # This is a simplified example. Real implementation needs robust error handling & message parsing.
        # Ensure subscribe was called before this.
        ps = self.get_pubsub()
        logger.info(f"Starting to listen on channel '{channel}' (synchronous)...")
        for message in ps.listen(): # This will block until a message or control signal
            if message['type'] == 'message':
                logger.debug(f"Received sync message on '{message['channel']}': {message['data']}")
                try:
                    # Assuming data is JSON string as per publish method
                    data_payload = json.loads(message['data'])
                    handler(data_payload) # Call the provided handler
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode JSON message from '{message['channel']}': {e}. Data: {message['data']}")
                except Exception as e:
                    logger.error(f"Handler error for message from '{message['channel']}': {e}", exc_info=True)
            elif message['type'] == 'subscribe':
                 logger.info(f"Successfully subscribed to channel: {message['channel']} (in listen loop)")
            # Handle other message types like unsubscribe, psubscribe etc. if needed

    def lock(
        self,
        name: str,
        timeout: Optional[int] = None, # Lock expiry time in seconds
        blocking_timeout: Optional[float] = None, # How long to wait to acquire lock
        sleep: float = 0.1, # Poll interval when blocking
    ) -> Optional[redis.lock.Lock]:
        """
        Acquire a distributed lock.
        """
        try:
            # redis-py lock takes timeout in seconds (float or int)
            # blocking_timeout in seconds (float or int)
            # Ensure sleep is float
            return self.client.lock(name, timeout=timeout, blocking_timeout=blocking_timeout, sleep=sleep)
        except RedisError as e: # LockError inherits from RedisError
            logger.error(f"Failed to acquire lock '{name}': {e}")
            # Depending on desired behavior, might not re-raise, or raise specific LockError
            return None # Indicate failure to acquire

    # unlock is typically called on the lock object itself, e.g., my_lock.release()
    # This method is less common for redis-py's lock object.
    # For now, let's assume the user manages the lock object.
    # def unlock(self, lock_obj: redis.lock.Lock) -> bool:
    #     try:
    #         lock_obj.release()
    #         return True
    #     except redis.lock.LockNotOwnedError: # Specific error if not owned
    #         logger.warning("Attempted to release a lock not owned or already released.")
    #         return False
    #     except RedisError as e:
    #         logger.error(f"Failed to release lock: {e}")
    #         return False

    def ping(self) -> bool:
        """
        Ping the Redis server.
        """
        try:
            return self.client.ping()
        except RedisError as e:
            logger.error(f"Redis PING failed: {e}")
            return False

    def close(self) -> None:
        """
        Close the Redis connection (pool will be closed).
        """
        # For synchronous client with pool, closing is typically done by pool.disconnect()
        # or letting Python's garbage collector handle it if pool is not global.
        # If we have a pubsub client, it should be closed.
        if self._pubsub:
            try:
                self._pubsub.close() # For redis-py PubSub objects
                logger.info("Redis PubSub connection closed.")
            except Exception as e:
                logger.error(f"Error closing Redis PubSub connection: {e}", exc_info=True)
            self._pubsub = None

        # For the client using a pool, the pool itself should be managed.
        # If _redis_pool_instance is the global pool, it's closed in close_redis_client().
        # If this RedisClient instance created its own pool (not current design), it would close it here.
        # For now, client.close() might not be what we want if it uses a shared pool.
        # Let close_redis_client handle pool disconnection.
        if self._client:
            try:
                # redis-py's client doesn't have a .close() method itself when using a pool.
                # The pool is what gets disconnected.
                logger.info("Redis client object cleared. Pool disconnection handled by close_redis_client().")
            except Exception as e:
                 logger.error(f"Error during RedisClient pseudo-close: {e}", exc_info=True)
        self._client = None # Clear the client reference

# --- Singleton Management for Synchronous Client ---

def _create_sync_pool() -> redis.BlockingConnectionPool:
    settings = get_settings()
    
    if not settings.REDIS_URL:
        logger.error("CRITICAL: settings.REDIS_URL is not set. Cannot create Redis pool.")
        raise CacheError("settings.REDIS_URL is not configured.")

    # Use redis.from_url() to parse the DSN and get connection parameters.
    try:
        # Create a temporary client instance to easily access parsed connection args from its pool
        temp_client = redis.from_url(str(settings.REDIS_URL))
        # The actual connection kwargs are stored in the pool instance created by from_url
        parsed_kwargs = temp_client.connection_pool.connection_kwargs
        # It's good practice to close the temporary client's connections if it established any,
        # though for just extracting args it might not be strictly necessary if no commands were run.
        temp_client.close() # Close the client itself
        temp_client.connection_pool.disconnect() # Disconnect its pool

    except Exception as e:
        logger.error(f"Failed to parse settings.REDIS_URL ('{settings.REDIS_URL}') using redis.from_url(): {e}")
        raise CacheError(f"Invalid REDIS_URL format or connection issue: {settings.REDIS_URL}") from e

    host = parsed_kwargs.get('host', 'localhost')
    port = parsed_kwargs.get('port', 6379)
    password = parsed_kwargs.get('password') # Will be None if not present
    db = parsed_kwargs.get('db', 0)
    # Check for SSL more reliably from parsed arguments or original scheme
    ssl_detected = str(settings.REDIS_URL).startswith("rediss://") or parsed_kwargs.get('ssl', False)

    logger.info(f"Creating new synchronous Redis connection pool for {host}:{port}, DB: {db}, SSL: {ssl_detected} (derived from settings.REDIS_URL: '{str(settings.REDIS_URL)}')")

    # Base connection kwargs for the new pool we are explicitly creating
    connection_kwargs = {
        'host': host,
        'port': port,
        'db': db,
        'decode_responses': True, # Standardize this
        'socket_timeout': None, 
        'socket_connect_timeout': getattr(settings, 'REDIS_SOCKET_CONNECT_TIMEOUT', 5)
    }
    
    if password:
        connection_kwargs['password'] = password

    if ssl_detected:
        connection_kwargs['ssl'] = True
        # redis-py generally handles SSL connection class setup automatically if 'ssl':True is passed.
        # Explicitly setting SSLConnection might be redundant or only for specific SSL context needs.
        # connection_kwargs['connection_class'] = redis.SSLConnection 
        
        # Carry over specific SSL params if parsed from URL, e.g. ssl_cert_reqs
        if 'ssl_cert_reqs' in parsed_kwargs:
             connection_kwargs['ssl_cert_reqs'] = parsed_kwargs['ssl_cert_reqs']
        # if 'ssl_ca_certs' in parsed_kwargs: # Example for CA certs
        #      connection_kwargs['ssl_ca_certs'] = parsed_kwargs['ssl_ca_certs']

        logger.info(f"SSL detected and enabled. Final connection_kwargs for pool: {connection_kwargs}")
    else:
        logger.info(f"SSL NOT detected. Final connection_kwargs for pool: {connection_kwargs}")
    
    return redis.BlockingConnectionPool(
        max_connections=getattr(settings, 'REDIS_POOL_SIZE', 10),
        **connection_kwargs 
    )

def get_redis_client() -> RedisClient:
    """
    Get a singleton instance of the synchronous RedisClient.
    Initializes the client and its pool on first call.
    """
    global _redis_client_instance
    global _redis_pool_instance

    if _redis_pool_instance is None or not _redis_pool_instance.connection_kwargs.get('host'): # Check if pool is truly uninitialized or disconnected
        logger.info("Synchronous Redis pool not initialized or disconnected. Creating new pool.")
        _redis_pool_instance = _create_sync_pool()
        # If client instance exists but pool was bad, it also needs re-init
        if _redis_client_instance is not None:
            logger.info("Re-initializing RedisClient due to pool re-creation.")
            _redis_client_instance = None 
    
    if _redis_client_instance is None:
        logger.info("Synchronous RedisClient instance not initialized. Creating new instance with pool.")
        _redis_client_instance = RedisClient(pool=_redis_pool_instance)
        try:
            if not _redis_client_instance.ping():
                logger.error("Initial PING to Redis failed for synchronous client. Check connection and settings.")
            else:
                logger.info("Synchronous Redis client initialized and PING successful.")
        except Exception as e:
            logger.error(f"Exception during initial PING for synchronous client: {e}", exc_info=True)
            _redis_client_instance = None
            _redis_pool_instance = None # Invalidate pool too, as it might be the cause
            raise CacheError(f"Failed initial Redis PING: {e}") from e

    # No elif for client being closed, as RedisClient.close() now just clears _client.
    # get_redis_client() will always return the singleton or create it if None.
            
    return _redis_client_instance

def close_redis_client():
    """
    Close the singleton Redis client's connection pool.
    """
    global _redis_client_instance
    global _redis_pool_instance

    if _redis_client_instance is not None:
        logger.info("Closing (clearing) synchronous RedisClient instance...")
        _redis_client_instance.close() 
        _redis_client_instance = None
    else:
        logger.info("Synchronous RedisClient instance was already None.")

    if _redis_pool_instance is not None:
        logger.info("Disconnecting synchronous Redis connection pool...")
        try:
            _redis_pool_instance.disconnect()
            logger.info("Synchronous Redis connection pool disconnected.")
        except Exception as e:
            logger.error(f"Error disconnecting synchronous Redis connection pool: {e}", exc_info=True)
        _redis_pool_instance = None
    else:
        logger.info("Synchronous Redis connection pool was already None.")

# Placeholder for actual message handler types
MessageHandlerType = Callable[[Dict[str, Any]], None]

# Example usage (primarily for testing this module directly)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Running RedisClient synchronous example...")

    try:
        redis_cli = get_redis_client()
        logger.info(f"Ping result: {redis_cli.ping()}")
        logger.info(f"SET mykey myvalue: {redis_cli.set('mykey', 'myvalue', ttl=60)}")
        logger.info(f"GET mykey: {redis_cli.get('mykey')}")
        logger.info(f"EXISTS mykey: {redis_cli.exists('mykey')}")
        logger.info(f"HSET myhash field1 value1: {redis_cli.hset('myhash', key='field1', value='value1')}")
        logger.info(f"HGET myhash field1: {redis_cli.hget('myhash', 'field1')}")
        all_fields = redis_cli.hgetall('myhash')
        logger.info(f"HGETALL myhash: {all_fields}")
    except ForexAiError as e:
        logger.error(f"A ForexAI specific error occurred: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred in the example: {e}", exc_info=True)
    finally:
        logger.info("Closing Redis client in example...")
        close_redis_client()
        logger.info("Redis client closed in example.")
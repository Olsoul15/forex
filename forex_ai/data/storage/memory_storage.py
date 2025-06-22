"""
In-memory implementation of storage interfaces with improved memory management.
"""

import json
import time
import asyncio
import logging
import threading
import fnmatch
import sys
from typing import Any, Dict, List, Optional, Mapping, Callable, Tuple
from collections import defaultdict
from datetime import datetime, timedelta

from forex_ai.data.storage.base import CompleteStorage
from forex_ai.utils.logging import get_logger

logger = get_logger(__name__)

class Lock:
    """Simple lock implementation with timeout support."""
    def __init__(self, name: str, timeout: Optional[int] = None):
        self.name = name
        self.timeout = timeout
        self.acquired_time = None
        self.owner = None
        
    def is_expired(self) -> bool:
        """Check if the lock has expired."""
        if self.timeout is None or self.acquired_time is None:
            return False
        return time.time() > self.acquired_time + self.timeout

class MemoryStorage(CompleteStorage):
    """Thread-safe in-memory implementation of storage interfaces with memory management."""
    
    def __init__(self):
        """Initialize storage with memory management."""
        self._data = {}  # Key-value storage
        self._hash_data = defaultdict(dict)  # Hash storage
        self._list_data = defaultdict(list)  # List storage
        self._expiry = {}  # TTL tracking
        self._subscribers = defaultdict(list)  # Pub/Sub subscribers
        self._locks = {}  # Lock storage
        self._lock = threading.Lock()  # Thread safety
        self._closed = False
        
        # Memory management settings
        self._cleanup_interval = 60  # seconds
        self._max_memory = 1024 * 1024 * 1024  # 1GB
        self._memory_threshold = 0.9  # 90% of max memory
        self._cleanup_running = False
        self._cleanup_task = None
        
        # Don't start cleanup task in __init__
        # It will be started when the application starts
    
    def _serialize(self, value: Any) -> str:
        """Serialize value to string with improved error handling."""
        try:
            if isinstance(value, (str, int, float, bool, type(None))):
                return json.dumps(value)
            return json.dumps(value, default=str)
        except Exception as e:
            logger.error(f"Serialization error: {str(e)}")
            raise ValueError(f"Could not serialize value: {value}")
    
    def _deserialize(self, value: str) -> Any:
        """Deserialize string to value with improved error handling."""
        try:
            return json.loads(value)
        except Exception as e:
            logger.error(f"Deserialization error: {str(e)}")
            return value  # Return raw value if deserialization fails
    
    def _get_memory_usage(self) -> Tuple[int, float]:
        """Get current memory usage in bytes and percentage."""
        total_bytes = sys.getsizeof(self._data)
        total_bytes += sys.getsizeof(self._hash_data)
        total_bytes += sys.getsizeof(self._list_data)
        total_bytes += sys.getsizeof(self._expiry)
        total_bytes += sys.getsizeof(self._subscribers)
        total_bytes += sys.getsizeof(self._locks)
        
        # Add size of stored values
        for value in self._data.values():
            total_bytes += sys.getsizeof(value)
        
        for hash_map in self._hash_data.values():
            total_bytes += sum(sys.getsizeof(v) for v in hash_map.values())
        
        for list_values in self._list_data.values():
            total_bytes += sum(sys.getsizeof(v) for v in list_values)
        
        return total_bytes, total_bytes / self._max_memory
    
    async def start(self):
        """Start the cleanup task."""
        if not self._cleanup_running:
            self._cleanup_running = True
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def stop(self):
        """Stop the cleanup task."""
        if self._cleanup_running:
            self._cleanup_running = False
            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
    
    async def _cleanup_loop(self):
        """Periodic cleanup loop."""
        while self._cleanup_running:
            try:
                await self._cleanup_expired()
                await self._check_memory_usage()
                await asyncio.sleep(self._cleanup_interval)
            except Exception as e:
                logger.error(f"Error in cleanup loop: {str(e)}")
    
    async def _cleanup_expired(self):
        """Clean up expired keys."""
        current_time = time.time()
        with self._lock:
            expired_keys = [
                key for key, expire_time in self._expiry.items()
                if current_time > expire_time
            ]
            for key in expired_keys:
                self._data.pop(key, None)
                self._expiry.pop(key, None)
    
    async def _check_memory_usage(self):
        """Check and manage memory usage."""
        total_bytes, usage_ratio = self._get_memory_usage()
        
        if usage_ratio > self._memory_threshold:
            logger.warning(f"Memory usage high: {usage_ratio:.2%}")
            with self._lock:
                # Remove expired items first
                await self._cleanup_expired()
                
                # If still above threshold, remove oldest items
                if self._get_memory_usage()[1] > self._memory_threshold:
                    sorted_items = sorted(
                        self._expiry.items(),
                        key=lambda x: x[1] if x[1] is not None else float('inf')
                    )
                    
                    # Remove oldest 10% of items
                    items_to_remove = int(len(sorted_items) * 0.1)
                    for key, _ in sorted_items[:items_to_remove]:
                        self._data.pop(key, None)
                        self._expiry.pop(key, None)
    
    async def ping(self) -> bool:
        """Check if storage is available."""
        return not self._closed
    
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from storage with improved error handling."""
        if self._check_expiry(key):
            return None
        try:
            with self._lock:
                value = self._data.get(key)
                if value is not None:
                    return self._deserialize(value)
                return None
        except Exception as e:
            logger.error(f"Error getting from storage: {str(e)}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in storage with improved error handling."""
        try:
            with self._lock:
                serialized_value = self._serialize(value)
                self._data[key] = serialized_value
                if ttl is not None:
                    self._expiry[key] = time.time() + ttl
                return True
        except Exception as e:
            logger.error(f"Error setting storage: {str(e)}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete a value from storage."""
        try:
            with self._lock:
                if key in self._data:
                    del self._data[key]
                    self._expiry.pop(key, None)
                    return True
                return False
        except Exception as e:
            logger.error(f"Error deleting from storage: {str(e)}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if a key exists in storage."""
        if self._check_expiry(key):
            return False
        return key in self._data
    
    async def increment(self, key: str) -> int:
        """Increment a counter."""
        try:
            with self._lock:
                if key not in self._data:
                    self._data[key] = "0"
                value = int(self._data[key])
                value += 1
                self._data[key] = str(value)
                return value
        except Exception as e:
            logger.error(f"Error incrementing storage: {str(e)}")
            return 0
    
    async def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching a pattern."""
        try:
            with self._lock:
                keys = [k for k in self._data.keys() if fnmatch.fnmatch(k, pattern)]
                for key in keys:
                    del self._data[key]
                    self._expiry.pop(key, None)
                return len(keys)
        except Exception as e:
            logger.error(f"Error clearing storage pattern: {str(e)}")
            return 0
    
    # Hash operations
    async def hget(self, name: str, key: str) -> Optional[str]:
        """Get a value from a hash."""
        try:
            return self._hash_data[name].get(key)
        except Exception as e:
            logger.error(f"Error in HGET: {str(e)}")
            return None
    
    async def hset(self, name: str, key: str = None, value: Any = None, mapping: Mapping[str, Any] = None) -> int:
        """Set a value in a hash."""
        try:
            with self._lock:
                if mapping is not None:
                    self._hash_data[name].update(mapping)
                    return len(mapping)
                elif key is not None:
                    self._hash_data[name][key] = value
                    return 1
                else:
                    raise ValueError("hset requires either key/value or mapping")
        except Exception as e:
            logger.error(f"Error in HSET: {str(e)}")
            return 0
    
    async def hgetall(self, name: str) -> Dict[str, str]:
        """Get all values from a hash."""
        try:
            return dict(self._hash_data[name])
        except Exception as e:
            logger.error(f"Error in HGETALL: {str(e)}")
            return {}
    
    async def hdel(self, name: str, *keys: str) -> int:
        """Delete values from a hash."""
        try:
            with self._lock:
                count = 0
                for key in keys:
                    if key in self._hash_data[name]:
                        del self._hash_data[name][key]
                        count += 1
                return count
        except Exception as e:
            logger.error(f"Error in HDEL: {str(e)}")
            return 0
    
    # List operations
    async def lpush(self, name: str, *values: Any) -> int:
        """Push values to the left of a list."""
        try:
            with self._lock:
                for value in values:
                    self._list_data[name].insert(0, value)
                return len(self._list_data[name])
        except Exception as e:
            logger.error(f"Error in LPUSH: {str(e)}")
            return 0
    
    async def rpush(self, name: str, *values: Any) -> int:
        """Push values to the right of a list."""
        try:
            with self._lock:
                self._list_data[name].extend(values)
                return len(self._list_data[name])
        except Exception as e:
            logger.error(f"Error in RPUSH: {str(e)}")
            return 0
    
    async def lpop(self, name: str) -> Optional[str]:
        """Pop a value from the left of a list."""
        try:
            with self._lock:
                if self._list_data[name]:
                    return self._list_data[name].pop(0)
                return None
        except Exception as e:
            logger.error(f"Error in LPOP: {str(e)}")
            return None
    
    async def rpop(self, name: str) -> Optional[str]:
        """Pop a value from the right of a list."""
        try:
            with self._lock:
                if self._list_data[name]:
                    return self._list_data[name].pop()
                return None
        except Exception as e:
            logger.error(f"Error in RPOP: {str(e)}")
            return None
    
    async def lrange(self, name: str, start: int, end: int) -> List[str]:
        """Get a range of values from a list."""
        try:
            return self._list_data[name][start:end]
        except Exception as e:
            logger.error(f"Error in LRANGE: {str(e)}")
            return []
    
    # Pub/Sub operations
    async def publish(self, channel: str, message: Any) -> int:
        """Publish a message to a channel."""
        try:
            if not isinstance(message, str):
                message = json.dumps(message)
            with self._lock:
                subscribers = len(self._subscribers[channel])
                for callback in self._subscribers[channel]:
                    try:
                        asyncio.create_task(callback(channel, message))
                    except Exception as e:
                        logger.error(f"Error in subscriber callback: {str(e)}")
                return subscribers
        except Exception as e:
            logger.error(f"Error in PUBLISH: {str(e)}")
            return 0
    
    async def subscribe(self, channel: str, callback: Callable[[str, Any], None]) -> None:
        """Subscribe to a channel."""
        with self._lock:
            self._subscribers[channel].append(callback)
    
    async def unsubscribe(self, channel: str, callback: Callable[[str, Any], None]) -> None:
        """Unsubscribe from a channel."""
        with self._lock:
            if callback in self._subscribers[channel]:
                self._subscribers[channel].remove(callback)
    
    # Lock operations
    async def acquire_lock(self, name: str, timeout: Optional[int] = None, blocking_timeout: Optional[float] = None) -> bool:
        """Acquire a lock."""
        start_time = time.time()
        while True:
            with self._lock:
                # Check if lock exists and is valid
                existing_lock = self._locks.get(name)
                if existing_lock is None or existing_lock.is_expired():
                    # Create new lock
                    new_lock = Lock(name, timeout)
                    new_lock.acquired_time = time.time()
                    new_lock.owner = id(asyncio.current_task())
                    self._locks[name] = new_lock
                    return True
                
                # If we're not blocking, return False immediately
                if blocking_timeout == 0:
                    return False
                
                # If we've exceeded blocking timeout, return False
                if blocking_timeout is not None and time.time() - start_time > blocking_timeout:
                    return False
            
            # Wait before trying again
            await asyncio.sleep(0.1)
    
    async def release_lock(self, name: str) -> bool:
        """Release a lock."""
        with self._lock:
            existing_lock = self._locks.get(name)
            if existing_lock is None:
                return False
            
            # Only the owner can release the lock
            if existing_lock.owner == id(asyncio.current_task()):
                del self._locks[name]
                return True
            return False
    
    async def close(self) -> None:
        """Close all connections."""
        self._closed = True
        with self._lock:
            self._data.clear()
            self._hash_data.clear()
            self._list_data.clear()
            self._expiry.clear()
            self._subscribers.clear()
            self._locks.clear()

# Create singleton instance
memory_storage = MemoryStorage() 
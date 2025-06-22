"""
Base interfaces for storage implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Mapping, Callable, TypeVar, Generic

T = TypeVar('T')

class BaseStorage(ABC):
    """Base interface for storage operations."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from storage."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in storage."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a value from storage."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if a key exists in storage."""
        pass

class HashStorage(BaseStorage):
    """Interface for hash operations."""
    
    @abstractmethod
    async def hget(self, name: str, key: str) -> Optional[str]:
        """Get a value from a hash."""
        pass
    
    @abstractmethod
    async def hset(self, name: str, key: str = None, value: Any = None, mapping: Mapping[str, Any] = None) -> int:
        """Set a value in a hash."""
        pass
    
    @abstractmethod
    async def hgetall(self, name: str) -> Dict[str, str]:
        """Get all values from a hash."""
        pass
    
    @abstractmethod
    async def hdel(self, name: str, *keys: str) -> int:
        """Delete values from a hash."""
        pass

class ListStorage(BaseStorage):
    """Interface for list operations."""
    
    @abstractmethod
    async def lpush(self, name: str, *values: Any) -> int:
        """Push values to the left of a list."""
        pass
    
    @abstractmethod
    async def rpush(self, name: str, *values: Any) -> int:
        """Push values to the right of a list."""
        pass
    
    @abstractmethod
    async def lpop(self, name: str) -> Optional[str]:
        """Pop a value from the left of a list."""
        pass
    
    @abstractmethod
    async def rpop(self, name: str) -> Optional[str]:
        """Pop a value from the right of a list."""
        pass
    
    @abstractmethod
    async def lrange(self, name: str, start: int, end: int) -> List[str]:
        """Get a range of values from a list."""
        pass

class PubSubStorage(ABC):
    """Interface for publish/subscribe operations."""
    
    @abstractmethod
    async def publish(self, channel: str, message: Any) -> int:
        """Publish a message to a channel."""
        pass
    
    @abstractmethod
    async def subscribe(self, channel: str, callback: Callable[[str, Any], None]) -> None:
        """Subscribe to a channel."""
        pass
    
    @abstractmethod
    async def unsubscribe(self, channel: str, callback: Callable[[str, Any], None]) -> None:
        """Unsubscribe from a channel."""
        pass

class LockStorage(ABC):
    """Interface for distributed locking."""
    
    @abstractmethod
    async def acquire_lock(self, name: str, timeout: Optional[int] = None, blocking_timeout: Optional[float] = None) -> bool:
        """Acquire a lock."""
        pass
    
    @abstractmethod
    async def release_lock(self, name: str) -> bool:
        """Release a lock."""
        pass

class CompleteStorage(HashStorage, ListStorage, PubSubStorage, LockStorage):
    """Complete storage interface combining all functionality."""
    
    @abstractmethod
    async def ping(self) -> bool:
        """Check if storage is available."""
        pass
    
    @abstractmethod
    async def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching a pattern."""
        pass
    
    @abstractmethod
    async def increment(self, key: str) -> int:
        """Increment a counter."""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close all connections."""
        pass 
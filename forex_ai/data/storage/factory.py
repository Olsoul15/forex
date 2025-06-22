"""
Storage factory module with improved instance management.
"""

import logging
import threading
from typing import Optional, Dict
from forex_ai.data.storage.base import CompleteStorage
from forex_ai.data.storage.memory_storage import MemoryStorage
from forex_ai.data.storage.redis_storage import RedisStorage
from forex_ai.exceptions import CacheError

logger = logging.getLogger(__name__)

class StorageFactory:
    """Thread-safe storage factory with instance management."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Ensure singleton pattern."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """Initialize factory if not already initialized."""
        if self._initialized:
            return
            
        with self._lock:
            if not self._initialized:
                self._storage_instances: Dict[str, CompleteStorage] = {}
                self._initialized = True
    
    async def get_storage(self, name: str = "default", force_memory: bool = False) -> CompleteStorage:
        """
        Get storage instance with improved error handling.
        
        Args:
            name: Storage instance name for multiple instances
            force_memory: If True, always use memory storage
            
        Returns:
            Storage instance
        """
        with self._lock:
            # Check if instance already exists
            if name in self._storage_instances:
                instance = self._storage_instances[name]
                # Verify instance is still healthy
                try:
                    if await instance.ping():
                        return instance
                except Exception:
                    logger.warning(f"Storage instance {name} is unhealthy, creating new instance")
                    await self._cleanup_instance(name)
            
            # Create new instance
            try:
                if force_memory:
                    instance = MemoryStorage()
                else:
                    try:
                        instance = RedisStorage()
                        if not await instance.ping():
                            raise CacheError("Redis ping failed")
                    except Exception as e:
                        logger.warning(f"Redis not available: {str(e)}, falling back to memory storage")
                        instance = MemoryStorage()
                
                self._storage_instances[name] = instance
                return instance
            except Exception as e:
                logger.error(f"Failed to create storage instance: {str(e)}")
                raise CacheError(f"Failed to create storage instance: {str(e)}") from e
    
    async def _cleanup_instance(self, name: str):
        """Clean up a storage instance."""
        if name in self._storage_instances:
            try:
                await self._storage_instances[name].close()
            except Exception as e:
                logger.error(f"Error closing storage instance {name}: {str(e)}")
            finally:
                self._storage_instances.pop(name, None)
    
    async def cleanup_all(self):
        """Clean up all storage instances."""
        with self._lock:
            for name in list(self._storage_instances.keys()):
                await self._cleanup_instance(name)

# Global factory instance
_factory = StorageFactory()

async def get_storage(force_memory: bool = False) -> CompleteStorage:
    """
    Get default storage instance.
    
    Args:
        force_memory: If True, always use memory storage
        
    Returns:
        Storage instance
    """
    return await _factory.get_storage(force_memory=force_memory) 
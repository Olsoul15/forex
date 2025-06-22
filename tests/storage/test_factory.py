"""
Tests for storage factory.
"""

import os
import pytest
import asyncio
from unittest.mock import patch, MagicMock

from forex_ai.data.storage.factory import StorageFactory, get_storage
from forex_ai.data.storage.memory_storage import MemoryStorage
from forex_ai.data.storage.redis_storage import RedisStorage
from forex_ai.exceptions import CacheError

@pytest.fixture(autouse=True)
def setup_redis_env():
    """Set up Redis environment variables."""
    os.environ["REDIS_HOST"] = "localhost"
    os.environ["REDIS_PORT"] = "6379"
    os.environ["REDIS_DB"] = "1"
    yield
    # Clean up is handled by pytest's fixture management

@pytest.fixture
async def factory():
    """Get a fresh factory instance."""
    factory = StorageFactory()
    yield factory
    await factory.cleanup_all()

async def test_factory_singleton(factory):
    """Test factory singleton pattern."""
    factory2 = StorageFactory()
    assert factory is factory2
    assert factory._initialized
    assert factory2._initialized

async def test_get_storage_redis_available(factory):
    """Test getting Redis storage when available."""
    try:
        storage = await factory.get_storage()
        assert isinstance(storage, RedisStorage)
        assert await storage.ping()
        
        # Get same instance again
        storage2 = await factory.get_storage()
        assert storage is storage2
    except Exception:
        pytest.skip("Redis not available")

async def test_get_storage_redis_unavailable():
    """Test falling back to memory storage when Redis is unavailable."""
    # Use invalid Redis port
    os.environ["REDIS_PORT"] = "65535"
    
    storage = await get_storage()
    assert isinstance(storage, MemoryStorage)
    assert await storage.ping()
    
    # Restore valid port
    os.environ["REDIS_PORT"] = "6379"

async def test_get_storage_force_memory(factory):
    """Test forcing memory storage."""
    storage = await factory.get_storage(force_memory=True)
    assert isinstance(storage, MemoryStorage)
    assert await storage.ping()

async def test_multiple_named_instances(factory):
    """Test managing multiple named storage instances."""
    # Create two different instances
    storage1 = await factory.get_storage("instance1")
    storage2 = await factory.get_storage("instance2")
    assert storage1 is not storage2
    
    # Get same instances again
    storage1_again = await factory.get_storage("instance1")
    storage2_again = await factory.get_storage("instance2")
    assert storage1 is storage1_again
    assert storage2 is storage2_again

async def test_instance_cleanup(factory):
    """Test cleaning up storage instances."""
    storage = await factory.get_storage("test_instance")
    await storage.set("test_key", "test_value")
    
    # Clean up instance
    await factory._cleanup_instance("test_instance")
    
    # Instance should be removed
    assert "test_instance" not in factory._storage_instances
    
    # New instance should be created
    new_storage = await factory.get_storage("test_instance")
    assert new_storage is not storage
    assert await new_storage.get("test_key") is None

async def test_cleanup_all(factory):
    """Test cleaning up all storage instances."""
    # Create multiple instances
    instances = []
    for i in range(3):
        storage = await factory.get_storage(f"instance_{i}")
        await storage.set("test_key", f"value_{i}")
        instances.append(storage)
    
    # Clean up all instances
    await factory.cleanup_all()
    
    # All instances should be removed
    assert len(factory._storage_instances) == 0
    
    # New instances should be created
    for i in range(3):
        new_storage = await factory.get_storage(f"instance_{i}")
        assert new_storage is not instances[i]
        assert await new_storage.get("test_key") is None

async def test_instance_health_check(factory):
    """Test instance health checking."""
    storage = await factory.get_storage("test_instance")
    assert isinstance(storage, (RedisStorage, MemoryStorage))
    
    # Simulate unhealthy instance
    with patch.object(storage, 'ping', return_value=False):
        # Should create new instance
        new_storage = await factory.get_storage("test_instance")
        assert new_storage is not storage

async def test_concurrent_factory_access(factory):
    """Test concurrent access to factory."""
    async def worker(worker_id: int):
        for i in range(10):
            storage = await factory.get_storage(f"instance_{worker_id}")
            await storage.set(f"key_{i}", f"value_{i}")
            assert await storage.get(f"key_{i}") == f"value_{i}"
            await asyncio.sleep(0.001)  # Simulate work
    
    # Run concurrent workers
    workers = [worker(i) for i in range(5)]
    await asyncio.gather(*workers)
    
    # Verify instance count
    assert len(factory._storage_instances) == 5

async def test_error_handling(factory):
    """Test factory error handling."""
    # Test invalid Redis connection
    with patch('redis.Redis.ping', side_effect=Exception("Redis error")):
        storage = await factory.get_storage()
        assert isinstance(storage, MemoryStorage)
    
    # Test storage creation error
    with patch('forex_ai.data.storage.memory_storage.MemoryStorage', side_effect=Exception("Creation error")):
        with pytest.raises(CacheError):
            await factory.get_storage(force_memory=True)

@pytest.mark.parametrize("redis_available", [True, False])
async def test_storage_caching(redis_available):
    """Test that storage instances are cached."""
    if redis_available:
        # Mock Redis ping to succeed
        with patch('redis.Redis.ping', return_value=True):
            storage1 = await get_storage()
            storage2 = await get_storage()
            assert storage1 is storage2
    else:
        # Mock Redis ping to fail
        with patch('redis.Redis.ping', side_effect=Exception("Redis unavailable")):
            storage1 = await get_storage()
            storage2 = await get_storage()
            assert storage1 is storage2
    
    await storage1.close()

async def test_storage_operations():
    """Test that both storage implementations work through factory."""
    # Test Redis storage
    try:
        redis_storage = await get_storage()
        assert isinstance(redis_storage, RedisStorage)
        await test_storage(redis_storage)
        await redis_storage.close()
    except Exception:
        pytest.skip("Redis not available")
    
    # Test memory storage
    memory_storage = await get_storage(force_memory=True)
    assert isinstance(memory_storage, MemoryStorage)
    await test_storage(memory_storage)
    await memory_storage.close()

async def test_storage(storage):
    """Helper to test storage operations."""
    # Test basic operations
    assert await storage.set("test_key", "test_value")
    assert await storage.get("test_key") == "test_value"
    assert await storage.delete("test_key")
    
    # Test hash operations
    assert await storage.hset("test_hash", "field1", "value1")
    assert await storage.hget("test_hash", "field1") == "value1"
    
    # Test list operations
    assert await storage.lpush("test_list", "value1") == 1
    assert await storage.lpop("test_list") == "value1" 
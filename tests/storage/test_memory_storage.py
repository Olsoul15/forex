"""
Tests for memory storage implementation.
"""

import pytest
import asyncio
import time
from forex_ai.data.storage.base import CompleteStorage
from forex_ai.data.storage.memory_storage import MemoryStorage
from tests.storage.test_storage_base import BaseStorageTests

class TestMemoryStorage(BaseStorageTests):
    """Test cases for memory storage implementation."""
    
    async def get_storage(self) -> CompleteStorage:
        """Get memory storage instance."""
        return MemoryStorage()
    
    async def test_memory_specific_behavior(self, storage: CompleteStorage):
        """Test memory-specific behavior."""
        # Test that different storage instances don't share data
        other_storage = MemoryStorage()
        
        await storage.set("test_key", "test_value")
        assert await storage.get("test_key") == "test_value"
        assert await other_storage.get("test_key") is None
        
        await other_storage.close()
    
    async def test_memory_cleanup(self, storage: CompleteStorage):
        """Test memory cleanup on close."""
        # Set some data
        await storage.set("key1", "value1")
        await storage.hset("hash1", "field1", "value1")
        await storage.lpush("list1", "value1")
        
        # Verify data exists
        assert await storage.exists("key1")
        assert await storage.hget("hash1", "field1") == "value1"
        assert await storage.lpop("list1") == "value1"
        
        # Close storage
        await storage.close()
        
        # Verify data is cleared
        assert not await storage.exists("key1")
        assert await storage.hget("hash1", "field1") is None
        assert await storage.lpop("list1") is None
    
    async def test_memory_management(self, storage: CompleteStorage):
        """Test memory management functionality."""
        if not isinstance(storage, MemoryStorage):
            pytest.skip("Test only applicable to memory storage")
        
        # Fill storage with large data
        large_value = "x" * 1000000  # 1MB string
        for i in range(100):  # Total ~100MB
            await storage.set(f"large_key_{i}", large_value, ttl=1)
        
        # Wait for cleanup
        await asyncio.sleep(2)
        
        # Verify cleanup occurred
        remaining_keys = len([k for k in storage._data.keys()])
        assert remaining_keys < 100
        
        # Check memory usage
        total_bytes, usage_ratio = storage._get_memory_usage()
        assert usage_ratio < storage._memory_threshold
    
    async def test_serialization(self, storage: CompleteStorage):
        """Test value serialization."""
        if not isinstance(storage, MemoryStorage):
            pytest.skip("Test only applicable to memory storage")
        
        # Test various data types
        test_values = [
            42,  # int
            3.14,  # float
            "string",  # str
            True,  # bool
            None,  # None
            [1, 2, 3],  # list
            {"key": "value"},  # dict
            {"set", "value"},  # set
            (1, 2, 3),  # tuple
            b"bytes"  # bytes
        ]
        
        for i, value in enumerate(test_values):
            key = f"test_key_{i}"
            assert await storage.set(key, value)
            retrieved = await storage.get(key)
            
            if isinstance(value, (set, bytes)):
                # These types need special comparison
                assert str(retrieved) == str(value)
            else:
                assert retrieved == value
    
    async def test_concurrent_access(self, storage: CompleteStorage):
        """Test concurrent access to storage."""
        if not isinstance(storage, MemoryStorage):
            pytest.skip("Test only applicable to memory storage")
        
        async def worker(worker_id: int):
            for i in range(100):
                key = f"worker_{worker_id}_key_{i}"
                value = f"value_{i}"
                assert await storage.set(key, value)
                assert await storage.get(key) == value
                await asyncio.sleep(0.001)  # Simulate work
        
        # Run multiple workers concurrently
        workers = [worker(i) for i in range(10)]
        await asyncio.gather(*workers)
        
        # Verify all data is intact
        for worker_id in range(10):
            for i in range(100):
                key = f"worker_{worker_id}_key_{i}"
                assert await storage.get(key) == f"value_{i}"
    
    async def test_ttl_behavior(self, storage: CompleteStorage):
        """Test TTL behavior in detail."""
        if not isinstance(storage, MemoryStorage):
            pytest.skip("Test only applicable to memory storage")
        
        # Test immediate expiration
        await storage.set("expire_0", "value", ttl=0)
        await asyncio.sleep(0.1)
        assert await storage.get("expire_0") is None
        
        # Test near-expiration
        await storage.set("expire_1", "value", ttl=1)
        assert await storage.get("expire_1") == "value"
        await asyncio.sleep(0.5)
        assert await storage.get("expire_1") == "value"
        await asyncio.sleep(0.6)
        assert await storage.get("expire_1") is None
        
        # Test no expiration
        await storage.set("no_expire", "value")
        await asyncio.sleep(1)
        assert await storage.get("no_expire") == "value" 
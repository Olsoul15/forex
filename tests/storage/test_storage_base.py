"""
Base test cases for storage implementations.
"""

import pytest
import asyncio
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod

from forex_ai.data.storage.base import CompleteStorage

class BaseStorageTests(ABC):
    """Base test cases for storage implementations."""
    
    @abstractmethod
    async def get_storage(self) -> CompleteStorage:
        """Get storage instance to test."""
        pass
    
    @pytest.fixture
    async def storage(self) -> CompleteStorage:
        """Storage fixture."""
        storage = await self.get_storage()
        yield storage
        await storage.close()
    
    async def test_basic_operations(self, storage: CompleteStorage):
        """Test basic key-value operations."""
        # Test set and get
        assert await storage.set("test_key", "test_value")
        assert await storage.get("test_key") == "test_value"
        
        # Test exists
        assert await storage.exists("test_key")
        assert not await storage.exists("nonexistent_key")
        
        # Test delete
        assert await storage.delete("test_key")
        assert not await storage.exists("test_key")
        
        # Test TTL
        assert await storage.set("ttl_key", "ttl_value", ttl=1)
        assert await storage.exists("ttl_key")
        await asyncio.sleep(1.1)  # Wait for TTL to expire
        assert not await storage.exists("ttl_key")
    
    async def test_complex_values(self, storage: CompleteStorage):
        """Test storing complex values."""
        test_data = {
            "string": "value",
            "number": 42,
            "list": [1, 2, 3],
            "dict": {"key": "value"},
            "bool": True,
            "null": None
        }
        
        assert await storage.set("complex_key", test_data)
        result = await storage.get("complex_key")
        assert result == test_data
    
    async def test_increment(self, storage: CompleteStorage):
        """Test counter operations."""
        # Test increment on new key
        assert await storage.increment("counter") == 1
        assert await storage.increment("counter") == 2
        assert await storage.increment("counter") == 3
        
        # Test get after increment
        value = await storage.get("counter")
        assert value == 3
    
    async def test_hash_operations(self, storage: CompleteStorage):
        """Test hash operations."""
        # Test single field
        assert await storage.hset("hash1", "field1", "value1") == 1
        assert await storage.hget("hash1", "field1") == "value1"
        
        # Test multiple fields
        mapping = {"field2": "value2", "field3": "value3"}
        assert await storage.hset("hash1", mapping=mapping) == 2
        
        # Test get all
        all_fields = await storage.hgetall("hash1")
        assert all_fields == {
            "field1": "value1",
            "field2": "value2",
            "field3": "value3"
        }
        
        # Test delete fields
        assert await storage.hdel("hash1", "field1", "field2") == 2
        assert await storage.hget("hash1", "field1") is None
        assert await storage.hget("hash1", "field3") == "value3"
    
    async def test_list_operations(self, storage: CompleteStorage):
        """Test list operations."""
        # Test push operations
        assert await storage.lpush("list1", "value1", "value2") == 2
        assert await storage.rpush("list1", "value3", "value4") == 4
        
        # Test range
        values = await storage.lrange("list1", 0, -1)
        assert values == ["value2", "value1", "value3", "value4"]
        
        # Test pop operations
        assert await storage.lpop("list1") == "value2"
        assert await storage.rpop("list1") == "value4"
        
        # Test remaining values
        values = await storage.lrange("list1", 0, -1)
        assert values == ["value1", "value3"]
    
    async def test_pattern_operations(self, storage: CompleteStorage):
        """Test pattern-based operations."""
        # Set up test data
        await storage.set("test:1", "value1")
        await storage.set("test:2", "value2")
        await storage.set("other:1", "value3")
        
        # Test clear pattern
        count = await storage.clear_pattern("test:*")
        assert count == 2
        
        # Verify cleared
        assert not await storage.exists("test:1")
        assert not await storage.exists("test:2")
        assert await storage.exists("other:1")
    
    async def test_pubsub_operations(self, storage: CompleteStorage):
        """Test publish/subscribe operations."""
        received_messages = []
        
        async def message_handler(channel: str, message: Any):
            received_messages.append((channel, message))
        
        # Subscribe to channel
        await storage.subscribe("test_channel", message_handler)
        
        # Publish messages
        await storage.publish("test_channel", "test_message")
        await storage.publish("test_channel", {"key": "value"})
        
        # Wait for messages to be processed
        await asyncio.sleep(0.1)
        
        # Unsubscribe
        await storage.unsubscribe("test_channel", message_handler)
        
        # Verify received messages
        assert len(received_messages) == 2
        assert received_messages[0] == ("test_channel", "test_message")
        assert received_messages[1] == ("test_channel", {"key": "value"})
    
    async def test_lock_operations(self, storage: CompleteStorage):
        """Test distributed locking."""
        # Test basic lock
        assert await storage.acquire_lock("test_lock")
        assert not await storage.acquire_lock("test_lock", blocking_timeout=0)
        assert await storage.release_lock("test_lock")
        
        # Test lock timeout
        assert await storage.acquire_lock("timeout_lock", timeout=1)
        await asyncio.sleep(1.1)  # Wait for lock to expire
        assert await storage.acquire_lock("timeout_lock", blocking_timeout=0)
        
        # Test blocking timeout
        assert await storage.acquire_lock("blocking_lock")
        start_time = asyncio.get_event_loop().time()
        assert not await storage.acquire_lock("blocking_lock", blocking_timeout=0.5)
        elapsed = asyncio.get_event_loop().time() - start_time
        assert 0.4 <= elapsed <= 0.6  # Check blocking timeout
    
    async def test_concurrent_operations(self, storage: CompleteStorage):
        """Test concurrent operations."""
        async def increment_counter():
            for _ in range(10):
                await storage.increment("concurrent_counter")
                await asyncio.sleep(0.01)  # Simulate work
        
        # Run multiple incrementers concurrently
        tasks = [increment_counter() for _ in range(5)]
        await asyncio.gather(*tasks)
        
        # Check final value (should be 50)
        assert await storage.get("concurrent_counter") == 50 
"""
Tests for Redis storage implementation.
"""

import os
import pytest
import asyncio
from typing import AsyncGenerator
import redis.exceptions

from forex_ai.data.storage.base import CompleteStorage
from forex_ai.data.storage.redis_storage import RedisStorage
from tests.storage.test_storage_base import BaseStorageTests
from forex_ai.exceptions import CacheError

class TestRedisStorage(BaseStorageTests):
    """Test cases for Redis storage implementation."""
    
    async def get_storage(self) -> CompleteStorage:
        """Get Redis storage instance."""
        # Use test-specific Redis database
        os.environ["REDIS_DB"] = "1"  # Use DB 1 for tests
        return RedisStorage()
    
    @pytest.fixture(autouse=True)
    async def cleanup_redis(self, storage: CompleteStorage):
        """Clean up Redis after each test."""
        yield
        # Clear all keys in the test database
        if isinstance(storage, RedisStorage):
            await storage.clear_pattern("*")
    
    async def test_redis_connection_error(self):
        """Test Redis connection error handling."""
        # Use invalid Redis port
        os.environ["REDIS_PORT"] = "65535"
        
        with pytest.raises(CacheError):
            RedisStorage()
        
        # Restore valid port
        os.environ["REDIS_PORT"] = "6379"
    
    async def test_redis_reconnection(self, storage: CompleteStorage):
        """Test Redis reconnection behavior."""
        if not isinstance(storage, RedisStorage):
            pytest.skip("Test only applicable to Redis storage")
        
        # Set test data
        await storage.set("test_key", "test_value")
        
        # Simulate connection drop by closing client
        storage._client.close()
        
        # Should reconnect automatically
        assert await storage.get("test_key") == "test_value"
    
    async def test_redis_pubsub_reconnection(self, storage: CompleteStorage):
        """Test Redis pub/sub reconnection."""
        if not isinstance(storage, RedisStorage):
            pytest.skip("Test only applicable to Redis storage")
            
        received_messages = []
        
        async def message_handler(channel: str, message: str):
            received_messages.append(message)
        
        # Subscribe to channel
        await storage.subscribe("test_channel", message_handler)
        
        # Simulate connection drop
        storage._pubsub.close()
        
        # Should reconnect and receive message
        await storage.publish("test_channel", "test_message")
        await asyncio.sleep(0.1)
        
        assert "test_message" in received_messages
    
    async def test_redis_transaction_success(self, storage: CompleteStorage):
        """Test successful Redis transaction."""
        if not isinstance(storage, RedisStorage):
            pytest.skip("Test only applicable to Redis storage")
        
        async with storage.transaction() as tr:
            tr.set("tx_key1", "value1")
            tr.set("tx_key2", "value2")
            tr.incr("tx_counter")
        
        assert await storage.get("tx_key1") == "value1"
        assert await storage.get("tx_key2") == "value2"
        assert await storage.get("tx_counter") == 1
    
    async def test_redis_transaction_rollback(self, storage: CompleteStorage):
        """Test Redis transaction rollback on error."""
        if not isinstance(storage, RedisStorage):
            pytest.skip("Test only applicable to Redis storage")
        
        # Set initial value
        await storage.set("tx_key", "original")
        
        # Transaction with error
        with pytest.raises(Exception):
            async with storage.transaction() as tr:
                tr.set("tx_key", "new_value")
                raise Exception("Simulated error")
        
        # Value should remain unchanged
        assert await storage.get("tx_key") == "original"
    
    async def test_redis_connection_retry(self, storage: CompleteStorage):
        """Test Redis connection retry logic."""
        if not isinstance(storage, RedisStorage):
            pytest.skip("Test only applicable to Redis storage")
        
        # Test retry count
        assert storage._connection_retries == 3
        
        # Test retry delay
        assert storage._retry_delay == 1.0
        
        # Test connection verification
        assert await storage.ping()
        
        # Test connection failure handling
        storage._client.close()
        assert await storage.ping()  # Should reconnect
    
    async def test_redis_concurrent_transactions(self, storage: CompleteStorage):
        """Test concurrent Redis transactions."""
        if not isinstance(storage, RedisStorage):
            pytest.skip("Test only applicable to Redis storage")
        
        async def increment_counter():
            async with storage.transaction() as tr:
                value = await storage.get("concurrent_counter")
                value = int(value) if value else 0
                tr.set("concurrent_counter", value + 1)
                await asyncio.sleep(0.01)  # Simulate work
        
        # Initialize counter
        await storage.set("concurrent_counter", 0)
        
        # Run concurrent increments
        tasks = [increment_counter() for _ in range(10)]
        await asyncio.gather(*tasks)
        
        # Final value should be 10 (no lost updates)
        assert await storage.get("concurrent_counter") == 10
    
    async def test_redis_connection_timeout(self, storage: CompleteStorage):
        """Test Redis connection timeout handling."""
        if not isinstance(storage, RedisStorage):
            pytest.skip("Test only applicable to Redis storage")
        
        # Test connection timeout settings
        assert storage._client.connection_pool.connection_kwargs["socket_timeout"] == 5
        assert storage._client.connection_pool.connection_kwargs["socket_connect_timeout"] == 5
        
        # Test health check interval
        assert storage._client.connection_pool.connection_kwargs["health_check_interval"] == 30
    
    async def test_redis_large_data(self, storage: CompleteStorage):
        """Test handling of large data in Redis."""
        if not isinstance(storage, RedisStorage):
            pytest.skip("Test only applicable to Redis storage")
        
        # Test large string
        large_string = "x" * 1000000  # 1MB string
        assert await storage.set("large_string", large_string)
        assert await storage.get("large_string") == large_string
        
        # Test large list
        large_list = list(range(10000))
        assert await storage.set("large_list", large_list)
        assert await storage.get("large_list") == large_list
        
        # Test large hash
        large_hash = {str(i): f"value_{i}" for i in range(1000)}
        for field, value in large_hash.items():
            assert await storage.hset("large_hash", field, value)
        assert await storage.hgetall("large_hash") == large_hash 
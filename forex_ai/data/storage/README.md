# Storage Implementation

This module provides a robust and flexible storage system for the Forex AI Trading System, with support for both Redis and in-memory storage backends.

## Features

- Thread-safe operations
- Automatic Redis fallback to memory storage
- Comprehensive storage interfaces
- Transaction support
- Memory management
- Connection retry and health checks
- TTL support
- Pub/Sub functionality
- Distributed locking

## Storage Interfaces

The storage system is built on a hierarchy of interfaces:

- `BaseStorage`: Basic key-value operations
- `HashStorage`: Hash map operations
- `ListStorage`: List operations
- `PubSubStorage`: Publish/Subscribe functionality
- `LockStorage`: Distributed locking
- `CompleteStorage`: Combines all interfaces

## Usage

### Basic Usage

```python
from forex_ai.data.storage.factory import get_storage

# Get storage instance (Redis with memory fallback)
storage = await get_storage()

# Store value
await storage.set("key", "value")

# Retrieve value
value = await storage.get("key")

# Delete value
await storage.delete("key")
```

### Redis Transactions

```python
from forex_ai.data.storage.factory import get_storage

storage = await get_storage()

# Use transaction context
async with storage.transaction() as tr:
    tr.set("key1", "value1")
    tr.set("key2", "value2")
    await tr.execute()
```

### Hash Operations

```python
# Store hash fields
await storage.hset("hash_key", "field1", "value1")
await storage.hset("hash_key", mapping={"field2": "value2"})

# Get hash fields
value = await storage.hget("hash_key", "field1")
all_fields = await storage.hgetall("hash_key")
```

### List Operations

```python
# Push values
await storage.lpush("list_key", "value1", "value2")
await storage.rpush("list_key", "value3")

# Pop values
first = await storage.lpop("list_key")
last = await storage.rpop("list_key")

# Get range
values = await storage.lrange("list_key", 0, -1)
```

### Pub/Sub

```python
async def message_handler(channel: str, message: str):
    print(f"Received {message} on {channel}")

# Subscribe to channel
await storage.subscribe("channel", message_handler)

# Publish message
await storage.publish("channel", "Hello!")

# Unsubscribe
await storage.unsubscribe("channel", message_handler)
```

### Distributed Locking

```python
# Acquire lock
if await storage.acquire_lock("resource_lock", timeout=60):
    try:
        # Do work
        pass
    finally:
        await storage.release_lock("resource_lock")
```

## Configuration

### Redis Configuration

Set the following environment variables:

```bash
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=optional_password
```

### Memory Management

The in-memory storage includes automatic memory management:

- Cleanup interval: 60 seconds
- Maximum memory: 1GB
- Memory threshold: 90%
- Automatic cleanup of expired keys
- LRU-like eviction when memory threshold is exceeded

## Error Handling

The storage system includes comprehensive error handling:

- Automatic Redis reconnection
- Connection retry with exponential backoff
- Fallback to memory storage on Redis failure
- Transaction rollback on errors
- Thread-safe operations

## Testing

Run the test suite:

```bash
pytest tests/storage/
```

Test coverage includes:
- Basic operations
- Redis-specific features
- Memory management
- Concurrent access
- Error scenarios
- Connection handling
- Transaction behavior

## Best Practices

1. Always use the factory to get storage instances:
```python
from forex_ai.data.storage.factory import get_storage
storage = await get_storage()
```

2. Use transactions for atomic operations:
```python
async with storage.transaction() as tr:
    # Multiple operations
    await tr.execute()
```

3. Set appropriate TTLs for cached data:
```python
await storage.set("cache_key", data, ttl=3600)  # 1 hour
```

4. Clean up resources:
```python
await storage.close()
```

5. Handle errors appropriately:
```python
try:
    await storage.set("key", "value")
except Exception as e:
    logger.error(f"Storage error: {str(e)}")
```

## Performance Considerations

1. Redis Connection Pool
   - Reuse connections
   - Configure appropriate pool size
   - Set reasonable timeouts

2. Memory Storage
   - Monitor memory usage
   - Configure cleanup intervals
   - Set appropriate TTLs

3. Transactions
   - Use for atomic operations
   - Keep transaction blocks short
   - Handle rollbacks properly

4. Pub/Sub
   - Clean up subscribers
   - Handle backpressure
   - Monitor message queue size 
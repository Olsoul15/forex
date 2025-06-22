"""
Common test fixtures for storage tests.
"""

import os
import pytest
import asyncio
from typing import AsyncGenerator

@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(autouse=True)
def setup_test_env():
    """Set up test environment variables."""
    # Store original environment
    original_env = {
        "REDIS_HOST": os.environ.get("REDIS_HOST"),
        "REDIS_PORT": os.environ.get("REDIS_PORT"),
        "REDIS_DB": os.environ.get("REDIS_DB"),
        "REDIS_PASSWORD": os.environ.get("REDIS_PASSWORD"),
    }
    
    # Set test environment
    os.environ["REDIS_HOST"] = "localhost"
    os.environ["REDIS_PORT"] = "6379"  # Default Redis port
    os.environ["REDIS_DB"] = "1"  # Use DB 1 for tests
    if "REDIS_PASSWORD" in os.environ:
        del os.environ["REDIS_PASSWORD"]
    
    yield
    
    # Restore original environment
    for key, value in original_env.items():
        if value is not None:
            os.environ[key] = value
        elif key in os.environ:
            del os.environ[key]

@pytest.fixture
def redis_env():
    """Set up Redis environment variables."""
    return {
        "host": os.environ["REDIS_HOST"],
        "port": int(os.environ["REDIS_PORT"]),
        "db": int(os.environ["REDIS_DB"]),
    } 
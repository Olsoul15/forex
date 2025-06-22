"""
Storage module for the Forex AI Trading System.

This module provides database and cache storage functionality
using Redis for primary storage with in-memory fallback.
"""

from forex_ai.data.storage.base import (
    BaseStorage,
    HashStorage,
    ListStorage,
    PubSubStorage,
    LockStorage,
    CompleteStorage
)
from forex_ai.data.storage.factory import get_storage

# Export public interfaces
__all__ = [
    'BaseStorage',
    'HashStorage',
    'ListStorage',
    'PubSubStorage',
    'LockStorage',
    'CompleteStorage',
    'get_storage',
] 
"""
Services for the AutoAgent integration with external systems.

This package provides service implementations for integrating with external
systems, including database services, notification services, and more.
"""

from forex_ai.integration.services.supabase_memory_service import SupabaseMemoryService

__all__ = [
    "SupabaseMemoryService",
]

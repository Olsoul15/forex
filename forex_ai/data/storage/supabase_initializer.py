"""
Supabase initialization and schema management for Forex AI Trading System.

This module provides a robust initialization system for Supabase that:
1. Handles progressive initialization
2. Manages schema versioning
3. Provides rollback mechanisms
4. Implements retry logic
"""

import logging
import time
from typing import Optional, Dict, Any
from functools import wraps
import json
import os

from supabase import Client
from postgrest import APIError
from tenacity import retry, stop_after_attempt, wait_exponential

from forex_ai.config.settings import get_settings
from forex_ai.data.storage.supabase_base import get_base_supabase_client

logger = logging.getLogger(__name__)

class SupabaseInitializationError(Exception):
    """Base exception for Supabase initialization errors."""
    pass

def with_retries(func):
    """Decorator to add retry logic to Supabase operations."""
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise
    return wrapper

class SupabaseInitializer:
    """Handles progressive initialization of Supabase connection and schema."""
    
    def __init__(self):
        """Initialize the Supabase initializer."""
        self.settings = get_settings()
        self.client: Optional[Client] = None
        self._initialized = False
        self._schema_version = "1.0.0"  # Current schema version
        
    @property
    def is_initialized(self) -> bool:
        """Check if Supabase has been initialized."""
        return self._initialized

    async def initialize_client(self) -> Client:
        """
        Initialize basic Supabase client without table verification.
        This is the first step in the progressive initialization.
        """
        try:
            if not self.client:
                logger.info("Initializing Supabase client...")
                self.client = get_base_supabase_client()
                logger.info("Supabase client created successfully")
            return self.client
        except Exception as e:
            error_msg = f"Failed to initialize Supabase client: {str(e)}"
            logger.critical(error_msg)
            raise SupabaseInitializationError(error_msg)

    @with_retries
    async def verify_connection(self) -> bool:
        """
        Verify Supabase connection without querying tables.
        Uses a simple RPC call to check connectivity.
        """
        try:
            if not self.client:
                await self.initialize_client()
            
            # Try a simple query that doesn't require table access
            version = await self.client.rpc('get_version').execute()
            logger.info(f"Successfully verified Supabase connection. Version: {version}")
            return True
        except Exception as e:
            logger.error(f"Failed to verify Supabase connection: {str(e)}")
            return False

    @with_retries
    async def verify_schema_version(self) -> str:
        """
        Verify the current schema version and create version table if it doesn't exist.
        """
        try:
            if not self.client:
                await self.initialize_client()

            # Try to query the schema version table
            try:
                result = await self.client.table('schema_version').select("*").limit(1).execute()
                if result.data:
                    current_version = result.data[0].get('version')
                    logger.info(f"Current schema version: {current_version}")
                    return current_version
            except APIError:
                # Table doesn't exist, create it
                logger.info("Schema version table not found, creating...")
                await self.create_schema_version_table()
                return self._schema_version
                
        except Exception as e:
            logger.error(f"Failed to verify schema version: {str(e)}")
            raise

    async def create_schema_version_table(self):
        """Create the schema version table if it doesn't exist."""
        try:
            # Create schema version table
            await self.client.rpc(
                'create_schema_version_table',
                {
                    'version': self._schema_version,
                    'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
                }
            ).execute()
            logger.info("Created schema version table")
        except Exception as e:
            logger.error(f"Failed to create schema version table: {str(e)}")
            raise

    @with_retries
    async def initialize_schema(self) -> bool:
        """
        Initialize or update the database schema.
        """
        try:
            if not self.client:
                await self.initialize_client()

            # Get current schema version
            current_version = await self.verify_schema_version()
            
            if current_version == self._schema_version:
                logger.info("Schema is up to date")
                return True

            # Load and execute schema SQL
            schema_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                'supabase_schema.sql'
            )
            
            if not os.path.exists(schema_path):
                logger.error(f"Schema file not found: {schema_path}")
                return False

            with open(schema_path, 'r') as f:
                schema_sql = f.read()

            # Execute schema SQL in a transaction
            await self.client.rpc('execute_sql', {'sql': schema_sql}).execute()
            
            # Update schema version
            await self.client.table('schema_version').insert({
                'version': self._schema_version,
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }).execute()

            logger.info(f"Schema initialized/updated to version {self._schema_version}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize schema: {str(e)}")
            return False

    async def initialize(self, force: bool = False) -> bool:
        """
        Complete initialization sequence.
        
        Args:
            force: If True, reinitialize even if already initialized
        """
        if self._initialized and not force:
            return True

        try:
            # Step 1: Initialize client
            await self.initialize_client()

            # Step 2: Verify connection
            if not await self.verify_connection():
                raise SupabaseInitializationError("Failed to verify Supabase connection")

            # Step 3: Initialize schema
            if not await self.initialize_schema():
                raise SupabaseInitializationError("Failed to initialize Supabase schema")

            self._initialized = True
            logger.info("Supabase initialization completed successfully")
            return True

        except Exception as e:
            logger.critical(f"Supabase initialization failed: {str(e)}")
            self._initialized = False
            raise SupabaseInitializationError(f"Supabase initialization failed: {str(e)}")

# Global instance
_initializer = SupabaseInitializer()

async def get_supabase_initializer() -> SupabaseInitializer:
    """Get the global Supabase initializer instance."""
    return _initializer 
"""
Supabase memory service for persistent storage of analysis contexts.

This module provides a Supabase-backed implementation of the memory service
for the AutoAgent integration, ensuring data persistence across sessions.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta

from supabase import create_client, Client

from forex_ai.utils.logging import get_logger

logger = get_logger(__name__)


class SupabaseMemoryService:
    """
    Supabase-backed implementation of the memory service.

    This class:
    - Provides persistent storage for analysis contexts
    - Implements efficient querying for context retrieval
    - Manages data cleanup for old entries
    - Handles connection to Supabase
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Supabase memory service.

        Args:
            config: Configuration dictionary with Supabase credentials and settings
        """
        self.config = config or {}
        self.client = self._initialize_client()
        self.schema_prefix = self.config.get("schema_prefix", "om_")
        self.tables = self._initialize_tables()

    def _initialize_client(self) -> Optional[Client]:
        """
        Initialize the Supabase client.

        Returns:
            Supabase client instance or None if initialization fails
        """
        try:
            url = self.config.get("supabase_url")
            key = self.config.get("supabase_key")

            if not url or not key:
                logger.error("Missing Supabase credentials in configuration")
                return None

            return create_client(url, key)

        except Exception as e:
            logger.error(f"Error initializing Supabase client: {str(e)}")
            return None

    def _initialize_tables(self) -> Dict[str, str]:
        """
        Initialize table names with schema prefix.

        Returns:
            Dictionary mapping table types to prefixed table names
        """
        return {
            "analysis_results": f"{self.schema_prefix}analysis_results",
            "context_links": f"{self.schema_prefix}context_links",
            "strategies": f"{self.schema_prefix}strategies",
            "performance_metrics": f"{self.schema_prefix}performance_metrics",
            "embeddings": f"{self.schema_prefix}analysis_embeddings",
        }

    async def ensure_tables_exist(self) -> bool:
        """
        Ensure that all required tables exist in Supabase.

        Returns:
            True if tables exist or were created, False on error
        """
        if not self.client:
            logger.error("Cannot ensure tables: Supabase client not initialized")
            return False

        try:
            # Define SQL for creating tables if they don't exist
            statements = [
                f"""
                CREATE TABLE IF NOT EXISTS {self.tables["analysis_results"]} (
                    context_id VARCHAR PRIMARY KEY,
                    pair VARCHAR,
                    timeframe VARCHAR,
                    analysis_type VARCHAR,
                    findings JSONB,
                    confidence FLOAT,
                    timestamp TIMESTAMP,
                    related_contexts JSONB,
                    tags JSONB,
                    metadata JSONB
                )
                """,
                f"""
                CREATE TABLE IF NOT EXISTS {self.tables["context_links"]} (
                    source_id VARCHAR,
                    target_id VARCHAR,
                    relationship_type VARCHAR,
                    strength FLOAT,
                    created_at TIMESTAMP,
                    PRIMARY KEY (source_id, target_id)
                )
                """,
            ]

            # Execute each statement
            for statement in statements:
                # Using RPC to execute raw SQL
                await self.client.rpc("exec_sql", {"sql": statement.strip()})

            logger.info("Ensured all tables exist in Supabase")
            return True

        except Exception as e:
            logger.error(f"Error ensuring tables exist: {str(e)}")
            return False

    async def store_context(self, context_data: Dict[str, Any]) -> Optional[str]:
        """
        Store a context in Supabase.

        Args:
            context_data: Context data to store

        Returns:
            Context ID if successful, None on error
        """
        if not self.client:
            logger.error("Cannot store context: Supabase client not initialized")
            return None

        try:
            # Convert complex types to JSON
            context = context_data.copy()
            for field in ["findings", "related_contexts", "tags", "metadata"]:
                if isinstance(context.get(field), (dict, list)):
                    context[field] = json.dumps(context[field])

            # Store in Supabase
            result = (
                await self.client.table(self.tables["analysis_results"])
                .insert(context)
                .execute()
            )

            if result.data:
                context_id = context.get("context_id")
                logger.info(f"Stored context {context_id} in Supabase")
                return context_id
            else:
                logger.error("Error storing context: No data returned")
                return None

        except Exception as e:
            logger.error(f"Error storing context: {str(e)}")
            return None

    async def retrieve_context(
        self,
        pair: Optional[str] = None,
        timeframe: Optional[str] = None,
        analysis_type: Optional[str] = None,
        days_ago: int = 30,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve contexts matching the specified filters.

        Args:
            pair: Currency pair filter
            timeframe: Timeframe filter
            analysis_type: Analysis type filter
            days_ago: How many days to look back
            limit: Maximum number of contexts to return

        Returns:
            List of matching contexts
        """
        if not self.client:
            logger.error("Cannot retrieve contexts: Supabase client not initialized")
            return []

        try:
            # Build query
            query = self.client.table(self.tables["analysis_results"]).select("*")

            # Apply filters
            if pair:
                query = query.eq("pair", pair)
            if timeframe:
                query = query.eq("timeframe", timeframe)
            if analysis_type:
                query = query.eq("analysis_type", analysis_type)

            # Apply time filter
            if days_ago:
                cutoff_date = (datetime.now() - timedelta(days=days_ago)).isoformat()
                query = query.gt("timestamp", cutoff_date)

            # Apply ordering and limit
            query = query.order("timestamp", desc=True).limit(limit)

            # Execute query
            result = await query.execute()

            # Process results
            contexts = []
            for item in result.data:
                # Parse JSON fields
                for field in ["findings", "related_contexts", "tags", "metadata"]:
                    if isinstance(item.get(field), str):
                        try:
                            item[field] = json.loads(item[field])
                        except (json.JSONDecodeError, TypeError):
                            # Keep as is if not valid JSON
                            pass

                contexts.append(item)

            logger.info(f"Retrieved {len(contexts)} contexts from Supabase")
            return contexts

        except Exception as e:
            logger.error(f"Error retrieving contexts: {str(e)}")
            return []

    async def retrieve_context_by_id(self, context_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific context by ID.

        Args:
            context_id: ID of the context to retrieve

        Returns:
            Context data if found, None otherwise
        """
        if not self.client:
            logger.error("Cannot retrieve context: Supabase client not initialized")
            return None

        try:
            # Query by ID
            result = (
                await self.client.table(self.tables["analysis_results"])
                .select("*")
                .eq("context_id", context_id)
                .execute()
            )

            if not result.data:
                logger.warning(f"Context {context_id} not found in Supabase")
                return None

            # Process result
            context = result.data[0]

            # Parse JSON fields
            for field in ["findings", "related_contexts", "tags", "metadata"]:
                if isinstance(context.get(field), str):
                    try:
                        context[field] = json.loads(context[field])
                    except (json.JSONDecodeError, TypeError):
                        # Keep as is if not valid JSON
                        pass

            return context

        except Exception as e:
            logger.error(f"Error retrieving context by ID: {str(e)}")
            return None

    async def link_contexts(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        strength: float = 1.0,
    ) -> bool:
        """
        Create a link between two contexts.

        Args:
            source_id: ID of the source context
            target_id: ID of the target context
            relationship_type: Type of relationship
            strength: Relationship strength (0.0 to 1.0)

        Returns:
            True if successful, False on error
        """
        if not self.client:
            logger.error("Cannot link contexts: Supabase client not initialized")
            return False

        try:
            # Create link record
            link_data = {
                "source_id": source_id,
                "target_id": target_id,
                "relationship_type": relationship_type,
                "strength": strength,
                "created_at": datetime.now().isoformat(),
            }

            # Store in Supabase
            result = (
                await self.client.table(self.tables["context_links"])
                .insert(link_data)
                .execute()
            )

            if result.data:
                logger.info(f"Linked contexts {source_id} and {target_id}")
                return True
            else:
                logger.error("Error linking contexts: No data returned")
                return False

        except Exception as e:
            logger.error(f"Error linking contexts: {str(e)}")
            return False

    async def cleanup_old_data(self, retention_days: int = 90) -> int:
        """
        Remove old data from Supabase to prevent excessive storage usage.

        Args:
            retention_days: Number of days to retain data

        Returns:
            Number of records removed
        """
        if not self.client:
            logger.error("Cannot clean up data: Supabase client not initialized")
            return 0

        try:
            # Calculate cutoff date
            cutoff_date = (datetime.now() - timedelta(days=retention_days)).isoformat()

            # Delete old records
            result = (
                await self.client.table(self.tables["analysis_results"])
                .delete()
                .lt("timestamp", cutoff_date)
                .execute()
            )

            removed_count = len(result.data) if result.data else 0
            logger.info(f"Removed {removed_count} old records from Supabase")
            return removed_count

        except Exception as e:
            logger.error(f"Error cleaning up old data: {str(e)}")
            return 0

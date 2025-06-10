"""
Analysis memory manager for persisting and retrieving analysis contexts.

This module provides functionality to save and load analysis contexts from
a database, enabling context awareness across analysis sessions.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import json
import uuid

from forex_ai.data.db_connector import get_db_client
from forex_ai.agents.context.enhanced_memory import AnalysisContext, EnhancedAgentMemory

logger = logging.getLogger(__name__)


class AnalysisMemoryManager:
    """
    Manager for persisting and retrieving analysis memory.
    """

    def __init__(self, table_name: str = "analysis_contexts"):
        """
        Initialize the analysis memory manager.

        Args:
            table_name: Database table name for storing contexts
        """
        self.db_client = get_db_client()
        self.table_name = table_name

    async def save_context(self, context: AnalysisContext) -> str:
        """
        Save analysis context to the database.

        Args:
            context: Analysis context to save

        Returns:
            ID of the saved context
        """
        try:
            # Convert to dict for storage
            context_dict = context.dict()

            # Convert datetime to string for storage
            if isinstance(context_dict.get("timestamp"), datetime):
                context_dict["timestamp"] = context_dict["timestamp"].isoformat()

            # Store in database
            result = (
                await self.db_client.table(self.table_name)
                .insert(context_dict)
                .execute()
            )

            logger.info(f"Saved analysis context {context.analysis_id} to database")
            return context.analysis_id

        except Exception as e:
            logger.error(f"Error saving analysis context: {str(e)}")
            # Fall back to returning the ID even if save failed
            return context.analysis_id

    async def get_context(self, context_id: str) -> Optional[AnalysisContext]:
        """
        Retrieve an analysis context by ID.

        Args:
            context_id: ID of the context to retrieve

        Returns:
            Retrieved analysis context, or None if not found
        """
        try:
            result = (
                await self.db_client.table(self.table_name)
                .select("*")
                .eq("analysis_id", context_id)
                .execute()
            )

            if result.data and len(result.data) > 0:
                context_data = result.data[0]

                # Convert timestamp string back to datetime
                if isinstance(context_data.get("timestamp"), str):
                    context_data["timestamp"] = datetime.fromisoformat(
                        context_data["timestamp"]
                    )

                return AnalysisContext(**context_data)

            return None

        except Exception as e:
            logger.error(f"Error retrieving analysis context: {str(e)}")
            return None

    async def get_related_contexts(
        self,
        pair: str,
        timeframe: str,
        analysis_type: Optional[str] = None,
        days_ago: int = 30,
        limit: int = 10,
    ) -> List[AnalysisContext]:
        """
        Get related analysis contexts from database.

        Args:
            pair: Currency pair
            timeframe: Time frame
            analysis_type: Type of analysis (optional)
            days_ago: Look back period in days
            limit: Maximum number of contexts to return

        Returns:
            List of related analysis contexts
        """
        try:
            # Build query
            query = (
                self.db_client.table(self.table_name)
                .select("*")
                .eq("pair", pair)
                .eq("timeframe", timeframe)
            )

            # Add analysis_type filter if provided
            if analysis_type:
                query = query.eq("analysis_type", analysis_type)

            # Add time filter
            min_date = (datetime.now() - timedelta(days=days_ago)).isoformat()
            query = query.gte("timestamp", min_date)

            # Execute query with limit
            result = await query.order("timestamp", desc=True).limit(limit).execute()

            contexts = []
            for item in result.data:
                # Convert timestamp string back to datetime
                if isinstance(item.get("timestamp"), str):
                    item["timestamp"] = datetime.fromisoformat(item["timestamp"])

                contexts.append(AnalysisContext(**item))

            return contexts

        except Exception as e:
            logger.error(f"Error retrieving related analysis contexts: {str(e)}")
            return []

    async def load_memory(
        self,
        memory: EnhancedAgentMemory,
        pair: str,
        timeframe: str,
        analysis_types: Optional[List[str]] = None,
        days_ago: int = 30,
        limit: int = 20,
    ) -> int:
        """
        Load relevant analysis contexts into memory.

        Args:
            memory: EnhancedAgentMemory to load contexts into
            pair: Currency pair
            timeframe: Time frame
            analysis_types: Types of analysis to include
            days_ago: Look back period in days
            limit: Maximum number of contexts to load

        Returns:
            Number of contexts loaded
        """
        try:
            # Build base query
            query = (
                self.db_client.table(self.table_name)
                .select("*")
                .eq("pair", pair)
                .eq("timeframe", timeframe)
            )

            # Add time filter
            min_date = (datetime.now() - timedelta(days=days_ago)).isoformat()
            query = query.gte("timestamp", min_date)

            # Execute query
            result = await query.order("timestamp", desc=True).limit(limit).execute()

            count = 0
            for item in result.data:
                # Skip if not in requested analysis types
                if analysis_types and item.get("analysis_type") not in analysis_types:
                    continue

                # Convert timestamp string back to datetime
                if isinstance(item.get("timestamp"), str):
                    item["timestamp"] = datetime.fromisoformat(item["timestamp"])

                # Create context and add to memory
                context = AnalysisContext(**item)
                memory.add_analysis_context(context)
                count += 1

            logger.info(f"Loaded {count} analysis contexts into memory")
            return count

        except Exception as e:
            logger.error(f"Error loading analysis contexts into memory: {str(e)}")
            return 0

    async def delete_old_contexts(self, days_ago: int = 90) -> int:
        """
        Delete contexts older than specified number of days.

        Args:
            days_ago: Delete contexts older than this many days

        Returns:
            Number of contexts deleted
        """
        try:
            # Calculate cutoff date
            cutoff_date = (datetime.now() - timedelta(days=days_ago)).isoformat()

            # Delete old contexts
            result = (
                await self.db_client.table(self.table_name)
                .delete()
                .lt("timestamp", cutoff_date)
                .execute()
            )

            deleted_count = len(result.data) if hasattr(result, "data") else 0
            logger.info(f"Deleted {deleted_count} old analysis contexts")

            return deleted_count

        except Exception as e:
            logger.error(f"Error deleting old analysis contexts: {str(e)}")
            return 0

    async def create_table_if_not_exists(self) -> bool:
        """
        Create the analysis contexts table if it doesn't exist.

        Returns:
            True if table created or already exists, False on error
        """
        try:
            # Check if table exists
            tables = await self.db_client.from_().execute()

            # If table doesn't exist in the list of tables
            if self.table_name not in tables:
                # SQL for creating the table
                sql = f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                  analysis_id VARCHAR PRIMARY KEY,
                  timestamp TIMESTAMP NOT NULL,
                  pair VARCHAR NOT NULL,
                  timeframe VARCHAR NOT NULL,
                  analysis_type VARCHAR NOT NULL,
                  findings JSONB NOT NULL,
                  confidence FLOAT,
                  related_analyses VARCHAR[] DEFAULT '{{}}',
                  tags VARCHAR[] DEFAULT '{{}}'
                );
                
                -- Create indexes for efficient querying
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_pair_timeframe 
                    ON {self.table_name}(pair, timeframe);
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_timestamp 
                    ON {self.table_name}(timestamp);
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_analysis_type 
                    ON {self.table_name}(analysis_type);
                """

                # Execute the SQL
                await self.db_client.execute(sql)
                logger.info(f"Created analysis contexts table: {self.table_name}")

            return True

        except Exception as e:
            logger.error(f"Error creating analysis contexts table: {str(e)}")
            return False

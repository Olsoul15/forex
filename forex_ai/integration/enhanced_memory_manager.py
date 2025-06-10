"""
Enhanced memory manager for AutoAgent integration.

This module provides an enhanced memory management system that maintains
context across analysis sessions and supports efficient context retrieval.
"""

import uuid
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import json
import asyncio

from forex_ai.data.db_connector import get_db_client
from forex_ai.utils.logging import get_logger
from forex_ai.integration.services.supabase_memory_service import SupabaseMemoryService

logger = get_logger(__name__)


class AnalysisContext:
    """Model for storing analysis context."""

    def __init__(
        self,
        context_id: Optional[str] = None,
        pair: Optional[str] = None,
        timeframe: Optional[str] = None,
        analysis_type: Optional[str] = None,
        findings: Optional[Dict[str, Any]] = None,
        confidence: float = 0.0,
        timestamp: Optional[datetime] = None,
        related_contexts: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize an analysis context.

        Args:
            context_id: Unique identifier for this context
            pair: Currency pair (e.g., "EUR/USD")
            timeframe: Timeframe (e.g., "1h", "4h", "1d")
            analysis_type: Type of analysis (e.g., "technical", "fundamental")
            findings: Analysis findings
            confidence: Confidence score (0.0 to 1.0)
            timestamp: When the analysis was performed
            related_contexts: IDs of related contexts
            tags: Tags for categorization
            metadata: Additional metadata
        """
        self.context_id = context_id or str(uuid.uuid4())
        self.pair = pair
        self.timeframe = timeframe
        self.analysis_type = analysis_type
        self.findings = findings or {}
        self.confidence = confidence
        self.timestamp = timestamp or datetime.now()
        self.related_contexts = related_contexts or []
        self.tags = tags or []
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "context_id": self.context_id,
            "pair": self.pair,
            "timeframe": self.timeframe,
            "analysis_type": self.analysis_type,
            "findings": self.findings,
            "confidence": self.confidence,
            "timestamp": (
                self.timestamp.isoformat()
                if isinstance(self.timestamp, datetime)
                else self.timestamp
            ),
            "related_contexts": self.related_contexts,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisContext":
        """Create from dictionary."""
        # Convert timestamp string to datetime if needed
        if isinstance(data.get("timestamp"), str):
            try:
                data["timestamp"] = datetime.fromisoformat(data["timestamp"])
            except (ValueError, TypeError):
                data["timestamp"] = datetime.now()

        return cls(**data)


class EnhancedMemoryManager:
    """
    Enhanced memory manager for AutoAgent integration.

    This class provides a comprehensive memory system that:
    - Stores analysis results with relevant metadata
    - Maintains cross-analysis relationships
    - Supports efficient context retrieval
    - Tracks strategy performance metrics
    - Implements tiered memory with relevance decay
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the enhanced memory manager.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Local db client for backwards compatibility
        self.db_client = get_db_client()
        self.memory_schema = self._initialize_schema()

        # In-memory cache for hot data
        self.in_memory_cache = {}
        self.max_cache_size = self.config.get("max_cache_size", 1000)
        self.cache_expiry_seconds = self.config.get(
            "cache_expiry_seconds", 3600
        )  # 1 hour default
        self.last_cache_cleanup = datetime.now()

        # Supabase service for cold storage
        self.supabase_service = SupabaseMemoryService(
            self.config.get("supabase_config")
        )
        self.use_supabase = self.config.get("use_supabase", True)

        # Schedule periodic cache cleanup
        self.cleanup_interval = self.config.get(
            "cleanup_interval_minutes", 60
        )  # Default: clean every hour
        self.cleanup_task = None

    def _initialize_schema(self) -> Dict[str, str]:
        """Initialize the memory schema."""
        schema_prefix = self.config.get("schema_prefix", "om_")

        return {
            "analysis_results": f"{schema_prefix}analysis_results",
            "context_links": f"{schema_prefix}context_links",
            "strategies": f"{schema_prefix}strategies",
            "performance_metrics": f"{schema_prefix}performance_metrics",
            "embeddings": f"{schema_prefix}analysis_embeddings",
        }

    async def ensure_tables_exist(self) -> None:
        """Ensure that all required tables exist in the database."""
        try:
            # For Supabase storage
            if self.use_supabase and self.supabase_service:
                await self.supabase_service.ensure_tables_exist()

            # Legacy db tables for backwards compatibility
            logger.info("Ensuring memory tables exist")

            # Example for analysis_results table
            await self.db_client.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.memory_schema["analysis_results"]} (
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
            """
            )

            # Start cache cleanup task
            self.start_cleanup_task()

        except Exception as e:
            logger.error(f"Error ensuring tables exist: {str(e)}")

    def start_cleanup_task(self) -> None:
        """Start periodic cache cleanup task."""
        if self.cleanup_task is None:
            self.cleanup_task = asyncio.create_task(self._periodic_cleanup())
            logger.info(
                f"Started memory cache cleanup task (interval: {self.cleanup_interval} minutes)"
            )

    async def _periodic_cleanup(self) -> None:
        """Periodically clean up memory cache and old data."""
        while True:
            try:
                # Wait for the cleanup interval
                await asyncio.sleep(self.cleanup_interval * 60)

                # Clean memory cache
                removed_from_cache = self._cleanup_memory_cache()
                logger.info(
                    f"Cleaned memory cache: removed {removed_from_cache} expired items"
                )

                # If using Supabase, clean up old data periodically
                if self.use_supabase and self.supabase_service:
                    retention_days = self.config.get("retention_days", 90)
                    removed_from_db = await self.supabase_service.cleanup_old_data(
                        retention_days
                    )
                    logger.info(
                        f"Cleaned Supabase: removed {removed_from_db} old records"
                    )

            except Exception as e:
                logger.error(f"Error in periodic cleanup: {str(e)}")
                # Sleep a short time to avoid tight loop in case of persistent errors
                await asyncio.sleep(60)

    def _cleanup_memory_cache(self) -> int:
        """
        Clean up expired items from memory cache.

        Returns:
            Number of items removed from cache
        """
        # Don't clean too frequently
        now = datetime.now()
        if (
            now - self.last_cache_cleanup
        ).total_seconds() < 60:  # At least 1 minute between cleanups
            return 0

        # Track how many items are removed
        removed_count = 0

        # Check each item for expiry
        expired_keys = []
        for context_id, context_data in self.in_memory_cache.items():
            # Get timestamp from context data
            timestamp = context_data.get("timestamp")
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp)
                except (ValueError, TypeError):
                    timestamp = None

            # Check if expired
            if (
                timestamp
                and (now - timestamp).total_seconds() > self.cache_expiry_seconds
            ):
                expired_keys.append(context_id)

        # Remove expired items
        for key in expired_keys:
            del self.in_memory_cache[key]
            removed_count += 1

        # Update last cleanup time
        self.last_cache_cleanup = now

        return removed_count

    async def store_analysis_result(
        self, analysis_result: Dict[str, Any], analysis_type: str
    ) -> str:
        """
        Store an analysis result in memory.

        Args:
            analysis_result: Analysis result to store
            analysis_type: Type of analysis

        Returns:
            Context ID of the stored analysis
        """
        # Generate a unique ID for this context if not provided
        context_id = analysis_result.get("context_id") or str(uuid.uuid4())

        # Create context object
        context = AnalysisContext(
            context_id=context_id,
            pair=analysis_result.get("pair"),
            timeframe=analysis_result.get("timeframe"),
            analysis_type=analysis_type,
            findings=analysis_result,
            confidence=analysis_result.get("confidence", 0.5),
            timestamp=datetime.now(),
            tags=analysis_result.get("tags", []),
            metadata=analysis_result.get("metadata", {}),
        )

        # Store context
        await self._store_context(context.to_dict())

        # Find and link related contexts
        await self._link_related_contexts(context_id, context.pair, context.timeframe)

        # Create embeddings for semantic search
        await self._create_embeddings(context_id, context.to_dict())

        return context_id

    async def _store_context(self, context_dict: Dict[str, Any]) -> None:
        """
        Store context in cache and optionally in Supabase.

        Args:
            context_dict: Context data to store
        """
        # Add to in-memory cache
        self._add_to_memory_cache(context_dict)

        # Store in Supabase if enabled
        if self.use_supabase and self.supabase_service:
            try:
                await self.supabase_service.store_context(context_dict)
            except Exception as e:
                logger.error(f"Error storing context to Supabase: {str(e)}")

    async def _create_embeddings(
        self, context_id: str, context_data: Dict[str, Any]
    ) -> None:
        """
        Create embeddings for semantic search.

        Args:
            context_id: Context ID
            context_data: Context data
        """
        # This is a placeholder for embedding creation
        # In a real implementation, we would create vector embeddings for semantic search
        pass

    async def _link_related_contexts(
        self, context_id: str, pair: Optional[str], timeframe: Optional[str]
    ) -> None:
        """
        Find and link related contexts.

        Args:
            context_id: ID of the context to link from
            pair: Currency pair
            timeframe: Timeframe
        """
        try:
            if not pair or not timeframe:
                return

            # Find recent contexts for the same pair and timeframe
            related_contexts = await self.retrieve_context(
                pair=pair,
                timeframe=timeframe,
                days_ago=7,  # Look back 7 days
                limit=5,  # Link to at most 5 related contexts
            )

            # Get IDs of related contexts (excluding self)
            related_ids = [
                c["context_id"]
                for c in related_contexts
                if c["context_id"] != context_id
            ]

            # Create links
            if self.use_supabase and self.supabase_service:
                for related_id in related_ids:
                    await self.supabase_service.link_contexts(
                        source_id=context_id,
                        target_id=related_id,
                        relationship_type="temporal_sequence",
                        strength=0.8,
                    )

            logger.debug(
                f"Linked context {context_id} to {len(related_ids)} related contexts"
            )

        except Exception as e:
            logger.error(f"Error linking related contexts: {str(e)}")

    def _add_to_memory_cache(self, context_data: Dict[str, Any]) -> None:
        """
        Add context to in-memory cache.

        Args:
            context_data: Context data
        """
        # Check if cache cleanup is needed
        if len(self.in_memory_cache) >= self.max_cache_size:
            # Apply LRU (Least Recently Used) strategy - sort by timestamp and remove oldest
            sorted_keys = sorted(
                self.in_memory_cache.keys(),
                key=lambda k: self.in_memory_cache[k].get("timestamp", ""),
                reverse=True,
            )

            # Keep only the newest entries (80% of max size to avoid frequent cleanups)
            keys_to_keep = sorted_keys[: int(self.max_cache_size * 0.8)]
            self.in_memory_cache = {k: self.in_memory_cache[k] for k in keys_to_keep}

            removed_count = len(sorted_keys) - len(keys_to_keep)
            logger.debug(
                f"Cache cleanup: removed {removed_count} least recently used items"
            )

        # Add to cache
        context_id = context_data.get("context_id")
        if context_id:
            self.in_memory_cache[context_id] = context_data

    async def retrieve_context(
        self,
        pair: Optional[str] = None,
        timeframe: Optional[str] = None,
        analysis_type: Optional[str] = None,
        days_ago: int = 30,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve context from memory.

        Args:
            pair: Currency pair filter
            timeframe: Timeframe filter
            analysis_type: Analysis type filter
            days_ago: How many days to look back
            limit: Maximum number of contexts to retrieve

        Returns:
            List of context items
        """
        # First try to get from Supabase if enabled
        if self.use_supabase and self.supabase_service:
            contexts = await self.supabase_service.retrieve_context(
                pair=pair,
                timeframe=timeframe,
                analysis_type=analysis_type,
                days_ago=days_ago,
                limit=limit,
            )

            # Add results to cache for future quick access
            for context in contexts:
                self._add_to_memory_cache(context)

            return contexts

        # Otherwise use legacy retrieval
        try:
            query = self.db_client.table(self.memory_schema["analysis_results"]).select(
                "*"
            )

            # Apply filters
            if pair:
                query = query.eq("pair", pair)
            if timeframe:
                query = query.eq("timeframe", timeframe)
            if analysis_type:
                query = query.eq("analysis_type", analysis_type)

            # Time filter
            if days_ago:
                cutoff_date = (datetime.now() - timedelta(days=days_ago)).isoformat()
                query = query.gt("timestamp", cutoff_date)

            # Order and limit
            query = query.order("timestamp", desc=True).limit(limit)

            # Execute
            result = await query.execute()

            # Process results
            contexts = []
            for item in result.data:
                # Convert JSON strings back to objects
                for field in ["findings", "related_contexts", "tags", "metadata"]:
                    if isinstance(item.get(field), str):
                        try:
                            item[field] = json.loads(item[field])
                        except (json.JSONDecodeError, TypeError):
                            # Keep as is if not valid JSON
                            pass

                contexts.append(item)

                # Add to cache for future quick access
                self._add_to_memory_cache(item)

            return contexts

        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return []

    async def retrieve_context_by_id(self, context_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific context by ID.

        Args:
            context_id: ID of the context to retrieve

        Returns:
            Context data or None if not found
        """
        # Check in-memory cache first for fastest retrieval
        if context_id in self.in_memory_cache:
            return self.in_memory_cache[context_id]

        # Check Supabase if enabled
        if self.use_supabase and self.supabase_service:
            context = await self.supabase_service.retrieve_context_by_id(context_id)
            if context:
                # Add to cache for future quick access
                self._add_to_memory_cache(context)
                return context

        # Legacy retrieval if not found or Supabase not enabled
        try:
            result = (
                await self.db_client.table(self.memory_schema["analysis_results"])
                .select("*")
                .eq("context_id", context_id)
                .execute()
            )

            if not result.data or not result.data[0]:
                return None

            # Process result
            item = result.data[0]

            # Convert JSON strings back to objects
            for field in ["findings", "related_contexts", "tags", "metadata"]:
                if isinstance(item.get(field), str):
                    try:
                        item[field] = json.loads(item[field])
                    except (json.JSONDecodeError, TypeError):
                        # Keep as is if not valid JSON
                        pass

            # Add to cache for future quick access
            self._add_to_memory_cache(item)

            return item

        except Exception as e:
            logger.error(f"Error retrieving context by ID: {str(e)}")
            return None

    async def get_related_contexts(
        self, context_id: str, relationship_type: Optional[str] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get contexts related to the specified context.

        Args:
            context_id: ID of the context
            relationship_type: Type of relationship to filter by
            limit: Maximum number of related contexts to return

        Returns:
            List of related contexts
        """
        try:
            # Query context links
            query = (
                self.db_client.table(self.memory_schema["context_links"])
                .select("target_id")
                .eq("source_id", context_id)
            )

            # Apply relationship filter if provided
            if relationship_type:
                query = query.eq("relationship_type", relationship_type)

            # Limit results
            query = query.limit(limit)

            # Execute
            result = await query.execute()

            if not result.data:
                return []

            # Extract target IDs
            target_ids = [
                item.get("target_id") for item in result.data if item.get("target_id")
            ]

            # Retrieve full contexts
            contexts = []
            for target_id in target_ids:
                context = await self.retrieve_context_by_id(target_id)
                if context:
                    contexts.append(context)

            return contexts

        except Exception as e:
            logger.error(f"Error retrieving related contexts: {str(e)}")
            return []

    async def search_contexts(
        self, query_text: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for contexts using semantic search.

        Args:
            query_text: Text to search for
            limit: Maximum number of results

        Returns:
            List of matching contexts
        """
        # This is a placeholder - actual implementation would depend on
        # your embedding model and vector database capabilities

        # In a real implementation, you would:
        # 1. Generate embedding vector for the query
        # 2. Search for similar vectors in your database
        # 3. Return the matching contexts

        logger.info(f"Semantic search for: {query_text}")

        # Fallback to simple text search
        return await self.simple_text_search(query_text, limit)

    async def simple_text_search(
        self, query_text: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Simple text-based search fallback.

        Args:
            query_text: Text to search for
            limit: Maximum number of results

        Returns:
            List of matching contexts
        """
        try:
            # This is a simplified implementation
            # In a real system, you would use full-text search capabilities of your database

            # For now, we'll retrieve recent contexts and filter in Python
            recent_contexts = await self.retrieve_context(days_ago=90, limit=100)

            # Filter contexts that contain the query text
            matching_contexts = []
            for context in recent_contexts:
                # Search in findings
                findings_str = json.dumps(context.get("findings", {})).lower()

                # Check if query appears in findings
                if query_text.lower() in findings_str:
                    matching_contexts.append(context)

                    # Stop once we have enough matches
                    if len(matching_contexts) >= limit:
                        break

            return matching_contexts

        except Exception as e:
            logger.error(f"Error in simple text search: {str(e)}")
            return []

    async def summarize_contexts(self, contexts: List[Dict[str, Any]]) -> str:
        """
        Generate a summary of multiple contexts.

        Args:
            contexts: List of contexts to summarize

        Returns:
            Formatted summary string
        """
        if not contexts:
            return "No analysis contexts available."

        # Sort by timestamp (newest first)
        sorted_contexts = sorted(
            contexts, key=lambda c: c.get("timestamp", ""), reverse=True
        )

        # Format contexts for LLM consumption
        result = "Previous analyses in chronological order:\n\n"

        for i, context in enumerate(reversed(sorted_contexts), 1):
            timestamp = context.get("timestamp", "Unknown time")
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp)
                    timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M")
                except (ValueError, TypeError):
                    timestamp_str = timestamp
            else:
                timestamp_str = str(timestamp)

            pair = context.get("pair", "Unknown pair")
            timeframe = context.get("timeframe", "Unknown timeframe")
            analysis_type = context.get("analysis_type", "Unknown type")
            confidence = context.get("confidence", 0)

            result += f"{i}. {timestamp_str}: {pair} ({timeframe})\n"
            result += f"   Type: {analysis_type}, Confidence: {confidence:.2f}\n"

            # Add findings summary
            findings = context.get("findings", {})
            if findings:
                # Format depends on the structure of your findings
                if isinstance(findings, dict):
                    # For dictionary findings
                    for key, value in findings.items():
                        if key not in ["pair", "timeframe", "timestamp"]:
                            result += f"   - {key}: {value}\n"
                else:
                    # For other formats
                    result += f"   Findings: {json.dumps(findings, indent=3)}\n"

            result += "\n"

        return result

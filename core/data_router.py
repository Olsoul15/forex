"""
Data routing for the AI Forex system with AutoAgent integration.

This module provides a system for routing data to appropriate processing tiers
based on data type and urgency, ensuring time-sensitive data is processed with
minimal latency while allowing deeper analysis of less time-sensitive data.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Callable, Awaitable, Tuple, Union
from enum import Enum
from datetime import datetime

from forex_ai.utils.logging import get_logger

logger = get_logger(__name__)


class ProcessingTier(str, Enum):
    """Enum for processing tiers."""

    DIRECT = "direct"
    ORCHESTRATION = "orchestration"
    DEEP_RESEARCH = "deep_research"


class Priority(str, Enum):
    """Enum for processing priorities."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class DataType(str, Enum):
    """Enum for data types."""

    PRICE_TICK = "PRICE_TICK"
    PATTERN_CONFIRMATION = "PATTERN_CONFIRMATION"
    NEWS_HEADLINE = "NEWS_HEADLINE"
    ECONOMIC_INDICATOR = "ECONOMIC_INDICATOR"
    SOCIAL_SENTIMENT = "SOCIAL_SENTIMENT"
    STRATEGY_RESEARCH = "STRATEGY_RESEARCH"
    UNKNOWN = "UNKNOWN"


class RoutingMetrics:
    """Class for tracking routing metrics."""

    def __init__(self):
        """Initialize routing metrics."""
        self.route_counts: Dict[str, Dict[str, int]] = {}
        self.latencies: Dict[str, List[float]] = {}
        self.errors: Dict[str, int] = {}
        self.start_time = datetime.now()

    def record_routing(self, data_type: str, tier: str) -> None:
        """
        Record a routing decision.

        Args:
            data_type: Type of data
            tier: Processing tier
        """
        if data_type not in self.route_counts:
            self.route_counts[data_type] = {}

        if tier not in self.route_counts[data_type]:
            self.route_counts[data_type][tier] = 0

        self.route_counts[data_type][tier] += 1

    def record_latency(self, data_type: str, latency_ms: float) -> None:
        """
        Record processing latency.

        Args:
            data_type: Type of data
            latency_ms: Latency in milliseconds
        """
        if data_type not in self.latencies:
            self.latencies[data_type] = []

        self.latencies[data_type].append(latency_ms)

    def record_error(self, data_type: str) -> None:
        """
        Record a processing error.

        Args:
            data_type: Type of data
        """
        if data_type not in self.errors:
            self.errors[data_type] = 0

        self.errors[data_type] += 1

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of routing metrics.

        Returns:
            Dictionary with routing metrics
        """
        summary = {
            "route_counts": self.route_counts,
            "errors": self.errors,
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
        }

        # Calculate average latencies
        avg_latencies = {}
        for data_type, latencies in self.latencies.items():
            if latencies:
                avg_latencies[data_type] = sum(latencies) / len(latencies)

        summary["avg_latencies_ms"] = avg_latencies

        return summary


class DataRouter:
    """
    Routes incoming data to appropriate processing tiers based on data type and urgency.

    This class:
    - Routes data to the appropriate processing tier
    - Ensures time-sensitive data is processed with minimal latency
    - Tracks routing decisions for monitoring and optimization
    - Implements fallback mechanisms for high-volume periods
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data router.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.routing_rules = self._load_routing_rules()
        self.metrics = RoutingMetrics()

        # Initialize handlers for each tier
        self.direct_handlers: Dict[str, Callable] = {}
        self.orchestration_handler: Optional[Callable] = None
        self.deep_research_handler: Optional[Callable] = None

        # Task queues for each tier
        self.orchestration_queue = asyncio.Queue()
        self.deep_research_queue = asyncio.Queue()

        # Status
        self.running = False
        self.workers: List[asyncio.Task] = []

    def _load_routing_rules(self) -> Dict[str, Dict[str, Any]]:
        """
        Load routing rules from configuration.

        Returns:
            Dictionary of routing rules
        """
        default_rules = {
            DataType.PRICE_TICK.value: {
                "tier": ProcessingTier.DIRECT.value,
                "priority": Priority.HIGH.value,
                "timeout_ms": 50,
            },
            DataType.PATTERN_CONFIRMATION.value: {
                "tier": ProcessingTier.DIRECT.value,
                "priority": Priority.HIGH.value,
                "timeout_ms": 100,
            },
            DataType.NEWS_HEADLINE.value: {
                "tier": ProcessingTier.DIRECT.value,
                "priority": Priority.MEDIUM.value,
                "timeout_ms": 200,
            },
            DataType.ECONOMIC_INDICATOR.value: {
                "tier": ProcessingTier.ORCHESTRATION.value,
                "priority": Priority.MEDIUM.value,
                "timeout_ms": 500,
            },
            DataType.SOCIAL_SENTIMENT.value: {
                "tier": ProcessingTier.ORCHESTRATION.value,
                "priority": Priority.LOW.value,
                "timeout_ms": 1000,
            },
            DataType.STRATEGY_RESEARCH.value: {
                "tier": ProcessingTier.DEEP_RESEARCH.value,
                "priority": Priority.LOW.value,
                "timeout_ms": 10000,
            },
        }

        # Merge with config rules if provided
        config_rules = self.config.get("routing_rules", {})

        # Use config rules with fallback to defaults
        merged_rules = default_rules.copy()
        merged_rules.update(config_rules)

        return merged_rules

    def _infer_data_type(self, data: Dict[str, Any]) -> str:
        """
        Infer data type from data characteristics.

        Args:
            data: Data to analyze

        Returns:
            Inferred data type
        """
        # Check if type is explicitly provided
        if "type" in data:
            return data["type"]

        # Infer from data structure
        if "price" in data or "open" in data or "close" in data:
            return DataType.PRICE_TICK.value

        if "pattern" in data or "indicator" in data:
            return DataType.PATTERN_CONFIRMATION.value

        if "headline" in data or "news" in data:
            return DataType.NEWS_HEADLINE.value

        if "economic" in data or "report" in data:
            return DataType.ECONOMIC_INDICATOR.value

        if "sentiment" in data or "social" in data:
            return DataType.SOCIAL_SENTIMENT.value

        if "research" in data or "strategy" in data:
            return DataType.STRATEGY_RESEARCH.value

        # Default to unknown
        return DataType.UNKNOWN.value

    def register_direct_handler(self, data_type: str, handler: Callable) -> None:
        """
        Register a handler for direct processing of a specific data type.

        Args:
            data_type: Data type to handle
            handler: Handler function
        """
        self.direct_handlers[data_type] = handler

    def register_orchestration_handler(self, handler: Callable) -> None:
        """
        Register a handler for orchestration processing.

        Args:
            handler: Handler function
        """
        self.orchestration_handler = handler

    def register_deep_research_handler(self, handler: Callable) -> None:
        """
        Register a handler for deep research processing.

        Args:
            handler: Handler function
        """
        self.deep_research_handler = handler

    async def start(self) -> None:
        """Start the router background workers."""
        if self.running:
            return

        self.running = True

        # Start worker tasks
        orchestration_workers = self.config.get("orchestration_workers", 2)
        deep_research_workers = self.config.get("deep_research_workers", 1)

        for _ in range(orchestration_workers):
            worker = asyncio.create_task(self._orchestration_worker())
            self.workers.append(worker)

        for _ in range(deep_research_workers):
            worker = asyncio.create_task(self._deep_research_worker())
            self.workers.append(worker)

        logger.info(f"Started {len(self.workers)} router background workers")

    async def stop(self) -> None:
        """Stop the router background workers."""
        if not self.running:
            return

        self.running = False

        # Cancel all worker tasks
        for worker in self.workers:
            worker.cancel()

        # Wait for all workers to finish
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)

        self.workers = []

        logger.info("Stopped router background workers")

    async def _orchestration_worker(self) -> None:
        """Worker for processing orchestration queue."""
        while self.running:
            try:
                # Get item from queue
                data, data_type, rule = await self.orchestration_queue.get()

                # Process item
                await self._process_orchestration_item(data, data_type, rule)

                # Mark item as done
                self.orchestration_queue.task_done()

            except asyncio.CancelledError:
                break

            except Exception as e:
                logger.error(f"Error in orchestration worker: {str(e)}")

    async def _deep_research_worker(self) -> None:
        """Worker for processing deep research queue."""
        while self.running:
            try:
                # Get item from queue
                data, data_type, rule = await self.deep_research_queue.get()

                # Process item
                await self._process_deep_research_item(data, data_type, rule)

                # Mark item as done
                self.deep_research_queue.task_done()

            except asyncio.CancelledError:
                break

            except Exception as e:
                logger.error(f"Error in deep research worker: {str(e)}")

    async def route(
        self, data: Dict[str, Any], data_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Route data to appropriate processing tier.

        Args:
            data: The data to route
            data_type: Type of data (will be inferred if not provided)

        Returns:
            Result from processing tier
        """
        start_time = time.time()

        try:
            # Determine data type if not provided
            if data_type is None:
                data_type = self._infer_data_type(data)

            # Get routing rule
            rule = self.routing_rules.get(
                data_type,
                {
                    "tier": ProcessingTier.ORCHESTRATION.value,
                    "priority": Priority.MEDIUM.value,
                    "timeout_ms": 500,
                },
            )

            # Track routing decision
            self.metrics.record_routing(data_type, rule["tier"])

            # Route based on tier
            if rule["tier"] == ProcessingTier.DIRECT.value:
                result = await self._route_to_direct(data, data_type, rule)
            elif rule["tier"] == ProcessingTier.ORCHESTRATION.value:
                result = await self._route_to_orchestration(data, data_type, rule)
            else:  # DEEP_RESEARCH
                result = await self._route_to_deep_research(data, data_type, rule)

            # Calculate and record latency
            latency_ms = (time.time() - start_time) * 1000
            self.metrics.record_latency(data_type, latency_ms)

            return result

        except Exception as e:
            # Record error
            if data_type:
                self.metrics.record_error(data_type)

            logger.error(f"Error routing data: {str(e)}")

            # Return error response
            return {"success": False, "error": str(e), "data_type": data_type}

    async def _route_to_direct(
        self, data: Dict[str, Any], data_type: str, rule: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Route data to direct processing.

        Args:
            data: Data to process
            data_type: Type of data
            rule: Routing rule

        Returns:
            Processing result
        """
        # Get handler for this data type
        handler = self.direct_handlers.get(data_type)

        if not handler:
            # Try to get a default handler
            handler = self.direct_handlers.get("default")

            if not handler:
                raise ValueError(f"No handler registered for data type: {data_type}")

        # Set timeout based on rule
        timeout_ms = rule.get("timeout_ms", 100)

        try:
            # Process with timeout
            result = await asyncio.wait_for(
                handler(data), timeout=timeout_ms / 1000  # Convert to seconds
            )

            return result

        except asyncio.TimeoutError:
            logger.warning(
                f"Direct processing timed out for {data_type}, falling back to orchestration"
            )

            # Fall back to orchestration tier
            return await self._route_to_orchestration(data, data_type, rule)

    async def _route_to_orchestration(
        self, data: Dict[str, Any], data_type: str, rule: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Route data to orchestration processing.

        Args:
            data: Data to process
            data_type: Type of data
            rule: Routing rule

        Returns:
            Processing result or acknowledgment
        """
        if self.orchestration_handler:
            # If high priority, process immediately
            if rule.get("priority") == Priority.HIGH.value:
                return await self.orchestration_handler(data, data_type)

            # Otherwise, queue for background processing
            await self.orchestration_queue.put((data, data_type, rule))

            # Return acknowledgment
            return {
                "success": True,
                "message": f"Data queued for orchestration processing",
                "data_type": data_type,
                "queue_size": self.orchestration_queue.qsize(),
            }
        else:
            return {
                "success": False,
                "message": "No orchestration handler registered",
                "data_type": data_type,
            }

    async def _route_to_deep_research(
        self, data: Dict[str, Any], data_type: str, rule: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Route data to deep research processing.

        Args:
            data: Data to process
            data_type: Type of data
            rule: Routing rule

        Returns:
            Processing result or acknowledgment
        """
        if self.deep_research_handler:
            # Queue for background processing
            await self.deep_research_queue.put((data, data_type, rule))

            # Return acknowledgment
            return {
                "success": True,
                "message": f"Data queued for deep research processing",
                "data_type": data_type,
                "queue_size": self.deep_research_queue.qsize(),
            }
        else:
            return {
                "success": False,
                "message": "No deep research handler registered",
                "data_type": data_type,
            }

    async def _process_orchestration_item(
        self, data: Dict[str, Any], data_type: str, rule: Dict[str, Any]
    ) -> None:
        """
        Process an item from the orchestration queue.

        Args:
            data: Data to process
            data_type: Type of data
            rule: Routing rule
        """
        if not self.orchestration_handler:
            logger.warning("Orchestration item received but no handler registered")
            return

        try:
            # Process with the orchestration handler
            await self.orchestration_handler(data, data_type)

        except Exception as e:
            logger.error(f"Error processing orchestration item: {str(e)}")
            self.metrics.record_error(data_type)

    async def _process_deep_research_item(
        self, data: Dict[str, Any], data_type: str, rule: Dict[str, Any]
    ) -> None:
        """
        Process an item from the deep research queue.

        Args:
            data: Data to process
            data_type: Type of data
            rule: Routing rule
        """
        if not self.deep_research_handler:
            logger.warning("Deep research item received but no handler registered")
            return

        try:
            # Process with the deep research handler
            await self.deep_research_handler(data, data_type)

        except Exception as e:
            logger.error(f"Error processing deep research item: {str(e)}")
            self.metrics.record_error(data_type)

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get routing metrics.

        Returns:
            Dictionary with routing metrics
        """
        metrics = self.metrics.get_summary()

        # Add queue sizes
        metrics["queue_sizes"] = {
            "orchestration": self.orchestration_queue.qsize(),
            "deep_research": self.deep_research_queue.qsize(),
        }

        return metrics

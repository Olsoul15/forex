"""
Base classes for the health check system.

This module defines the foundational classes used by the health check system
to standardize health check behaviors and reporting.
"""

from enum import Enum
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import logging
import time
import traceback
from threading import Lock

# Configure logging
logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Status values for health checks."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    LOCKED = "locked"
    RECOVERING = "recovering"

    @property
    def is_operational(self) -> bool:
        """Return True if the status indicates the component is operational."""
        return self in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]


class ComponentHealth:
    """
    Health information for a specific component.

    This class encapsulates all health-related information for a component,
    including its status, details about any issues, and performance metrics.
    """

    def __init__(
        self,
        name: str,
        status: HealthStatus = HealthStatus.UNKNOWN,
        message: str = "",
        details: Optional[Dict[str, Any]] = None,
        last_checked: Optional[datetime] = None,
        dependencies: Optional[List[str]] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize component health information.

        Args:
            name: The name of the component
            status: Current health status
            message: Human-readable status message
            details: Detailed information about the component's health
            last_checked: When the health was last checked
            dependencies: List of dependencies this component relies on
            metrics: Performance metrics for this component
        """
        self.name = name
        self.status = status
        self.message = message
        self.details = details or {}
        self.last_checked = last_checked or datetime.now()
        self.dependencies = dependencies or []
        self.metrics = metrics or {}
        self.history = []  # Track status changes

        # Record initial state
        self._record_status_change(status, message)

    def update(
        self,
        status: HealthStatus,
        message: str = "",
        details: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update the health information.

        Args:
            status: New health status
            message: Human-readable status message
            details: Detailed information about the component's health
            metrics: Updated performance metrics
        """
        # Only record if status changed
        if status != self.status:
            self._record_status_change(status, message)

        self.status = status
        self.message = message
        if details:
            self.details.update(details)
        if metrics:
            self.metrics.update(metrics)
        self.last_checked = datetime.now()

    def _record_status_change(self, status: HealthStatus, message: str) -> None:
        """Record a status change in the history."""
        self.history.append(
            {
                "timestamp": datetime.now(),
                "from_status": (
                    self.status if hasattr(self, "status") else HealthStatus.UNKNOWN
                ),
                "to_status": status,
                "message": message,
            }
        )

        # Keep history at a reasonable size
        if len(self.history) > 100:
            self.history = self.history[-100:]

    def to_dict(self) -> Dict[str, Any]:
        """Convert health information to a dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "last_checked": self.last_checked.isoformat(),
            "details": self.details,
            "dependencies": self.dependencies,
            "metrics": self.metrics,
            "history": [
                {
                    "timestamp": entry["timestamp"].isoformat(),
                    "from_status": entry["from_status"].value,
                    "to_status": entry["to_status"].value,
                    "message": entry["message"],
                }
                for entry in self.history[-5:]  # Return only the most recent entries
            ],
        }


class HealthCheck:
    """
    Base class for health checks.

    This abstract class defines the interface and common functionality
    for all health checks in the system.
    """

    def __init__(
        self,
        name: str,
        check_interval: float = 60.0,
        timeout: float = 10.0,
        dependencies: Optional[List[str]] = None,
        recovery_strategy: Optional[Callable] = None,
    ):
        """
        Initialize the health check.

        Args:
            name: Name of the component being checked
            check_interval: How often to run the check (in seconds)
            timeout: Maximum time a check should take (in seconds)
            dependencies: List of dependencies this component relies on
            recovery_strategy: Optional function to call for recovery
        """
        self.name = name
        self.check_interval = check_interval
        self.timeout = timeout
        self.dependencies = dependencies or []
        self.recovery_strategy = recovery_strategy
        self.lock = Lock()

        # Initialize health information
        self.health = ComponentHealth(
            name=name,
            status=HealthStatus.UNKNOWN,
            message="Health check not yet run",
            dependencies=self.dependencies,
        )

        self.last_check_time = 0
        self.consecutive_failures = 0
        self.max_consecutive_failures = 3

    def should_check(self) -> bool:
        """Determine if a health check should be run now."""
        return time.time() - self.last_check_time >= self.check_interval

    def check_health(self) -> ComponentHealth:
        """
        Run the health check and return the component's health.

        This method handles the boilerplate around running the check,
        including timeouts, error handling, and recovery attempts.

        Returns:
            ComponentHealth: The component's health information
        """
        if not self.should_check():
            return self.health

        # Remember when we started the check
        start_time = time.time()
        self.last_check_time = start_time

        # Use lock to prevent concurrent checks
        if not self.lock.acquire(blocking=False):
            logger.debug(f"Health check for {self.name} already in progress, skipping")
            return self.health

        try:
            # Run the actual check with timeout
            details = {}
            metrics = {"check_duration": 0}

            try:
                # Run the check implementation
                result = self._check_implementation()

                # Update health based on result
                if result.get("status") == "success":
                    status = HealthStatus.HEALTHY
                    message = result.get("message", "Component is healthy")
                    details = result.get("details", {})
                    self.consecutive_failures = 0
                else:
                    status = (
                        HealthStatus.DEGRADED
                        if result.get("is_degraded", False)
                        else HealthStatus.UNHEALTHY
                    )
                    message = result.get("message", "Component check failed")
                    details = result.get("details", {})
                    self.consecutive_failures += 1

                    # Try recovery if we have too many failures
                    if (
                        self.consecutive_failures >= self.max_consecutive_failures
                        and self.recovery_strategy is not None
                    ):
                        self._attempt_recovery()

            except Exception as e:
                # Handle check failure
                status = HealthStatus.UNHEALTHY
                message = f"Health check failed: {str(e)}"
                details = {"error": str(e), "traceback": traceback.format_exc()}
                self.consecutive_failures += 1
                logger.error(f"Health check for {self.name} failed: {str(e)}")

                # Try recovery if we have too many failures
                if (
                    self.consecutive_failures >= self.max_consecutive_failures
                    and self.recovery_strategy is not None
                ):
                    self._attempt_recovery()

            # Calculate check duration
            check_duration = time.time() - start_time
            metrics["check_duration"] = check_duration

            # Update health information
            self.health.update(
                status=status, message=message, details=details, metrics=metrics
            )

            logger.debug(f"Health check for {self.name} completed: {status.value}")

            return self.health

        finally:
            self.lock.release()

    def _check_implementation(self) -> Dict[str, Any]:
        """
        Implement the actual health check logic.

        This method should be overridden by subclasses to implement
        the specific health check logic for a component.

        Returns:
            Dict with health check results containing at minimum:
            - status: "success" or "failure"
            - message: Human-readable message
            - details: Dict with check-specific details
            - is_degraded: Whether the failure indicates degraded rather than unhealthy
        """
        raise NotImplementedError(
            "Health check implementation must be provided by subclass"
        )

    def _attempt_recovery(self) -> bool:
        """
        Attempt to recover the component.

        Returns:
            True if recovery was successful, False otherwise
        """
        if self.recovery_strategy is None:
            return False

        logger.info(f"Attempting recovery for {self.name}")
        self.health.update(
            status=HealthStatus.RECOVERING,
            message=f"Attempting recovery after {self.consecutive_failures} failures",
        )

        try:
            result = self.recovery_strategy()
            if result:
                logger.info(f"Recovery successful for {self.name}")
                self.health.update(
                    status=HealthStatus.DEGRADED,
                    message="Component recovered but needs verification",
                )
                self.consecutive_failures = 0
                return True
            else:
                logger.warning(f"Recovery failed for {self.name}")
                self.health.update(
                    status=HealthStatus.UNHEALTHY, message="Recovery attempt failed"
                )
                return False
        except Exception as e:
            logger.error(
                f"Recovery attempt for {self.name} raised an exception: {str(e)}"
            )
            self.health.update(
                status=HealthStatus.UNHEALTHY,
                message=f"Recovery attempt failed with error: {str(e)}",
            )
            return False

"""
Error locking system to prevent cascade failures.

This module provides error locking mechanisms to temporarily disable
components when they encounter repeated failures, preventing cascade
failures across the system.
"""

import logging
import threading
import time
from typing import Dict, List, Any, Optional, Callable, Set, Tuple
from datetime import datetime, timedelta

from forex_ai.health.base import HealthStatus

# Configure logging
logger = logging.getLogger(__name__)


class ErrorLock:
    """
    Error lock for a specific component.

    This class implements a lock that is activated when a component
    experiences repeated failures, preventing further operation until
    the lock expires or is manually released.
    """

    def __init__(
        self,
        component_name: str,
        lock_threshold: int = 3,
        lock_duration: float = 300.0,
        cooldown_duration: float = 60.0,
        escalation_factor: float = 2.0,
        max_lock_duration: float = 3600.0,
    ):
        """
        Initialize the error lock.

        Args:
            component_name: Name of the component to lock
            lock_threshold: Number of consecutive failures before locking
            lock_duration: Initial duration of lock in seconds
            cooldown_duration: Time to wait after success before resetting failure count
            escalation_factor: Factor to multiply lock duration by on repeated locks
            max_lock_duration: Maximum lock duration in seconds
        """
        self.component_name = component_name
        self.lock_threshold = lock_threshold
        self.lock_duration = lock_duration
        self.cooldown_duration = cooldown_duration
        self.escalation_factor = escalation_factor
        self.max_lock_duration = max_lock_duration

        self._lock = threading.RLock()
        self._failure_count = 0
        self._is_locked = False
        self._lock_expiry = None
        self._last_success = None
        self._lock_history = []
        self._current_lock_duration = lock_duration

    def record_failure(self, error: Optional[Exception] = None) -> bool:
        """
        Record a component failure.

        Args:
            error: Optional exception that caused the failure

        Returns:
            True if component is now locked, False otherwise
        """
        with self._lock:
            # If already locked, extend lock if error is severe
            if self._is_locked:
                if error and self._is_severe_error(error):
                    self._extend_lock()
                return True

            # Reset failure count if cooldown period has passed
            if (
                self._last_success
                and (datetime.now() - self._last_success).total_seconds()
                > self.cooldown_duration
            ):
                self._failure_count = 0

            # Increment failure count
            self._failure_count += 1

            # Check if threshold reached
            if self._failure_count >= self.lock_threshold:
                self._activate_lock()
                return True

            return False

    def record_success(self) -> None:
        """Record a successful component operation."""
        with self._lock:
            self._last_success = datetime.now()

            # Reset failure count if component is not locked
            if not self._is_locked:
                self._failure_count = 0

    def is_locked(self) -> bool:
        """
        Check if the component is currently locked.

        Returns:
            True if component is locked, False otherwise
        """
        with self._lock:
            # If not locked, return False
            if not self._is_locked:
                return False

            # Check if lock has expired
            if datetime.now() > self._lock_expiry:
                # Lock expired, clear it
                self._deactivate_lock()
                return False

            return True

    def release(self) -> bool:
        """
        Manually release a lock.

        Returns:
            True if lock was released, False if not locked
        """
        with self._lock:
            if not self._is_locked:
                return False

            self._deactivate_lock()
            return True

    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the lock.

        Returns:
            Dict with lock status information
        """
        with self._lock:
            status = {
                "component_name": self.component_name,
                "is_locked": self._is_locked,
                "failure_count": self._failure_count,
                "lock_threshold": self.lock_threshold,
                "current_lock_duration": self._current_lock_duration,
            }

            if self._is_locked:
                status["lock_expiry"] = self._lock_expiry.isoformat()
                status["remaining_seconds"] = max(
                    0, (self._lock_expiry - datetime.now()).total_seconds()
                )

            if self._last_success:
                status["last_success"] = self._last_success.isoformat()
                seconds_since_success = (
                    datetime.now() - self._last_success
                ).total_seconds()
                status["seconds_since_success"] = seconds_since_success
                status["in_cooldown"] = seconds_since_success <= self.cooldown_duration

            return status

    def get_lock_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get lock history.

        Args:
            limit: Maximum number of history entries to return

        Returns:
            List of lock history events
        """
        with self._lock:
            return self._lock_history[-limit:] if self._lock_history else []

    def _activate_lock(self) -> None:
        """Activate the lock for the current duration."""
        self._is_locked = True
        self._lock_expiry = datetime.now() + timedelta(
            seconds=self._current_lock_duration
        )

        # Record in history
        self._lock_history.append(
            {
                "event": "locked",
                "timestamp": datetime.now().isoformat(),
                "duration": self._current_lock_duration,
                "expiry": self._lock_expiry.isoformat(),
                "failure_count": self._failure_count,
            }
        )

        # Keep history size reasonable
        if len(self._lock_history) > 100:
            self._lock_history = self._lock_history[-100:]

        logger.warning(
            f"Component {self.component_name} locked for {self._current_lock_duration:.1f}s "
            f"after {self._failure_count} consecutive failures"
        )

    def _deactivate_lock(self) -> None:
        """Deactivate the lock."""
        was_locked = self._is_locked
        self._is_locked = False
        self._failure_count = 0

        # Only record in history if actually unlocked
        if was_locked:
            self._lock_history.append(
                {
                    "event": "unlocked",
                    "timestamp": datetime.now().isoformat(),
                    "reason": (
                        "manual" if datetime.now() < self._lock_expiry else "expired"
                    ),
                }
            )

            logger.info(f"Component {self.component_name} unlocked")

    def _extend_lock(self) -> None:
        """Extend lock duration for severe errors."""
        if not self._is_locked:
            return

        # Increase lock duration, but don't exceed maximum
        self._current_lock_duration = min(
            self._current_lock_duration * self.escalation_factor, self.max_lock_duration
        )

        # Update expiry
        self._lock_expiry = datetime.now() + timedelta(
            seconds=self._current_lock_duration
        )

        # Record in history
        self._lock_history.append(
            {
                "event": "extended",
                "timestamp": datetime.now().isoformat(),
                "duration": self._current_lock_duration,
                "expiry": self._lock_expiry.isoformat(),
            }
        )

        logger.warning(
            f"Lock extended for component {self.component_name} "
            f"to {self._current_lock_duration:.1f}s"
        )

    def _is_severe_error(self, error: Exception) -> bool:
        """
        Determine if an error is severe enough to extend a lock.

        Args:
            error: The exception to evaluate

        Returns:
            True if the error is severe, False otherwise
        """
        # Consider certain exception types as severe
        severe_types = (ConnectionError, TimeoutError, MemoryError, RuntimeError)

        return isinstance(error, severe_types)


class LockManager:
    """
    Manager for error locks across the system.

    This class provides centralized management of error locks for all components,
    including creating, checking, and releasing locks as needed.
    """

    def __init__(self, default_threshold: int = 3, default_duration: float = 300.0):
        """
        Initialize the lock manager.

        Args:
            default_threshold: Default number of failures before locking
            default_duration: Default lock duration in seconds
        """
        self.default_threshold = default_threshold
        self.default_duration = default_duration

        self._locks: Dict[str, ErrorLock] = {}
        self._lock = threading.RLock()

    def create_lock(
        self,
        component_name: str,
        lock_threshold: Optional[int] = None,
        lock_duration: Optional[float] = None,
        cooldown_duration: float = 60.0,
        escalation_factor: float = 2.0,
        max_lock_duration: float = 3600.0,
    ) -> ErrorLock:
        """
        Create a new error lock for a component.

        Args:
            component_name: Name of the component to lock
            lock_threshold: Number of consecutive failures before locking
            lock_duration: Initial duration of lock in seconds
            cooldown_duration: Time to wait after success before resetting failure count
            escalation_factor: Factor to multiply lock duration by on repeated locks
            max_lock_duration: Maximum lock duration in seconds

        Returns:
            The created error lock
        """
        with self._lock:
            # Use default values if not specified
            if lock_threshold is None:
                lock_threshold = self.default_threshold
            if lock_duration is None:
                lock_duration = self.default_duration

            # Create new lock
            error_lock = ErrorLock(
                component_name=component_name,
                lock_threshold=lock_threshold,
                lock_duration=lock_duration,
                cooldown_duration=cooldown_duration,
                escalation_factor=escalation_factor,
                max_lock_duration=max_lock_duration,
            )

            # Store and return
            self._locks[component_name] = error_lock
            return error_lock

    def get_lock(self, component_name: str) -> Optional[ErrorLock]:
        """
        Get the error lock for a component.

        Args:
            component_name: Name of the component

        Returns:
            The error lock for the component, or None if not found
        """
        with self._lock:
            return self._locks.get(component_name)

    def record_failure(
        self,
        component_name: str,
        error: Optional[Exception] = None,
        create_if_missing: bool = True,
    ) -> bool:
        """
        Record a failure for a component.

        Args:
            component_name: Name of the component
            error: Optional exception that caused the failure
            create_if_missing: Whether to create a lock if one doesn't exist

        Returns:
            True if component is now locked, False otherwise
        """
        with self._lock:
            # Get or create lock
            lock = self._locks.get(component_name)
            if lock is None:
                if create_if_missing:
                    lock = self.create_lock(component_name)
                else:
                    return False

            # Record failure
            return lock.record_failure(error)

    def record_success(self, component_name: str) -> None:
        """
        Record a successful operation for a component.

        Args:
            component_name: Name of the component
        """
        with self._lock:
            lock = self._locks.get(component_name)
            if lock:
                lock.record_success()

    def is_locked(self, component_name: str) -> bool:
        """
        Check if a component is locked.

        Args:
            component_name: Name of the component

        Returns:
            True if component is locked, False otherwise
        """
        with self._lock:
            lock = self._locks.get(component_name)
            return lock.is_locked() if lock else False

    def release(self, component_name: str) -> bool:
        """
        Release a lock for a component.

        Args:
            component_name: Name of the component

        Returns:
            True if lock was released, False otherwise
        """
        with self._lock:
            lock = self._locks.get(component_name)
            return lock.release() if lock else False

    def get_all_locks(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all locks.

        Returns:
            Dict mapping component names to lock statuses
        """
        with self._lock:
            return {name: lock.get_status() for name, lock in self._locks.items()}

    def get_active_locks(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all currently active locks.

        Returns:
            Dict mapping component names to lock statuses for locked components
        """
        with self._lock:
            return {
                name: lock.get_status()
                for name, lock in self._locks.items()
                if lock.is_locked()
            }

    def reset_all_locks(self) -> int:
        """
        Reset all locks in the system.

        Returns:
            Number of locks that were reset
        """
        with self._lock:
            reset_count = 0
            for lock in self._locks.values():
                if lock.is_locked():
                    lock.release()
                    reset_count += 1
            return reset_count

"""
Health monitoring system.

This module provides continuous monitoring of system health,
including scheduled health checks, status reporting, and notifications.
"""

import logging
import time
import threading
import json
import os
from typing import Dict, List, Any, Optional, Set, Callable
from datetime import datetime, timedelta

from forex_ai.health.base import HealthCheck, HealthStatus, ComponentHealth
from forex_ai.health.system import SystemHealthCheck

# Configure logging
logger = logging.getLogger(__name__)


class HealthMonitor:
    """
    System health monitor that continuously checks component health.

    This class provides a central monitoring system that:
    1. Manages and schedules health checks for all components
    2. Maintains a history of health status changes
    3. Provides health status reporting capabilities
    4. Supports notification callbacks for health status changes
    """

    def __init__(
        self,
        check_interval: float = 60.0,
        history_limit: int = 1000,
        notification_callbacks: Optional[List[Callable]] = None,
        report_directory: Optional[str] = None,
    ):
        """
        Initialize the health monitor.

        Args:
            check_interval: How often to run the monitoring cycle (in seconds)
            history_limit: Maximum number of health status entries to keep
            notification_callbacks: List of functions to call when health status changes
            report_directory: Directory to save health reports
        """
        self.check_interval = check_interval
        self.history_limit = history_limit
        self.notification_callbacks = notification_callbacks or []
        self.report_directory = report_directory

        # Create report directory if specified
        if self.report_directory and not os.path.exists(self.report_directory):
            os.makedirs(self.report_directory)

        # Component health checks
        self.health_checks: Dict[str, HealthCheck] = {}

        # System health check (manages components)
        self.system_check = SystemHealthCheck()

        # Monitor state
        self.running = False
        self.monitor_thread = None
        self.last_status: Dict[str, HealthStatus] = {}
        self.health_history: List[Dict[str, Any]] = []

        logger.info("Health monitor initialized")

    def add_check(self, check: HealthCheck) -> None:
        """
        Add a health check to be monitored.

        Args:
            check: The health check to add
        """
        if check.name in self.health_checks:
            logger.warning(f"Health check {check.name} already exists, replacing")

        self.health_checks[check.name] = check
        self.system_check.add_component_check(check)

        # Initialize status history
        self.last_status[check.name] = HealthStatus.UNKNOWN

        logger.info(f"Added health check: {check.name}")

    def remove_check(self, check_name: str) -> bool:
        """
        Remove a health check by name.

        Args:
            check_name: Name of the health check to remove

        Returns:
            True if check was found and removed, False otherwise
        """
        if check_name in self.health_checks:
            check = self.health_checks.pop(check_name)
            self.system_check.remove_component_check(check_name)
            self.last_status.pop(check_name, None)
            logger.info(f"Removed health check: {check_name}")
            return True
        return False

    def start(self) -> bool:
        """
        Start the health monitoring thread.

        Returns:
            True if started successfully, False otherwise
        """
        if self.running:
            logger.warning("Health monitor already running")
            return False

        self.running = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True, name="health-monitor"
        )
        self.monitor_thread.start()

        logger.info("Health monitor started")
        return True

    def stop(self) -> bool:
        """
        Stop the health monitoring thread.

        Returns:
            True if stopped successfully, False otherwise
        """
        if not self.running:
            logger.warning("Health monitor not running")
            return False

        self.running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)

        logger.info("Health monitor stopped")
        return True

    def _monitoring_loop(self) -> None:
        """Main monitoring loop that runs health checks at specified interval."""
        logger.info("Health monitoring loop started")

        while self.running:
            try:
                # Run all health checks
                self._run_health_checks()

                # Generate and save health report
                if self.report_directory:
                    self._save_health_report()

                # Sleep until next check
                for _ in range(
                    int(self.check_interval * 2)
                ):  # Check for stop condition twice per second
                    if not self.running:
                        break
                    time.sleep(0.5)

            except Exception as e:
                logger.error(f"Error in health monitoring loop: {str(e)}")
                time.sleep(5.0)  # Sleep on error and try again

        logger.info("Health monitoring loop stopped")

    def _run_health_checks(self) -> None:
        """Run all health checks and process results."""
        start_time = time.time()
        logger.debug("Running health checks")

        # Run individual component checks
        for name, check in self.health_checks.items():
            if check.should_check():
                try:
                    result = check.check_health()
                    self._process_health_result(name, result)
                except Exception as e:
                    logger.error(f"Error running health check {name}: {str(e)}")

        # Run system health check
        try:
            result = self.system_check.check_health()
            self._process_health_result("system", result)
        except Exception as e:
            logger.error(f"Error running system health check: {str(e)}")

        check_duration = time.time() - start_time
        logger.debug(f"Health checks completed in {check_duration:.2f}s")

    def _process_health_result(self, name: str, result: ComponentHealth) -> None:
        """
        Process a health check result.

        Args:
            name: Name of the health check
            result: Health check result
        """
        # Check if status changed
        prev_status = self.last_status.get(name, HealthStatus.UNKNOWN)
        curr_status = result.status

        if prev_status != curr_status:
            # Status changed, record in history
            self._record_status_change(name, prev_status, curr_status, result.message)

            # Notify if needed
            self._send_notifications(name, prev_status, curr_status, result)

            # Update last status
            self.last_status[name] = curr_status

    def _record_status_change(
        self,
        name: str,
        prev_status: HealthStatus,
        curr_status: HealthStatus,
        message: str,
    ) -> None:
        """
        Record a health status change in history.

        Args:
            name: Name of the health check
            prev_status: Previous health status
            curr_status: Current health status
            message: Status change message
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "component": name,
            "from_status": prev_status.value,
            "to_status": curr_status.value,
            "message": message,
        }

        self.health_history.append(entry)

        # Log status change
        log_level = logging.WARNING if not curr_status.is_operational else logging.INFO
        logger.log(
            log_level,
            f"Health status changed for {name}: {prev_status.value} -> {curr_status.value}: {message}",
        )

        # Keep history within limit
        if len(self.health_history) > self.history_limit:
            self.health_history = self.health_history[-self.history_limit :]

    def _send_notifications(
        self,
        name: str,
        prev_status: HealthStatus,
        curr_status: HealthStatus,
        result: ComponentHealth,
    ) -> None:
        """
        Send notifications for health status changes.

        Args:
            name: Name of the health check
            prev_status: Previous health status
            curr_status: Current health status
            result: Health check result
        """
        # Only notify on significant changes
        significant_change = (
            # Healthy -> Unhealthy
            (prev_status.is_operational and not curr_status.is_operational)
            or
            # Unhealthy -> Healthy
            (not prev_status.is_operational and curr_status.is_operational)
            or
            # First status for a component
            (
                prev_status == HealthStatus.UNKNOWN
                and curr_status != HealthStatus.UNKNOWN
            )
        )

        if significant_change:
            for callback in self.notification_callbacks:
                try:
                    callback(name, prev_status, curr_status, result)
                except Exception as e:
                    logger.error(f"Error in notification callback: {str(e)}")

    def _save_health_report(self) -> None:
        """Generate and save a health report to the report directory."""
        if not self.report_directory:
            return

        try:
            # Generate report data
            report = self.get_health_report()

            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(
                self.report_directory, f"health_report_{timestamp}.json"
            )

            # Save report
            with open(filename, "w") as f:
                json.dump(report, f, indent=2)

            # Create/update latest report symlink
            latest_filename = os.path.join(
                self.report_directory, "health_report_latest.json"
            )
            try:
                if os.path.exists(latest_filename):
                    os.remove(latest_filename)
                os.symlink(filename, latest_filename)
            except Exception as e:
                logger.warning(f"Could not create symlink to latest report: {str(e)}")
                # On Windows, symlinks might not work, just copy the file
                try:
                    import shutil

                    shutil.copy2(filename, latest_filename)
                except Exception as copy_err:
                    logger.error(f"Could not copy latest report: {str(copy_err)}")

            logger.debug(f"Health report saved to {filename}")

        except Exception as e:
            logger.error(f"Error saving health report: {str(e)}")

    def get_health_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive health report.

        Returns:
            Dict with health report data
        """
        # Get system status summary
        system_summary = self.system_check.get_status_summary()

        # Collect component statuses
        component_statuses = {}
        for name, check in self.health_checks.items():
            component_statuses[name] = check.health.to_dict()

        # Count status types
        status_counts = {
            "healthy": 0,
            "degraded": 0,
            "unhealthy": 0,
            "unknown": 0,
            "locked": 0,
            "recovering": 0,
        }

        for status in self.last_status.values():
            status_lower = status.value.lower()
            if status_lower in status_counts:
                status_counts[status_lower] += 1

        # Collect recent history
        recent_history = self.health_history[-20:] if self.health_history else []

        return {
            "timestamp": datetime.now().isoformat(),
            "system_status": system_summary["status"],
            "system_message": system_summary["message"],
            "status_counts": status_counts,
            "component_statuses": component_statuses,
            "recent_history": recent_history,
            "system_info": system_summary["system_info"],
            "uptime_seconds": system_summary["uptime_seconds"],
        }

    def get_component_history(
        self, component_name: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get health history for a specific component.

        Args:
            component_name: Name of the component
            limit: Maximum number of history entries to return

        Returns:
            List of history entries for the component
        """
        # Filter history for the component
        component_history = [
            entry
            for entry in self.health_history
            if entry["component"] == component_name
        ]

        # Return most recent entries up to limit
        return component_history[-limit:] if component_history else []

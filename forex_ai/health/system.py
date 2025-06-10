"""
System-wide health check functionality.

This module provides system-level health checking capabilities, aggregating
the health status of all components and subsystems.
"""

import os
import psutil
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import platform
import time

from forex_ai.health.base import HealthCheck, HealthStatus, ComponentHealth

# Configure logging
logger = logging.getLogger(__name__)


class SystemHealthCheck(HealthCheck):
    """
    System-wide health check that aggregates component checks.

    This class provides a holistic view of system health by:
    1. Aggregating the status of all registered component health checks
    2. Monitoring system resources (CPU, memory, disk)
    3. Tracking overall system performance metrics
    """

    def __init__(
        self,
        component_checks: Optional[List[HealthCheck]] = None,
        check_interval: float = 30.0,
        resource_thresholds: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the system health check.

        Args:
            component_checks: List of component health checks to monitor
            check_interval: How often to run the check (in seconds)
            resource_thresholds: Dict with resource thresholds (cpu_percent, memory_percent, disk_percent)
        """
        super().__init__(
            name="forex_ai_system", check_interval=check_interval, timeout=30.0
        )

        self.component_checks = component_checks or []
        self.resource_thresholds = resource_thresholds or {
            "cpu_percent": 85.0,  # 85% CPU usage is concerning
            "memory_percent": 80.0,  # 80% memory usage is concerning
            "disk_percent": 90.0,  # 90% disk usage is concerning
        }

        # Collect system info at initialization
        self.system_info = self._get_system_info()

        # Track start time for uptime calculation
        self.start_time = time.time()

    def add_component_check(self, check: HealthCheck) -> None:
        """
        Add a component health check to be monitored.

        Args:
            check: The component health check to add
        """
        if check not in self.component_checks:
            self.component_checks.append(check)

    def remove_component_check(self, check_name: str) -> bool:
        """
        Remove a component health check by name.

        Args:
            check_name: Name of the check to remove

        Returns:
            True if check was found and removed, False otherwise
        """
        for i, check in enumerate(self.component_checks):
            if check.name == check_name:
                self.component_checks.pop(i)
                return True
        return False

    def _check_implementation(self) -> Dict[str, Any]:
        """
        Implement the system health check.

        This health check:
        1. Checks all registered component health
        2. Monitors system resources
        3. Calculates overall system health status

        Returns:
            Dict with health check results
        """
        # Check system resources
        resource_metrics = self._check_system_resources()
        resource_status = self._evaluate_resources(resource_metrics)

        # Check component health
        component_status = self._check_components()

        # Compile overall health
        if (
            resource_status["status"] == "failure"
            or component_status["status"] == "failure"
        ):
            status = "failure"
            is_degraded = resource_status.get(
                "is_degraded", False
            ) and component_status.get("is_degraded", False)
            message = "System health check failed"
        else:
            status = "success"
            is_degraded = False
            message = "System is healthy"

        # Calculate uptime
        uptime_seconds = time.time() - self.start_time

        # Compile metrics
        metrics = {
            "uptime_seconds": uptime_seconds,
            "component_count": len(self.component_checks),
            "healthy_component_count": component_status["healthy_count"],
            "unhealthy_component_count": component_status["unhealthy_count"],
            "degraded_component_count": component_status["degraded_count"],
        }
        metrics.update(resource_metrics)

        return {
            "status": status,
            "message": message,
            "is_degraded": is_degraded,
            "details": {
                "system_info": self.system_info,
                "resource_status": resource_status,
                "component_status": component_status,
            },
            "metrics": metrics,
        }

    def _get_system_info(self) -> Dict[str, Any]:
        """
        Collect basic system information.

        Returns:
            Dict with system information
        """
        info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
            "hostname": platform.node(),
        }

        # Add more system info
        try:
            memory_info = psutil.virtual_memory()
            disk_info = psutil.disk_usage("/")

            info.update(
                {
                    "cpu_count": psutil.cpu_count(),
                    "total_memory_gb": round(memory_info.total / (1024**3), 2),
                    "total_disk_gb": round(disk_info.total / (1024**3), 2),
                }
            )
        except Exception as e:
            logger.warning(f"Could not collect detailed system info: {str(e)}")

        return info

    def _check_system_resources(self) -> Dict[str, float]:
        """
        Check system resource usage.

        Returns:
            Dict with resource metrics
        """
        metrics = {}

        try:
            # CPU usage
            metrics["cpu_percent"] = psutil.cpu_percent(interval=0.5)

            # Memory usage
            memory = psutil.virtual_memory()
            metrics["memory_percent"] = memory.percent
            metrics["memory_used_gb"] = round(memory.used / (1024**3), 2)
            metrics["memory_available_gb"] = round(memory.available / (1024**3), 2)

            # Disk usage
            disk = psutil.disk_usage("/")
            metrics["disk_percent"] = disk.percent
            metrics["disk_used_gb"] = round(disk.used / (1024**3), 2)
            metrics["disk_free_gb"] = round(disk.free / (1024**3), 2)

        except Exception as e:
            logger.error(f"Error checking system resources: {str(e)}")
            # Set some default values to avoid KeyErrors
            metrics.update(
                {"cpu_percent": 0.0, "memory_percent": 0.0, "disk_percent": 0.0}
            )

        return metrics

    def _evaluate_resources(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Evaluate system resources against thresholds.

        Args:
            metrics: Dict with resource metrics

        Returns:
            Dict with resource status evaluation
        """
        issues = []
        warnings = []

        # Check CPU
        if metrics.get("cpu_percent", 0) > self.resource_thresholds["cpu_percent"]:
            issues.append(
                f"CPU usage at {metrics['cpu_percent']}% exceeds threshold of {self.resource_thresholds['cpu_percent']}%"
            )
        elif metrics.get("cpu_percent", 0) > (
            self.resource_thresholds["cpu_percent"] * 0.8
        ):
            warnings.append(
                f"CPU usage at {metrics['cpu_percent']}% approaching threshold"
            )

        # Check memory
        if (
            metrics.get("memory_percent", 0)
            > self.resource_thresholds["memory_percent"]
        ):
            issues.append(
                f"Memory usage at {metrics['memory_percent']}% exceeds threshold of {self.resource_thresholds['memory_percent']}%"
            )
        elif metrics.get("memory_percent", 0) > (
            self.resource_thresholds["memory_percent"] * 0.8
        ):
            warnings.append(
                f"Memory usage at {metrics['memory_percent']}% approaching threshold"
            )

        # Check disk
        if metrics.get("disk_percent", 0) > self.resource_thresholds["disk_percent"]:
            issues.append(
                f"Disk usage at {metrics['disk_percent']}% exceeds threshold of {self.resource_thresholds['disk_percent']}%"
            )
        elif metrics.get("disk_percent", 0) > (
            self.resource_thresholds["disk_percent"] * 0.8
        ):
            warnings.append(
                f"Disk usage at {metrics['disk_percent']}% approaching threshold"
            )

        # Determine overall status
        if issues:
            return {
                "status": "failure",
                "is_degraded": True,  # Resource issues cause degraded status, not complete failure
                "message": f"Resource issues detected: {'; '.join(issues)}",
                "issues": issues,
                "warnings": warnings,
            }
        elif warnings:
            return {
                "status": "success",
                "is_degraded": False,
                "message": f"Resources OK but approaching limits: {'; '.join(warnings)}",
                "issues": [],
                "warnings": warnings,
            }
        else:
            return {
                "status": "success",
                "is_degraded": False,
                "message": "All system resources within acceptable limits",
                "issues": [],
                "warnings": [],
            }

    def _check_components(self) -> Dict[str, Any]:
        """
        Check the health of all registered components.

        Returns:
            Dict with component health status
        """
        # Counters for component status
        healthy_count = 0
        degraded_count = 0
        unhealthy_count = 0
        unknown_count = 0

        component_results = []
        issues = []

        # Check each component
        for check in self.component_checks:
            try:
                # Run the component health check
                health = check.check_health()

                # Track status counts
                if health.status == HealthStatus.HEALTHY:
                    healthy_count += 1
                elif health.status == HealthStatus.DEGRADED:
                    degraded_count += 1
                    issues.append(f"{check.name} is degraded: {health.message}")
                elif health.status == HealthStatus.UNHEALTHY:
                    unhealthy_count += 1
                    issues.append(f"{check.name} is unhealthy: {health.message}")
                else:
                    unknown_count += 1

                # Add to results
                component_results.append(health.to_dict())

            except Exception as e:
                logger.error(f"Error checking component {check.name}: {str(e)}")
                unhealthy_count += 1
                issues.append(f"Error checking {check.name}: {str(e)}")

        # Determine overall status
        if unhealthy_count > 0:
            status = "failure"
            is_degraded = False
            message = (
                f"Component health check failed: {unhealthy_count} unhealthy components"
            )
        elif degraded_count > 0:
            status = "failure"
            is_degraded = True
            message = (
                f"Component health check degraded: {degraded_count} degraded components"
            )
        else:
            status = "success"
            is_degraded = False
            message = f"All components healthy: {healthy_count} components OK"

        return {
            "status": status,
            "is_degraded": is_degraded,
            "message": message,
            "healthy_count": healthy_count,
            "degraded_count": degraded_count,
            "unhealthy_count": unhealthy_count,
            "unknown_count": unknown_count,
            "issues": issues,
            "components": component_results,
        }

    def get_status_summary(self) -> Dict[str, Any]:
        """
        Get a summary of system health status.

        Returns:
            Dict with system health summary
        """
        # Run health check if needed
        if self.should_check():
            self.check_health()

        # Count component statuses
        component_counts = {
            "total": len(self.component_checks),
            "healthy": 0,
            "degraded": 0,
            "unhealthy": 0,
            "unknown": 0,
            "locked": 0,
        }

        for check in self.component_checks:
            if check.health.status == HealthStatus.HEALTHY:
                component_counts["healthy"] += 1
            elif check.health.status == HealthStatus.DEGRADED:
                component_counts["degraded"] += 1
            elif check.health.status == HealthStatus.UNHEALTHY:
                component_counts["unhealthy"] += 1
            elif check.health.status == HealthStatus.LOCKED:
                component_counts["locked"] += 1
            else:
                component_counts["unknown"] += 1

        # Determine overall status
        if component_counts["unhealthy"] > 0:
            status = HealthStatus.UNHEALTHY
            message = f"{component_counts['unhealthy']} components unhealthy"
        elif component_counts["degraded"] > 0:
            status = HealthStatus.DEGRADED
            message = f"{component_counts['degraded']} components degraded"
        elif component_counts["unknown"] > component_counts["total"] / 2:
            status = HealthStatus.UNKNOWN
            message = f"{component_counts['unknown']} components in unknown state"
        else:
            status = HealthStatus.HEALTHY
            message = "System healthy"

        return {
            "status": status.value,
            "message": message,
            "component_counts": component_counts,
            "uptime_seconds": time.time() - self.start_time,
            "last_checked": self.health.last_checked.isoformat(),
            "system_info": self.system_info,
        }

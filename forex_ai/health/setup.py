"""
Health check system setup module.

This module provides functions for initializing the health check system
in a Flask application.
"""

import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from forex_ai.health.base import HealthCheck, HealthStatus, ComponentHealth
from forex_ai.health.system import SystemHealthCheck
from forex_ai.health.component import (
    DatabaseHealthCheck,
    APIHealthCheck,
    ModelHealthCheck,
    AgentHealthCheck,
)
from forex_ai.health.monitor import HealthMonitor
from forex_ai.health.locks import ErrorLock, LockManager

# Configure logging
logger = logging.getLogger(__name__)


def setup_health_system(app, config=None):
    """
    Set up the health check system in a Flask application.

    Args:
        app: Flask application
        config: Health check configuration

    Returns:
        HealthMonitor: The configured health monitor
    """
    # Get or create configuration
    health_config = config or {}

    # Create report directory if specified
    report_dir = health_config.get("report_directory")
    if report_dir and not os.path.exists(report_dir):
        os.makedirs(report_dir, exist_ok=True)

    # Create lock manager
    lock_manager = LockManager(
        default_threshold=health_config.get("default_threshold", 3),
        default_duration=health_config.get("default_duration", 300.0),
    )

    # Create notification callbacks
    notification_callbacks = []

    # Add logging callback
    if health_config.get("log_notifications", True):

        def log_notification(name, prev_status, curr_status, result):
            if not curr_status.is_operational and prev_status.is_operational:
                # Component became unhealthy
                logger.warning(
                    f"HEALTH ALERT: {name} changed from {prev_status.value} to {curr_status.value}: {result.message}"
                )
            elif curr_status.is_operational and not prev_status.is_operational:
                # Component recovered
                logger.info(
                    f"HEALTH RECOVERY: {name} changed from {prev_status.value} to {curr_status.value}: {result.message}"
                )

        notification_callbacks.append(log_notification)

    # Create health monitor
    monitor = HealthMonitor(
        check_interval=health_config.get("check_interval", 60.0),
        history_limit=health_config.get("history_limit", 1000),
        notification_callbacks=notification_callbacks,
        report_directory=report_dir,
    )

    # Create and add component checks
    checks = create_component_checks(health_config.get("components", {}))
    for check in checks:
        monitor.add_check(check)

    # Store monitor and lock manager in app context
    app.health_monitor = monitor
    app.lock_manager = lock_manager

    # Start monitor if auto_start is enabled
    if health_config.get("auto_start", True):
        monitor.start()
        logger.info("Health monitor started")

    return monitor


def create_component_checks(components_config) -> List[HealthCheck]:
    """
    Create health checks for components based on configuration.

    Args:
        components_config: Dictionary with component configurations

    Returns:
        List of health check instances
    """
    checks = []

    # Create database checks
    for name, config in components_config.get("database", {}).items():
        try:
            check = DatabaseHealthCheck(
                db_module_path=config["module_path"],
                connection_func_name=config.get("connection_func", "get_connection"),
                test_query=config.get("test_query", "SELECT 1"),
                check_interval=config.get("check_interval", 60.0),
                timeout=config.get("timeout", 5.0),
            )
            checks.append(check)
            logger.info(f"Created database health check: {name}")
        except KeyError as e:
            logger.error(
                f"Missing required configuration for database check {name}: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Error creating database health check {name}: {str(e)}")

    # Create API checks
    for name, config in components_config.get("api", {}).items():
        try:
            check = APIHealthCheck(
                name=name,
                endpoint_url=config["endpoint_url"],
                method=config.get("method", "GET"),
                headers=config.get("headers"),
                params=config.get("params"),
                test_response_key=config.get("test_response_key"),
                expected_status_code=config.get("expected_status_code", 200),
                check_interval=config.get("check_interval", 300.0),
                timeout=config.get("timeout", 30.0),
            )
            checks.append(check)
            logger.info(f"Created API health check: {name}")
        except KeyError as e:
            logger.error(
                f"Missing required configuration for API check {name}: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Error creating API health check {name}: {str(e)}")

    # Create model checks
    for name, config in components_config.get("model", {}).items():
        try:
            check = ModelHealthCheck(
                name=name,
                model_module_path=config["module_path"],
                model_class_name=config["class_name"],
                test_input=config["test_input"],
                check_interval=config.get("check_interval", 300.0),
                timeout=config.get("timeout", 30.0),
            )
            checks.append(check)
            logger.info(f"Created model health check: {name}")
        except KeyError as e:
            logger.error(
                f"Missing required configuration for model check {name}: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Error creating model health check {name}: {str(e)}")

    # Create agent checks
    for name, config in components_config.get("agent", {}).items():
        try:
            check = AgentHealthCheck(
                name=name,
                agent_module_path=config["module_path"],
                agent_class_name=config["class_name"],
                test_input=config["test_input"],
                expected_output_key=config.get("expected_output_key"),
                check_interval=config.get("check_interval", 300.0),
                timeout=config.get("timeout", 30.0),
            )
            checks.append(check)
            logger.info(f"Created agent health check: {name}")
        except KeyError as e:
            logger.error(
                f"Missing required configuration for agent check {name}: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Error creating agent health check {name}: {str(e)}")

    return checks


def create_recovery_function(component_type, component_name, recovery_config):
    """
    Create a recovery function for a component.

    Args:
        component_type: Type of component (database, api, model, agent)
        component_name: Name of the component
        recovery_config: Recovery configuration

    Returns:
        Function that attempts to recover the component
    """
    recovery_type = recovery_config.get("type", "restart")

    if recovery_type == "restart":
        # Create restart recovery function
        def restart_recovery():
            logger.info(
                f"Attempting to restart {component_type} component: {component_name}"
            )
            try:
                # Implement restart logic based on component type
                # This is a placeholder - actual implementation would depend on the component
                return True
            except Exception as e:
                logger.error(
                    f"Failed to restart {component_type} component {component_name}: {str(e)}"
                )
                return False

        return restart_recovery

    elif recovery_type == "reconnect":
        # Create reconnect recovery function (for databases, APIs)
        def reconnect_recovery():
            logger.info(
                f"Attempting to reconnect to {component_type}: {component_name}"
            )
            try:
                # Implement reconnect logic based on component type
                # This is a placeholder - actual implementation would depend on the component
                return True
            except Exception as e:
                logger.error(
                    f"Failed to reconnect to {component_type} {component_name}: {str(e)}"
                )
                return False

        return reconnect_recovery

    elif recovery_type == "reload":
        # Create reload recovery function (for models)
        def reload_recovery():
            logger.info(f"Attempting to reload {component_type}: {component_name}")
            try:
                # Implement reload logic based on component type
                # This is a placeholder - actual implementation would depend on the component
                return True
            except Exception as e:
                logger.error(
                    f"Failed to reload {component_type} {component_name}: {str(e)}"
                )
                return False

        return reload_recovery

    elif recovery_type == "custom" and "function" in recovery_config:
        # Return custom recovery function
        return recovery_config["function"]

    else:
        # Default no-op recovery function
        def noop_recovery():
            logger.warning(
                f"No recovery strategy defined for {component_type}: {component_name}"
            )
            return False

        return noop_recovery


def stop_health_system(app):
    """
    Stop the health check system in a Flask application.

    Args:
        app: Flask application
    """
    # Get health monitor from app context
    monitor = getattr(app, "health_monitor", None)

    if monitor:
        # Stop monitor
        monitor.stop()
        logger.info("Health monitor stopped")
    else:
        logger.warning("Health monitor not found in app context")

    # Clear health monitor and lock manager from app context
    app.health_monitor = None
    app.lock_manager = None

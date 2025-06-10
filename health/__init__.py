"""
Health check system for the Forex AI Trading System.

This module provides comprehensive health checking functionality for all components
of the Forex AI system, including:

- Individual component health checks
- System-wide health status reporting
- Error locking to prevent cascade failures
- Automated recovery mechanisms
- Health metrics collection
"""

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

__all__ = [
    "HealthCheck",
    "HealthStatus",
    "ComponentHealth",
    "SystemHealthCheck",
    "DatabaseHealthCheck",
    "APIHealthCheck",
    "ModelHealthCheck",
    "AgentHealthCheck",
    "HealthMonitor",
    "ErrorLock",
    "LockManager",
]

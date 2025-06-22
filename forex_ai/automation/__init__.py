"""
Automation module for the Forex AI Trading System.

This module provides a custom workflow engine to replace N8N for task automation.
"""

from forex_ai.automation.engine import (
    TaskStatus,
    TaskDefinition,
    TaskResult,
    Workflow,
    WorkflowEngine,
    get_workflow_engine,
)

__all__ = [
    'TaskStatus',
    'TaskDefinition',
    'TaskResult',
    'Workflow',
    'WorkflowEngine',
    'get_workflow_engine',
]

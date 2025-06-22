"""
Custom Workflow Engine for Forex AI Trading System

This module provides a custom workflow engine to replace N8N for task automation.
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class TaskStatus(str, Enum):
    """Task status enum."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskDefinition(BaseModel):
    """Task definition model."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    function: Callable
    args: List[Any] = Field(default_factory=list)
    kwargs: Dict[str, Any] = Field(default_factory=dict)
    schedule: Optional[str] = None  # Cron-like schedule
    depends_on: List[str] = Field(default_factory=list)  # List of task IDs this task depends on
    retry_count: int = 0
    max_retries: int = 3
    timeout: int = 60  # Timeout in seconds
    created_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        arbitrary_types_allowed = True

class TaskResult(BaseModel):
    """Task result model."""
    task_id: str
    status: TaskStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[float] = None  # Duration in seconds

class Workflow(BaseModel):
    """Workflow model."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    tasks: List[TaskDefinition] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        arbitrary_types_allowed = True

class WorkflowEngine:
    """
    Custom workflow engine to replace N8N for task automation.
    
    This engine provides:
    - Task scheduling and execution
    - Dependency management
    - Retry logic
    - Timeout handling
    """
    
    def __init__(self):
        """Initialize the workflow engine."""
        self.workflows: Dict[str, Workflow] = {}
        self.task_results: Dict[str, TaskResult] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.scheduled_tasks: Dict[str, asyncio.Task] = {}
        self.loop = asyncio.get_event_loop()
        logger.info("Workflow engine initialized")
    
    def create_workflow(self, name: str, description: Optional[str] = None) -> Workflow:
        """
        Create a new workflow.
        
        Args:
            name: The name of the workflow.
            description: Optional description of the workflow.
            
        Returns:
            The created workflow.
        """
        workflow = Workflow(name=name, description=description)
        self.workflows[workflow.id] = workflow
        logger.info(f"Created workflow: {name} (ID: {workflow.id})")
        return workflow
    
    def add_task(
        self,
        workflow_id: str,
        name: str,
        function: Callable,
        args: List[Any] = None,
        kwargs: Dict[str, Any] = None,
        schedule: Optional[str] = None,
        depends_on: List[str] = None,
        max_retries: int = 3,
        timeout: int = 60,
    ) -> TaskDefinition:
        """
        Add a task to a workflow.
        
        Args:
            workflow_id: The ID of the workflow to add the task to.
            name: The name of the task.
            function: The function to execute.
            args: List of positional arguments to pass to the function.
            kwargs: Dictionary of keyword arguments to pass to the function.
            schedule: Cron-like schedule for recurring tasks.
            depends_on: List of task IDs this task depends on.
            max_retries: Maximum number of retries if the task fails.
            timeout: Timeout in seconds.
            
        Returns:
            The created task definition.
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow with ID {workflow_id} not found")
        
        args = args or []
        kwargs = kwargs or {}
        depends_on = depends_on or []
        
        task = TaskDefinition(
            name=name,
            function=function,
            args=args,
            kwargs=kwargs,
            schedule=schedule,
            depends_on=depends_on,
            max_retries=max_retries,
            timeout=timeout,
        )
        
        self.workflows[workflow_id].tasks.append(task)
        logger.info(f"Added task \"{name}\" to workflow {workflow_id}")
        return task
    
    async def execute_task(self, task: TaskDefinition) -> TaskResult:
        """
        Execute a task.
        
        Args:
            task: The task definition to execute.
            
        Returns:
            The task result.
        """
        task_result = TaskResult(
            task_id=task.id,
            status=TaskStatus.RUNNING,
            start_time=datetime.now(),
        )
        self.task_results[task.id] = task_result
        
        try:
            # Set a timeout for the task
            result = await asyncio.wait_for(
                self._execute_function(task.function, task.args, task.kwargs),
                timeout=task.timeout,
            )
            
            task_result.status = TaskStatus.COMPLETED
            task_result.result = result
            logger.info(f"Task \"{task.name}\" completed successfully")
        except asyncio.TimeoutError:
            task_result.status = TaskStatus.FAILED
            task_result.error = f"Task timed out after {task.timeout} seconds"
            logger.error(f"Task \"{task.name}\" timed out after {task.timeout} seconds")
        except Exception as e:
            task_result.status = TaskStatus.FAILED
            task_result.error = str(e)
            logger.error(f"Task \"{task.name}\" failed: {str(e)}")
        
        task_result.end_time = datetime.now()
        task_result.duration = (task_result.end_time - task_result.start_time).total_seconds()
        self.task_results[task.id] = task_result
        
        return task_result
    
    async def _execute_function(self, func: Callable, args: List[Any], kwargs: Dict[str, Any]) -> Any:
        """
        Execute a function, handling both synchronous and asynchronous functions.
        
        Args:
            func: The function to execute.
            args: Positional arguments to pass to the function.
            kwargs: Keyword arguments to pass to the function.
            
        Returns:
            The result of the function.
        """
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    
    async def run_workflow(self, workflow_id: str) -> Dict[str, TaskResult]:
        """
        Run a workflow.
        
        Args:
            workflow_id: The ID of the workflow to run.
            
        Returns:
            Dictionary of task results.
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow with ID {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        logger.info(f"Running workflow: {workflow.name} (ID: {workflow_id})")
        
        # Reset task results for this workflow
        for task in workflow.tasks:
            self.task_results[task.id] = TaskResult(
                task_id=task.id,
                status=TaskStatus.PENDING,
            )
        
        # Build dependency graph
        tasks_by_id = {task.id: task for task in workflow.tasks}
        
        # Execute tasks in dependency order
        remaining_tasks = set(tasks_by_id.keys())
        completed_tasks = set()
        
        while remaining_tasks:
            # Find tasks that can be executed (all dependencies satisfied)
            executable_tasks = []
            for task_id in remaining_tasks:
                task = tasks_by_id[task_id]
                if all(dep_id in completed_tasks for dep_id in task.depends_on):
                    executable_tasks.append(task)
            
            if not executable_tasks:
                # Circular dependency or missing dependency
                logger.error(f"Circular or missing dependency detected in workflow {workflow_id}")
                break
            
            # Execute tasks in parallel
            tasks = [self.execute_task(task) for task in executable_tasks]
            results = await asyncio.gather(*tasks)
            
            # Update completed tasks
            for task, result in zip(executable_tasks, results):
                remaining_tasks.remove(task.id)
                if result.status == TaskStatus.COMPLETED:
                    completed_tasks.add(task.id)
        
        logger.info(f"Workflow {workflow.name} completed")
        return {task_id: self.task_results[task_id] for task_id in tasks_by_id}
    
    def schedule_workflow(self, workflow_id: str, interval: int) -> asyncio.Task:
        """
        Schedule a workflow to run at regular intervals.
        
        Args:
            workflow_id: The ID of the workflow to schedule.
            interval: The interval in seconds.
            
        Returns:
            The scheduled task.
        """
        async def _run_scheduled_workflow():
            while True:
                try:
                    await self.run_workflow(workflow_id)
                except Exception as e:
                    logger.error(f"Error running scheduled workflow {workflow_id}: {str(e)}")
                await asyncio.sleep(interval)
        
        task = self.loop.create_task(_run_scheduled_workflow())
        self.scheduled_tasks[workflow_id] = task
        logger.info(f"Scheduled workflow {workflow_id} to run every {interval} seconds")
        return task
    
    def cancel_scheduled_workflow(self, workflow_id: str) -> None:
        """
        Cancel a scheduled workflow.
        
        Args:
            workflow_id: The ID of the workflow to cancel.
        """
        if workflow_id in self.scheduled_tasks:
            self.scheduled_tasks[workflow_id].cancel()
            del self.scheduled_tasks[workflow_id]
            logger.info(f"Cancelled scheduled workflow {workflow_id}")
        else:
            logger.warning(f"No scheduled workflow found with ID {workflow_id}")
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """
        Get the status of a workflow.
        
        Args:
            workflow_id: The ID of the workflow to get the status of.
            
        Returns:
            Dictionary with workflow status.
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow with ID {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        task_statuses = {}
        
        for task in workflow.tasks:
            if task.id in self.task_results:
                result = self.task_results[task.id]
                task_statuses[task.id] = {
                    "name": task.name,
                    "status": result.status,
                    "start_time": result.start_time,
                    "end_time": result.end_time,
                    "duration": result.duration,
                    "error": result.error,
                }
            else:
                task_statuses[task.id] = {
                    "name": task.name,
                    "status": TaskStatus.PENDING,
                }
        
        return {
            "id": workflow.id,
            "name": workflow.name,
            "description": workflow.description,
            "created_at": workflow.created_at,
            "tasks": task_statuses,
            "is_scheduled": workflow_id in self.scheduled_tasks,
        }

# Singleton instance
_engine_instance = None

def get_workflow_engine() -> WorkflowEngine:
    """
    Get the workflow engine singleton instance.
    
    Returns:
        WorkflowEngine instance.
    """
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = WorkflowEngine()
    return _engine_instance

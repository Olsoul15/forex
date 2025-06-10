"""
Base Agent Module for the AI Forex Trading System.

This module defines the core agent architecture, providing base classes
for all specialized agents in the system. It includes the BaseAgent abstract class,
AgentTool interface, memory management, and metrics tracking.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable, Union, Type
from dataclasses import dataclass, field
import uuid
from datetime import datetime
import json
from pathlib import Path
import time
import asyncio
from pydantic import BaseModel
from forex_ai.utils.logging import get_logger
from forex_ai.exceptions import AgentError

logger = get_logger(__name__)


class AgentTool(ABC):
    """
    Interface for tools that agents can use to interact with the environment.

    Agent tools provide specialized functionality that agents can leverage
    to perform specific tasks, such as fetching data, executing trades, etc.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the tool."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Get a description of what the tool does."""
        pass

    @abstractmethod
    def run(self, **kwargs) -> Any:
        """
        Execute the tool's functionality.

        Args:
            **kwargs: Tool-specific parameters

        Returns:
            Tool-specific result
        """
        pass

    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the parameters the tool accepts.

        Returns:
            Dictionary mapping parameter names to parameter metadata
        """
        return {}


class AgentMemory:
    """
    Memory management for agents, storing observations, decisions, and state.
    """

    def __init__(self, max_items: int = 1000):
        """
        Initialize agent memory.

        Args:
            max_items: Maximum number of items to store in memory
        """
        self.observations = []
        self.decisions = []
        self.state = {}
        self.max_items = max_items

    def add_observation(self, observation: Dict[str, Any]) -> None:
        """
        Add an observation to memory.

        Args:
            observation: Dictionary containing observation data
        """
        if len(self.observations) >= self.max_items:
            self.observations.pop(0)  # Remove oldest item

        # Add timestamp if not present
        if "timestamp" not in observation:
            observation["timestamp"] = datetime.now().isoformat()

        self.observations.append(observation)

    def add_decision(self, decision: Dict[str, Any]) -> None:
        """
        Add a decision to memory.

        Args:
            decision: Dictionary containing decision data
        """
        if len(self.decisions) >= self.max_items:
            self.decisions.pop(0)  # Remove oldest item

        # Add timestamp if not present
        if "timestamp" not in decision:
            decision["timestamp"] = datetime.now().isoformat()

        self.decisions.append(decision)

    def update_state(self, key: str, value: Any) -> None:
        """
        Update a state variable.

        Args:
            key: State variable name
            value: State variable value
        """
        self.state[key] = value

    def get_state(self, key: str, default: Any = None) -> Any:
        """
        Get a state variable.

        Args:
            key: State variable name
            default: Default value if key not found

        Returns:
            State variable value
        """
        return self.state.get(key, default)

    def get_recent_observations(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get the n most recent observations.

        Args:
            n: Number of observations to return

        Returns:
            List of recent observations
        """
        return self.observations[-n:]

    def get_recent_decisions(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get the n most recent decisions.

        Args:
            n: Number of decisions to return

        Returns:
            List of recent decisions
        """
        return self.decisions[-n:]

    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """
        Save memory to a file.

        Args:
            file_path: Path to save the memory
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        memory_data = {
            "observations": self.observations,
            "decisions": self.decisions,
            "state": self.state,
            "exported_at": datetime.now().isoformat(),
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(memory_data, f, indent=2)

    def load_from_file(self, file_path: Union[str, Path]) -> None:
        """
        Load memory from a file.

        Args:
            file_path: Path to load the memory from

        Raises:
            FileNotFoundError: If file does not exist
        """
        file_path = Path(file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            memory_data = json.load(f)

        self.observations = memory_data.get("observations", [])
        self.decisions = memory_data.get("decisions", [])
        self.state = memory_data.get("state", {})

        # Ensure we don't exceed max_items
        if len(self.observations) > self.max_items:
            self.observations = self.observations[-self.max_items :]
        if len(self.decisions) > self.max_items:
            self.decisions = self.decisions[-self.max_items :]


class AgentMetrics:
    """
    Performance metrics tracking for agents.
    """

    def __init__(self):
        """Initialize agent metrics tracking."""
        self.execution_times = []
        self.decision_counts = {}
        self.error_counts = {}
        self.custom_metrics = {}
        self.start_time = None

    def start_execution(self) -> None:
        """Mark the start of a task execution."""
        self.start_time = time.time()

    def end_execution(self, task_name: str) -> float:
        """
        Mark the end of a task execution and record the time.

        Args:
            task_name: Name of the task

        Returns:
            Execution time in seconds
        """
        if self.start_time is None:
            logger.warning("end_execution called without start_execution")
            return 0.0

        execution_time = time.time() - self.start_time
        self.execution_times.append(
            {
                "task": task_name,
                "time": execution_time,
                "timestamp": datetime.now().isoformat(),
            }
        )
        self.start_time = None

        return execution_time

    def record_decision(self, decision_type: str) -> None:
        """
        Record a decision of a specific type.

        Args:
            decision_type: Type of decision made
        """
        self.decision_counts[decision_type] = (
            self.decision_counts.get(decision_type, 0) + 1
        )

    def record_error(self, error_type: str) -> None:
        """
        Record an error of a specific type.

        Args:
            error_type: Type of error encountered
        """
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

    def track_metric(self, metric_name: str, value: float) -> None:
        """
        Track a custom metric.

        Args:
            metric_name: Name of the metric
            value: Value of the metric
        """
        if metric_name not in self.custom_metrics:
            self.custom_metrics[metric_name] = []

        self.custom_metrics[metric_name].append(
            {"value": value, "timestamp": datetime.now().isoformat()}
        )

    def get_average_execution_time(self, task_name: Optional[str] = None) -> float:
        """
        Get the average execution time.

        Args:
            task_name: Optional task name to filter by

        Returns:
            Average execution time in seconds
        """
        if task_name:
            times = [
                entry["time"]
                for entry in self.execution_times
                if entry["task"] == task_name
            ]
        else:
            times = [entry["time"] for entry in self.execution_times]

        return sum(times) / len(times) if times else 0.0

    def get_decision_count(self, decision_type: Optional[str] = None) -> int:
        """
        Get the count of decisions.

        Args:
            decision_type: Optional decision type to filter by

        Returns:
            Count of decisions
        """
        if decision_type:
            return self.decision_counts.get(decision_type, 0)
        else:
            return sum(self.decision_counts.values())

    def get_error_count(self, error_type: Optional[str] = None) -> int:
        """
        Get the count of errors.

        Args:
            error_type: Optional error type to filter by

        Returns:
            Count of errors
        """
        if error_type:
            return self.error_counts.get(error_type, 0)
        else:
            return sum(self.error_counts.values())

    def get_metric_values(self, metric_name: str) -> List[Dict[str, Any]]:
        """
        Get values for a custom metric.

        Args:
            metric_name: Name of the metric

        Returns:
            List of metric values with timestamps
        """
        return self.custom_metrics.get(metric_name, [])

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all metrics.

        Returns:
            Dictionary containing metrics summary
        """
        return {
            "execution_times": {
                "average": self.get_average_execution_time(),
                "by_task": {
                    task: self.get_average_execution_time(task)
                    for task in set(entry["task"] for entry in self.execution_times)
                },
            },
            "decisions": self.decision_counts,
            "errors": self.error_counts,
            "custom_metrics": {
                metric_name: {
                    "latest": entries[-1]["value"] if entries else None,
                    "average": (
                        sum(entry["value"] for entry in entries) / len(entries)
                        if entries
                        else 0.0
                    ),
                    "count": len(entries),
                }
                for metric_name, entries in self.custom_metrics.items()
            },
        }


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system.

    This class defines the common interface and functionality for all agents.
    Specialized agents should inherit from this class and implement the abstract methods.
    """

    def __init__(self, name: str = None, config: Dict[str, Any] = None):
        """
        Initialize the base agent.

        Args:
            name: Name of the agent (defaults to class name if None)
            config: Configuration parameters for the agent
        """
        self.id = str(uuid.uuid4())
        self.name = name or self.__class__.__name__
        self.config = config or {}
        self.tools = {}
        self.memory = AgentMemory()
        self.metrics = AgentMetrics()
        self.running = False

        # Initialize agent
        self.initialize()

    def initialize(self) -> None:
        """
        Initialize the agent.

        This method is called during initialization and can be overridden
        by subclasses to perform any setup needed.
        """
        pass

    def add_tool(self, tool: AgentTool) -> None:
        """
        Add a tool to the agent.

        Args:
            tool: AgentTool instance to add
        """
        self.tools[tool.name] = tool
        logger.info(f"Agent {self.name} added tool: {tool.name}")

    def get_tool(self, tool_name: str) -> Optional[AgentTool]:
        """
        Get a tool by name.

        Args:
            tool_name: Name of the tool to get

        Returns:
            AgentTool instance if found, None otherwise
        """
        return self.tools.get(tool_name)

    def use_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Use a tool by name.

        Args:
            tool_name: Name of the tool to use
            **kwargs: Parameters to pass to the tool

        Returns:
            Result of the tool execution

        Raises:
            AgentError: If the tool is not found
        """
        tool = self.get_tool(tool_name)
        if not tool:
            raise AgentError(f"Tool not found: {tool_name}")

        try:
            self.metrics.start_execution()
            result = tool.run(**kwargs)
            self.metrics.end_execution(f"tool:{tool_name}")
            return result
        except Exception as e:
            self.metrics.record_error(f"tool_error:{tool_name}")
            logger.error(f"Error using tool {tool_name}: {str(e)}")
            raise AgentError(f"Error using tool {tool_name}: {str(e)}") from e

    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """
        Process input data and generate output.

        This is the main entry point for agent processing.
        Subclasses must implement this method.

        Args:
            input_data: Input data to process

        Returns:
            Processing result
        """
        pass

    async def process_async(self, input_data: Any) -> Any:
        """
        Asynchronous version of process.

        Default implementation runs process in a thread pool.
        Subclasses can override for true async processing.

        Args:
            input_data: Input data to process

        Returns:
            Processing result
        """
        return await asyncio.to_thread(self.process, input_data)

    def start(self) -> None:
        """Start the agent."""
        self.running = True
        logger.info(f"Agent {self.name} started")

    def stop(self) -> None:
        """Stop the agent."""
        self.running = False
        logger.info(f"Agent {self.name} stopped")

    def is_running(self) -> bool:
        """
        Check if the agent is running.

        Returns:
            True if the agent is running, False otherwise
        """
        return self.running

    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the agent.

        Returns:
            Dictionary containing agent status
        """
        return {
            "id": self.id,
            "name": self.name,
            "running": self.running,
            "tools": list(self.tools.keys()),
            "metrics": self.metrics.get_summary(),
        }

    def __str__(self) -> str:
        """
        Get string representation of the agent.

        Returns:
            String representation
        """
        return f"{self.name} (ID: {self.id})"

"""
Base agent framework for the Forex AI Trading System.
Provides core functionality for all agents in the system.
"""

import logging
import uuid
import time
import json
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Callable, TypeVar, Generic
from pydantic import BaseModel, Field, validator

from forex_ai.exceptions import (
    AgentError,
    AgentExecutionError,
    ToolExecutionError as AgentToolError,
)
from forex_ai.custom_types import (
    AgentMetrics as TypesAgentMetrics,
    AgentMemoryEntry,
    AgentTool,
)

# Type definitions
T = TypeVar("T")
Input = TypeVar("Input")
Output = TypeVar("Output")

logger = logging.getLogger(__name__)


class AgentMemoryEntry(BaseModel):
    """
    Entry in the agent's memory.

    Attributes:
        id: Unique identifier for the memory entry
        timestamp: When the entry was created
        input_data: The input that was processed
        output_data: The output that was generated
        metadata: Additional metadata about the processing
    """

    id: str
    timestamp: datetime
    input_data: Any
    output_data: Optional[Any] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentTool(BaseModel, Generic[Input, Output]):
    """
    Tool that can be used by an agent to perform specific tasks.

    Attributes:
        name: Tool name
        description: Tool description
        enabled: Whether the tool is enabled
        func: Function that implements the tool
        required_permissions: Permissions required to use this tool
    """

    name: str
    description: str
    enabled: bool = True
    func: Callable[[Input], Output]
    required_permissions: List[str] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

    def execute(self, input_data: Input) -> Output:
        """
        Execute the tool with the given input.

        Args:
            input_data: Input for the tool

        Returns:
            Tool execution result

        Raises:
            AgentToolError: If tool execution fails
        """
        if not self.enabled:
            raise AgentToolError(f"Tool '{self.name}' is disabled")

        try:
            return self.func(input_data)
        except Exception as e:
            raise AgentToolError(f"Tool '{self.name}' execution failed: {str(e)}")


class AgentMetrics(BaseModel):
    """
    Metrics for agent performance monitoring.

    Attributes:
        total_requests: Total number of requests processed
        successful_requests: Number of successfully processed requests
        failed_requests: Number of failed requests
        avg_response_time: Average response time in seconds
        total_processing_time: Total processing time in seconds
        last_execution_time: Timestamp of last execution
    """

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    total_processing_time: float = 0.0
    last_execution_time: Optional[datetime] = None

    def update_execution_metrics(self, start_time: float, success: bool) -> None:
        """
        Update execution metrics after processing a request.

        Args:
            start_time: Request processing start time (time.time())
            success: Whether the request was successful
        """
        execution_time = time.time() - start_time
        self.total_requests += 1

        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1

        self.total_processing_time += execution_time
        self.avg_response_time = self.total_processing_time / self.total_requests
        self.last_execution_time = datetime.now()

    def to_types_metrics(self) -> TypesAgentMetrics:
        """
        Convert to TypesAgentMetrics for compatibility with the rest of the system.

        Returns:
            TypesAgentMetrics instance
        """
        success_rate = (
            self.successful_requests / self.total_requests
            if self.total_requests > 0
            else 1.0
        )

        return TypesAgentMetrics(
            calls=self.total_requests,
            errors=self.failed_requests,
            avg_response_time=self.avg_response_time,
            last_call_time=self.last_execution_time,
            success_rate=success_rate,
        )


class BaseAgent:
    """
    Base class for all agents in the system.
    Provides core functionality for agent operations.

    Attributes:
        name: Agent name
        model: LLM model to use for processing
        tools: List of tools available to the agent
        memory: Agent memory of past interactions
        metrics: Performance metrics
        logger: Logger instance
    """

    def __init__(
        self,
        name: str,
        model_id: str,
        tools: Optional[List[AgentTool]] = None,
        max_memory_entries: int = 100,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the agent.

        Args:
            name: Agent name
            model_id: ID of the model to use
            tools: List of tools available to the agent
            max_memory_entries: Maximum number of memory entries to keep
            logger: Logger instance
        """
        self.name = name
        self.model_id = model_id
        self.tools = tools or []
        self.memory: List[AgentMemoryEntry] = []
        self.max_memory_entries = max_memory_entries
        self.logger = logger or logging.getLogger(f"agent.{name}")

        # Get model controller - Import moved here
        from forex_ai.models.controller import get_model_controller

        self.model_controller = get_model_controller()

    @property
    def model(self):
        """Get the model instance from the controller."""
        return self.model_controller.get_model(self.model_id)

    @property
    def metrics(self) -> AgentMetrics:
        """Get agent metrics from the model controller."""
        status = self.model_controller.get_model_status(self.model_id)
        return AgentMetrics(
            calls=status["metrics"]["calls"],
            errors=status["metrics"]["errors"],
            avg_response_time=status["metrics"]["avg_response_time"],
            last_call_time=status["metrics"]["last_call_time"],
            success_rate=status["metrics"]["success_rate"],
        )

    def process(self, input_data: Any) -> Any:
        """
        Process input data using the model.

        Args:
            input_data: Input data to process

        Returns:
            Processed output
        """
        start_time = time.time()
        success = True

        try:
            result = self.model.predict(input_data)
            return result
        except Exception as e:
            success = False
            self.logger.error(f"Error processing input: {str(e)}")
            raise
        finally:
            inference_time = time.time() - start_time
            self.model_controller.update_metrics(
                self.model_id, success=success, inference_time=inference_time
            )

    def add_memory_entry(self, entry: AgentMemoryEntry) -> None:
        """
        Add an entry to agent memory.

        Args:
            entry: Memory entry to add
        """
        self.memory.append(entry)
        if len(self.memory) > self.max_memory_entries:
            self.memory = self.memory[-self.max_memory_entries :]

    def clear_memory(self) -> None:
        """Clear agent memory."""
        self.memory = []

    def get_metrics(self) -> TypesAgentMetrics:
        """
        Get agent metrics.

        Returns:
            Agent metrics
        """
        metrics = self.metrics
        return TypesAgentMetrics(
            calls=metrics.calls,
            errors=metrics.errors,
            avg_response_time=metrics.avg_response_time,
            last_call_time=metrics.last_call_time,
            success_rate=metrics.success_rate,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert agent to dictionary.

        Returns:
            Dictionary representation of the agent
        """
        return {
            "name": self.name,
            "model_id": self.model_id,
            "tools": [t.name for t in self.tools],
            "memory_size": len(self.memory),
            "max_memory_entries": self.max_memory_entries,
            "metrics": self.get_metrics().dict(),
        }

    def run(self, input_data: Any) -> Any:
        """
        Process input and return response.

        Args:
            input_data: Input data to process

        Returns:
            Processing result

        Raises:
            AgentExecutionError: If processing fails
        """
        start_time = time.time()
        memory_id = str(uuid.uuid4())
        success = False

        # Create memory entry for this execution
        memory_entry = AgentMemoryEntry(
            id=memory_id,
            timestamp=datetime.now(),
            input_data=input_data,
            metadata={"agent_name": self.name},
        )

        try:
            self.logger.info(f"Agent '{self.name}' processing input")

            # Process the input (to be implemented by subclasses)
            result = self._process(input_data)

            # Update memory with result
            memory_entry.output_data = result
            memory_entry.metadata["success"] = True

            self.logger.info(f"Agent '{self.name}' successfully processed input")
            success = True
            return result

        except Exception as e:
            # Log the error
            self.logger.error(f"Agent '{self.name}' execution failed: {str(e)}")
            self.logger.error(traceback.format_exc())

            # Update memory with error information
            memory_entry.metadata["success"] = False
            memory_entry.metadata["error"] = str(e)
            memory_entry.metadata["error_type"] = type(e).__name__

            # Re-raise as AgentExecutionError
            raise AgentExecutionError(
                f"Agent '{self.name}' execution failed: {str(e)}"
            ) from e

        finally:
            # Update metrics
            self.metrics.update_execution_metrics(start_time, success)

            # Add to memory
            self.memory.append(memory_entry)

            # Prune memory if necessary
            if len(self.memory) > self.max_memory_entries:
                self.memory = self.memory[-self.max_memory_entries :]

            # Log metrics
            self.logger.debug(
                f"Agent '{self.name}' metrics: {json.dumps(self.metrics.dict(), default=str)}"
            )

    def _process(self, input_data: Any) -> Any:
        """
        Process the input data. To be implemented by subclasses.

        Args:
            input_data: Input data to process

        Returns:
            Processing result

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement _process method")

    def add_tool(self, tool: AgentTool) -> None:
        """
        Add a tool to the agent.

        Args:
            tool: Tool to add

        Raises:
            AgentError: If a tool with the same name already exists
        """
        # Check if a tool with this name already exists
        if any(t.name == tool.name for t in self.tools):
            raise AgentError(f"Tool with name '{tool.name}' already exists")

        self.tools.append(tool)
        self.logger.info(f"Added tool '{tool.name}' to agent '{self.name}'")

    def remove_tool(self, tool_name: str) -> bool:
        """
        Remove a tool from the agent.

        Args:
            tool_name: Name of the tool to remove

        Returns:
            True if tool was removed, False if not found
        """
        tool_index = None

        for i, tool in enumerate(self.tools):
            if tool.name == tool_name:
                tool_index = i
                break

        if tool_index is not None:
            self.tools.pop(tool_index)
            self.logger.info(f"Removed tool '{tool_name}' from agent '{self.name}'")
            return True

        return False

    def get_tool(self, tool_name: str) -> Optional[AgentTool]:
        """
        Get a tool by name.

        Args:
            tool_name: Name of the tool to get

        Returns:
            Tool if found, None otherwise
        """
        for tool in self.tools:
            if tool.name == tool_name:
                return tool

        return None

    def execute_tool(self, tool_name: str, input_data: Any) -> Any:
        """
        Execute a tool by name.

        Args:
            tool_name: Name of the tool to execute
            input_data: Input for the tool

        Returns:
            Tool execution result

        Raises:
            AgentError: If tool doesn't exist
            AgentToolError: If tool execution fails
        """
        tool = self.get_tool(tool_name)

        if not tool:
            raise AgentError(f"Tool '{tool_name}' not found")

        start_time = time.time()
        self.logger.debug(f"Executing tool '{tool_name}'")

        try:
            result = tool.execute(input_data)
            execution_time = time.time() - start_time
            self.logger.debug(f"Tool '{tool_name}' executed in {execution_time:.2f}s")
            return result
        except AgentToolError:
            # Re-raise tool errors
            raise
        except Exception as e:
            # Wrap other exceptions
            raise AgentToolError(f"Tool '{tool_name}' execution failed: {str(e)}")

    def get_memory(self, limit: Optional[int] = None) -> List[AgentMemoryEntry]:
        """
        Get agent memory entries.

        Args:
            limit: Maximum number of entries to return (most recent first)

        Returns:
            List of memory entries
        """
        entries = list(reversed(self.memory))  # Most recent first

        if limit is not None:
            entries = entries[:limit]

        return entries

    def reset_metrics(self) -> None:
        """Reset agent performance metrics."""
        self.metrics = AgentMetrics()
        self.logger.info(f"Reset metrics for agent '{self.name}'")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert agent to dictionary.

        Returns:
            Dictionary representation of the agent
        """
        return {
            "name": self.name,
            "model_id": self.model_id,
            "tools": [t.name for t in self.tools],
            "memory_size": len(self.memory),
            "max_memory_entries": self.max_memory_entries,
            "metrics": self.get_metrics().dict(),
        }

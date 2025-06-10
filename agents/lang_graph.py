"""
LangGraph Integration Module for the AI Forex Trading System.

This module provides integration with LangGraph for creating workflows
that coordinate agent activities in a structured, graph-based manner.
"""

from typing import Dict, List, Optional, Any, Union, Callable, Type
import asyncio
from datetime import datetime
import json
import uuid
from pathlib import Path

from .base import BaseAgent
from .communication import AgentCoordinator, AgentMessage
from forex_ai.utils.logging import get_logger
from forex_ai.exceptions import LangGraphError

logger = get_logger(__name__)

try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import AnyMessage

    LANGGRAPH_AVAILABLE = True
except ImportError:
    logger.warning("LangGraph not available. Some functionality will be limited.")
    LANGGRAPH_AVAILABLE = False

    # Create dummy classes to avoid errors
    class StateGraph:
        def __init__(self, *args, **kwargs):
            pass

    class AnyMessage:
        def __init__(self, *args, **kwargs):
            pass

    END = "END"


class GraphNode:
    """Base class for nodes in a LangGraph workflow."""

    def __init__(self, name: str):
        """
        Initialize a graph node.

        Args:
            name: Name of the node
        """
        self.name = name

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the current state and return updated state.

        Args:
            state: Current workflow state

        Returns:
            Updated workflow state
        """
        raise NotImplementedError("Subclasses must implement process method")

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Callable interface for the node.

        Args:
            state: Current workflow state

        Returns:
            Updated workflow state
        """
        return asyncio.run(self.process(state))


class AgentNode(GraphNode):
    """Node that processes state through an agent."""

    def __init__(self, name: str, agent: BaseAgent):
        """
        Initialize an agent node.

        Args:
            name: Name of the node
            agent: Agent instance
        """
        super().__init__(name)
        self.agent = agent

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the current state through the agent.

        Args:
            state: Current workflow state

        Returns:
            Updated workflow state with agent output
        """
        try:
            # Extract input data for agent
            input_data = state.get("input", {})

            # Process with agent - use async version if possible
            if hasattr(self.agent, "process_async"):
                result = await self.agent.process_async(input_data)
            else:
                result = self.agent.process(input_data)

            # Update state with result
            state[self.name] = result

            # Add result to last_node_result for easier access
            state["last_node_result"] = result

            # Update execution history
            if "execution_history" not in state:
                state["execution_history"] = []

            state["execution_history"].append(
                {
                    "node": self.name,
                    "timestamp": datetime.now().isoformat(),
                    "success": True,
                }
            )

            return state
        except Exception as e:
            logger.error(f"Error in agent node {self.name}: {str(e)}")

            # Update state with error
            state[self.name] = {"error": str(e)}

            # Update execution history
            if "execution_history" not in state:
                state["execution_history"] = []

            state["execution_history"].append(
                {
                    "node": self.name,
                    "timestamp": datetime.now().isoformat(),
                    "success": False,
                    "error": str(e),
                }
            )

            return state


class FunctionNode(GraphNode):
    """Node that processes state through a function."""

    def __init__(self, name: str, func: Callable):
        """
        Initialize a function node.

        Args:
            name: Name of the node
            func: Function to execute
        """
        super().__init__(name)
        self.func = func

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the current state through the function.

        Args:
            state: Current workflow state

        Returns:
            Updated workflow state with function output
        """
        try:
            # Call function with state
            result = self.func(state)

            # Update state with result
            state[self.name] = result

            # Add result to last_node_result for easier access
            state["last_node_result"] = result

            # Update execution history
            if "execution_history" not in state:
                state["execution_history"] = []

            state["execution_history"].append(
                {
                    "node": self.name,
                    "timestamp": datetime.now().isoformat(),
                    "success": True,
                }
            )

            return state
        except Exception as e:
            logger.error(f"Error in function node {self.name}: {str(e)}")

            # Update state with error
            state[self.name] = {"error": str(e)}

            # Update execution history
            if "execution_history" not in state:
                state["execution_history"] = []

            state["execution_history"].append(
                {
                    "node": self.name,
                    "timestamp": datetime.now().isoformat(),
                    "success": False,
                    "error": str(e),
                }
            )

            return state


class ConditionalEdge:
    """Conditional edge for a LangGraph workflow."""

    def __init__(self, condition: Callable[[Dict[str, Any]], str]):
        """
        Initialize a conditional edge.

        Args:
            condition: Function that evaluates the state and returns the next node name
        """
        self.condition = condition

    def __call__(self, state: Dict[str, Any]) -> str:
        """
        Callable interface for the edge.

        Args:
            state: Current workflow state

        Returns:
            Name of the next node
        """
        return self.condition(state)


class WorkflowDefinition:
    """Definition of a LangGraph workflow."""

    def __init__(self, name: str):
        """
        Initialize a workflow definition.

        Args:
            name: Name of the workflow
        """
        self.name = name
        self.nodes = {}
        self.edges = {}
        self.entry_point = None

    def add_node(self, node: GraphNode) -> "WorkflowDefinition":
        """
        Add a node to the workflow.

        Args:
            node: Node to add

        Returns:
            Self for chaining
        """
        self.nodes[node.name] = node
        return self

    def add_edge(
        self, from_node: str, to_node: str, condition: Optional[Callable] = None
    ) -> "WorkflowDefinition":
        """
        Add an edge between nodes.

        Args:
            from_node: Source node name
            to_node: Target node name
            condition: Optional condition for the edge

        Returns:
            Self for chaining

        Raises:
            LangGraphError: If source or target node not found
        """
        if from_node not in self.nodes:
            raise LangGraphError(f"Source node not found: {from_node}")

        if to_node != END and to_node not in self.nodes:
            raise LangGraphError(f"Target node not found: {to_node}")

        if from_node not in self.edges:
            self.edges[from_node] = {}

        if condition:
            self.edges[from_node][to_node] = ConditionalEdge(condition)
        else:
            # Simple edge
            self.edges[from_node][to_node] = lambda _: True

        return self

    def set_entry_point(self, node_name: str) -> "WorkflowDefinition":
        """
        Set the entry point node.

        Args:
            node_name: Name of the entry point node

        Returns:
            Self for chaining

        Raises:
            LangGraphError: If node not found
        """
        if node_name not in self.nodes:
            raise LangGraphError(f"Node not found: {node_name}")

        self.entry_point = node_name
        return self

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert workflow definition to dictionary.

        Returns:
            Dictionary representation of the workflow
        """
        return {
            "name": self.name,
            "nodes": list(self.nodes.keys()),
            "edges": [
                {
                    "from": from_node,
                    "to": to_node,
                    "has_condition": not callable(condition)
                    or isinstance(condition, ConditionalEdge),
                }
                for from_node, edges in self.edges.items()
                for to_node, condition in edges.items()
            ],
            "entry_point": self.entry_point,
        }


class WorkflowEngine:
    """Engine for executing LangGraph workflows."""

    def __init__(self, coordinator: Optional[AgentCoordinator] = None):
        """
        Initialize the workflow engine.

        Args:
            coordinator: Optional agent coordinator for workflow execution
        """
        self.workflows = {}
        self.instances = {}
        self.coordinator = coordinator or AgentCoordinator()

    def register_workflow(self, workflow: WorkflowDefinition) -> None:
        """
        Register a workflow definition.

        Args:
            workflow: Workflow definition to register
        """
        if not LANGGRAPH_AVAILABLE:
            raise LangGraphError(
                "LangGraph is not available. Cannot register workflow."
            )

        if not workflow.entry_point:
            raise LangGraphError(f"Workflow {workflow.name} has no entry point")

        # Create LangGraph StateGraph
        graph = StateGraph()

        # Add nodes
        for node_name, node in workflow.nodes.items():
            graph.add_node(node_name, node)

        # Add conditional edges
        for from_node, edges in workflow.edges.items():
            if len(edges) == 1:
                # Simple edge
                to_node = next(iter(edges.keys()))
                graph.add_edge(from_node, to_node)
            else:
                # Conditional edges
                conditions = {}
                for to_node, condition in edges.items():
                    if isinstance(condition, ConditionalEdge):
                        conditions[to_node] = condition
                    else:
                        # Simple condition that always returns True
                        conditions[to_node] = lambda _: True

                # Create router function
                def router(state):
                    for target, condition in conditions.items():
                        try:
                            result = condition(state)
                            if result:
                                return target
                        except Exception as e:
                            logger.error(
                                f"Error in edge condition from {from_node} to {target}: {str(e)}"
                            )

                    # Default to END if no condition matches
                    return END

                graph.add_conditional_edges(from_node, router)

        # Set entry point
        graph.set_entry_point(workflow.entry_point)

        # Compile graph
        compiled_graph = graph.compile()

        # Store workflow
        self.workflows[workflow.name] = {
            "definition": workflow,
            "graph": compiled_graph,
        }

        logger.info(f"Registered workflow: {workflow.name}")

    def create_instance(
        self, workflow_name: str, initial_state: Dict[str, Any] = None
    ) -> str:
        """
        Create a new instance of a workflow.

        Args:
            workflow_name: Name of the workflow
            initial_state: Initial state for the workflow

        Returns:
            Instance ID

        Raises:
            LangGraphError: If workflow not found
        """
        if workflow_name not in self.workflows:
            raise LangGraphError(f"Workflow not found: {workflow_name}")

        # Generate instance ID
        instance_id = str(uuid.uuid4())

        # Initialize state
        state = initial_state or {}

        # Add metadata
        state["__workflow_name__"] = workflow_name
        state["__instance_id__"] = instance_id
        state["__status__"] = "created"
        state["__created_at__"] = datetime.now().isoformat()

        # Store instance
        self.instances[instance_id] = {
            "workflow_name": workflow_name,
            "state": state,
            "status": "created",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

        logger.info(f"Created workflow instance: {instance_id} of {workflow_name}")

        return instance_id

    def execute_instance(self, instance_id: str) -> Dict[str, Any]:
        """
        Execute a workflow instance.

        Args:
            instance_id: Instance ID

        Returns:
            Final workflow state

        Raises:
            LangGraphError: If instance not found
        """
        if not LANGGRAPH_AVAILABLE:
            raise LangGraphError("LangGraph is not available. Cannot execute workflow.")

        if instance_id not in self.instances:
            raise LangGraphError(f"Instance not found: {instance_id}")

        # Get instance
        instance = self.instances[instance_id]

        # Get workflow
        workflow_name = instance["workflow_name"]
        if workflow_name not in self.workflows:
            raise LangGraphError(f"Workflow not found: {workflow_name}")

        workflow = self.workflows[workflow_name]

        # Update instance status
        instance["status"] = "running"
        instance["updated_at"] = datetime.now().isoformat()

        try:
            # Execute workflow
            final_state = workflow["graph"].invoke(instance["state"])

            # Update instance
            instance["state"] = final_state
            instance["status"] = "completed"
            instance["updated_at"] = datetime.now().isoformat()

            logger.info(f"Executed workflow instance: {instance_id}")

            return final_state
        except Exception as e:
            logger.error(f"Error executing workflow instance {instance_id}: {str(e)}")

            # Update instance
            instance["status"] = "failed"
            instance["error"] = str(e)
            instance["updated_at"] = datetime.now().isoformat()

            raise LangGraphError(f"Error executing workflow: {str(e)}") from e

    def get_instance_state(self, instance_id: str) -> Dict[str, Any]:
        """
        Get the current state of a workflow instance.

        Args:
            instance_id: Instance ID

        Returns:
            Current workflow state

        Raises:
            LangGraphError: If instance not found
        """
        if instance_id not in self.instances:
            raise LangGraphError(f"Instance not found: {instance_id}")

        return self.instances[instance_id]["state"]

    def get_instance_status(self, instance_id: str) -> Dict[str, Any]:
        """
        Get the status of a workflow instance.

        Args:
            instance_id: Instance ID

        Returns:
            Instance status information

        Raises:
            LangGraphError: If instance not found
        """
        if instance_id not in self.instances:
            raise LangGraphError(f"Instance not found: {instance_id}")

        instance = self.instances[instance_id]

        return {
            "instance_id": instance_id,
            "workflow_name": instance["workflow_name"],
            "status": instance["status"],
            "created_at": instance["created_at"],
            "updated_at": instance["updated_at"],
            "error": instance.get("error"),
        }

    def save_instance(self, instance_id: str, file_path: Union[str, Path]) -> None:
        """
        Save a workflow instance to a file.

        Args:
            instance_id: Instance ID
            file_path: Path to save the instance

        Raises:
            LangGraphError: If instance not found
        """
        if instance_id not in self.instances:
            raise LangGraphError(f"Instance not found: {instance_id}")

        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.instances[instance_id], f, indent=2)

    def load_instance(self, file_path: Union[str, Path]) -> str:
        """
        Load a workflow instance from a file.

        Args:
            file_path: Path to load the instance from

        Returns:
            Instance ID

        Raises:
            LangGraphError: If file not found or workflow not found
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise LangGraphError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            instance = json.load(f)

        # Check if workflow exists
        workflow_name = instance["workflow_name"]
        if workflow_name not in self.workflows:
            raise LangGraphError(f"Workflow not found: {workflow_name}")

        # Get instance ID
        instance_id = instance["state"]["__instance_id__"]

        # Store instance
        self.instances[instance_id] = instance

        return instance_id

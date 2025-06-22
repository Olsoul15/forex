"""
Agent Manager for the Forex AI Trading System.

This module provides a manager for agent lifecycle and communication.
"""

import logging
import uuid
from typing import Dict, List, Any, Optional, Type, Union
from datetime import datetime

from forex_ai.agents.framework.agent import BaseAgent
from forex_ai.agents.framework.agent_types import UserQuery, SystemResponse

# Setup logging
logger = logging.getLogger(__name__)


class AgentManager:
    """
    Manager for agent lifecycle and communication.

    This class provides facilities for creating, managing, and coordinating
    agents in the system.
    """

    def __init__(self):
        """Initialize the agent manager."""
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_types: Dict[str, Type[BaseAgent]] = {}

    def register_agent_type(self, agent_type: Type[BaseAgent], name: Optional[str] = None) -> None:
        """
        Register an agent type with the manager.

        Args:
            agent_type: Agent class to register
            name: Name to register the agent type under (defaults to class name)
        """
        type_name = name or agent_type.__name__
        self.agent_types[type_name] = agent_type
        logger.info(f"Registered agent type: {type_name}")

    def create_agent(
        self, agent_type_name: str, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create an agent of the specified type.

        Args:
            agent_type_name: Name of the agent type to create
            name: Name for the new agent instance
            config: Configuration for the new agent

        Returns:
            ID of the created agent

        Raises:
            ValueError: If the agent type is not registered
        """
        if agent_type_name not in self.agent_types:
            raise ValueError(f"Agent type not registered: {agent_type_name}")

        agent_class = self.agent_types[agent_type_name]
        agent_name = name or f"{agent_type_name}_{uuid.uuid4().hex[:8]}"
        agent_config = config or {}

        agent = agent_class(name=agent_name, config=agent_config)
        self.agents[agent.id] = agent

        logger.info(f"Created agent: {agent_name} (ID: {agent.id}) of type {agent_type_name}")
        return agent.id

    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """
        Get an agent by ID.

        Args:
            agent_id: ID of the agent to get

        Returns:
            Agent if found, None otherwise
        """
        return self.agents.get(agent_id)

    def get_agents_by_type(self, agent_type_name: str) -> List[BaseAgent]:
        """
        Get all agents of a specific type.

        Args:
            agent_type_name: Type of agents to find

        Returns:
            List of matching agents
        """
        if agent_type_name not in self.agent_types:
            return []

        agent_class = self.agent_types[agent_type_name]
        return [agent for agent in self.agents.values() if isinstance(agent, agent_class)]

    def remove_agent(self, agent_id: str) -> bool:
        """
        Remove an agent from the manager.

        Args:
            agent_id: ID of the agent to remove

        Returns:
            True if removed, False if not found
        """
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            del self.agents[agent_id]
            logger.info(f"Removed agent: {agent.name} (ID: {agent_id})")
            return True
        return False

    async def process_query(self, query: Union[str, UserQuery], agent_id: Optional[str] = None) -> SystemResponse:
        """
        Process a user query using the specified agent.

        Args:
            query: User query string or UserQuery object
            agent_id: ID of the agent to use (if None, uses a default agent)

        Returns:
            System response

        Raises:
            ValueError: If the agent is not found or no default agent is available
        """
        # Convert string query to UserQuery if needed
        if isinstance(query, str):
            query = UserQuery(query_text=query)

        # If no agent specified but we have agents, use the first one
        if agent_id is None and self.agents:
            agent_id = next(iter(self.agents.keys()))

        if agent_id is None or agent_id not in self.agents:
            raise ValueError(f"Agent not found: {agent_id}")

        agent = self.agents[agent_id]
        logger.info(f"Processing query with agent: {agent.name} (ID: {agent_id})")

        # Process query with agent
        try:
            # This is a placeholder - actual implementation would depend on the agent interface
            response_text = f"Response from {agent.name}"
            return SystemResponse(
                response_text=response_text,
                source_agent=agent.name,
            )
        except Exception as e:
            logger.error(f"Error processing query with agent {agent.name}: {str(e)}")
            return SystemResponse(
                response_text=f"Error processing query: {str(e)}",
                source_agent="system",
                confidence=0.0,
            ) 
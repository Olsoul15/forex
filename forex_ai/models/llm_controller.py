"""
LLM Controller
Manages the initialization and access to the MCP agent.
"""

import logging
from typing import Dict, Any, Optional

from forex_ai.config.settings import get_settings
from forex_ai.models.mcp import MCPAgent, get_mcp_agent

# Configure logging
logger = logging.getLogger(__name__)


class LLMController:
    """
    Manages the initialization and access to the MCP agent.
    """

    def __init__(self):
        """
        Initializes the LLMController and the MCP agent.
        """
        self.settings = get_settings()
        self._clients: Dict[str, Any] = {}
        self._initialize_clients()

    def _initialize_clients(self) -> None:
        """
        Initializes the MCP agent.
        """
        try:
            # Initialize the MCP agent
            mcp_agent = get_mcp_agent()
            if mcp_agent.is_initialized():
                self._clients["mcp_agent"] = mcp_agent
                logger.info("MCP agent initialized successfully.")
            else:
                logger.error("Failed to initialize MCP agent.")
        except Exception as e:
            logger.error(f"Failed to initialize MCP agent: {str(e)}")

    def get_client(self, provider: str = "mcp_agent") -> Optional[Any]:
        """
        Returns the client for the specified provider.

        Args:
            provider: The name of the LLM provider. Defaults to "mcp_agent".

        Returns:
            The client object if available, otherwise None.
        """
        if provider != "mcp_agent":
            logger.warning(f"Provider '{provider}' is not supported. Only 'mcp_agent' is available.")
            return None
        return self._clients.get(provider)

    def get_available_models(self) -> Dict[str, Any]:
        """
        Returns a dictionary of available models.
        """
        if "mcp_agent" not in self._clients:
            return {}

        # MCP agent models
        mcp_models = {
            "mcp-agent": {
                "name": "mcp-agent",
                "provider": "mcp_agent",
                "class_name": "MCPAgent",
            },
        }
        return mcp_models

    def get_status(self) -> Dict[str, bool]:
        """
        Returns the connection status of the MCP agent.

        Returns:
            A dictionary with the provider name and its connection status.
        """
        return {"mcp_agent": "mcp_agent" in self._clients}

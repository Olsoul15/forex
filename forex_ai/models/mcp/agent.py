"""
MCP Agent for Forex AI Trading System

This module provides a client for interacting with MCP (Multi-Context Processing) agents
through various LLM providers.
"""

import os
import logging
import json
import asyncio
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
from datetime import datetime

from pydantic import BaseModel, Field

from forex_ai.config.settings import get_settings

logger = logging.getLogger(__name__)

class MessageRole(str, Enum):
    """Message roles for chat conversations."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

class Message(BaseModel):
    """Message model for chat conversations."""
    role: MessageRole
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ToolCall(BaseModel):
    """Tool call model."""
    id: str
    function: Dict[str, Any]

class MCPResponse(BaseModel):
    """Response model for MCP agent."""
    success: bool
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    model_name: Optional[str] = None
    raw_response: Optional[Any] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    error_message: Optional[str] = None
    cost: Optional[float] = None
    latency: Optional[float] = None

class MCPAgent:
    """
    Client for interacting with MCP agents through various LLM providers.
    
    This is a simplified version of the Google Generative Language client,
    designed specifically for the Forex AI Trading System.
    """
    
    def __init__(self):
        """Initialize the MCP agent."""
        self._is_initialized = False
        self.settings = get_settings()
        self.model_name = "mcp-agent"
        self.temperature = 0.7
        self.max_tokens = 4096
        self.history: List[Message] = []
        
        # Initialize the agent
        try:
            # In a real implementation, we would initialize the LLM client here
            # For now, we'll just set _is_initialized to True
            self._is_initialized = True
            logger.info(f"MCP agent initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing MCP agent: {e}", exc_info=True)
            self._is_initialized = False
    
    def is_initialized(self) -> bool:
        """Check if the agent is initialized."""
        return self._is_initialized
    
    def _add_to_history(self, role: str, content: str, name: Optional[str] = None, tool_call_id: Optional[str] = None):
        """Add a message to the history."""
        self.history.append(Message(
            role=role,
            content=content,
            name=name,
            tool_call_id=tool_call_id
        ))
        
        # Limit history size to avoid token overflow
        max_history = 20
        if len(self.history) > max_history:
            self.history = self.history[-max_history:]
    
    def get_history(self) -> List[Message]:
        """Get the conversation history."""
        return self.history
    
    def clear_history(self):
        """Clear the conversation history."""
        self.history = []
        logger.info("Chat history cleared.")
    
    async def ask(
        self,
        messages: List[Message],
        model_name_override: Optional[str] = None,
        temperature_override: Optional[float] = None,
        max_tokens_override: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> MCPResponse:
        """
        Send a message to the MCP agent and get a response.
        
        Args:
            messages: List of messages to send.
            model_name_override: Override the default model name.
            temperature_override: Override the default temperature.
            max_tokens_override: Override the default max tokens.
            tools: List of tools to make available to the agent.
            
        Returns:
            MCPResponse object with the agent's response.
        """
        if not self._is_initialized:
            return MCPResponse(
                success=False,
                error_message="MCP agent not initialized."
            )
        
        start_time = asyncio.get_event_loop().time()
        
        current_model_name = model_name_override or self.model_name
        current_temperature = temperature_override if temperature_override is not None else self.temperature
        current_max_tokens = max_tokens_override or self.max_tokens
        
        # Add incoming messages to our internal history
        for msg in messages:
            self._add_to_history(
                role=msg.role,
                content=msg.content,
                name=msg.name,
                tool_call_id=msg.tool_call_id
            )
        
        try:
            # In a real implementation, we would call the LLM API here
            # For now, we'll just return a mock response
            
            # Check if tools are provided and the last message is from the user
            if tools and messages and messages[-1].role == "user":
                # Mock a tool call
                tool_calls = [
                    ToolCall(
                        id="tool_call_1",
                        function={
                            "name": tools[0]["function"]["name"],
                            "arguments": "{}"
                        }
                    )
                ]
                
                response_text = None
                self._add_to_history(role="assistant", content="[Tool call requested]")
            else:
                # Mock a text response
                response_text = "This is a mock response from the MCP agent."
                tool_calls = None
                self._add_to_history(role="assistant", content=response_text)
            
            # Simulate some token counts
            input_tokens = sum(len(msg.content.split()) * 1.3 for msg in messages)
            output_tokens = len(response_text.split()) * 1.3 if response_text else 20
            
            latency = asyncio.get_event_loop().time() - start_time
            
            return MCPResponse(
                success=True,
                content=response_text,
                tool_calls=tool_calls,
                model_name=current_model_name,
                input_tokens=int(input_tokens),
                output_tokens=int(output_tokens),
                latency=latency
            )
        
        except Exception as e:
            logger.error(f"Error during MCP agent ask call: {e}", exc_info=True)
            latency = asyncio.get_event_loop().time() - start_time
            
            return MCPResponse(
                success=False,
                error_message=str(e),
                model_name=current_model_name,
                latency=latency
            )
    
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text.
        
        Args:
            text: The text to count tokens for.
            
        Returns:
            The number of tokens.
        """
        # In a real implementation, we would use a tokenizer
        # For now, we'll just use a simple approximation
        return int(len(text.split()) * 1.3)

# Singleton instance
_agent_instance = None

def get_mcp_agent() -> MCPAgent:
    """
    Get the MCP agent singleton instance.
    
    Returns:
        MCPAgent instance.
    """
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = MCPAgent()
    return _agent_instance 
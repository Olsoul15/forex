"""
MCP (Multi-Context Processing) Agent module for Forex AI Trading System.

This module provides a client for interacting with MCP agents through various LLM providers.
"""

from forex_ai.models.mcp.agent import MCPAgent, Message, MessageRole, ToolCall, MCPResponse, get_mcp_agent

__all__ = [
    'MCPAgent',
    'Message',
    'MessageRole',
    'ToolCall',
    'MCPResponse',
    'get_mcp_agent',
] 
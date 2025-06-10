"""
Agent framework core components for the Forex AI Trading System.

This package provides the core agent framework components, including:
- Base agent classes
- Agent tools and interfaces
- Memory management
- LangGraph integration
"""

# Import public interfaces from submodules
from forex_ai.agents.framework.agent import (
    BaseAgent,
    AgentTool,
    AgentMemoryEntry,
    AgentMetrics,
)

# Will be available after implementation
# from forex_ai.agents.framework.langgraph_integration import (
#     LangGraphWorkflow,
#     LangGraphNode,
#     ConditionalEdge
# )

__all__ = [
    # Base agent classes
    "BaseAgent",
    "AgentTool",
    "AgentMemoryEntry",
    "AgentMetrics",
    # LangGraph integration (to be implemented)
    # 'LangGraphWorkflow',
    # 'LangGraphNode',
    # 'ConditionalEdge'
]

"""
Agent framework for the Forex AI Trading System.

This package provides the agent-based architecture, including:
- Base agent framework (memory, tools, metrics)
- Specialized agents (fundamental analysis, sentiment analysis, etc.)
- LangGraph integration for complex workflows
- Agent communication and coordination
"""

# Import public interfaces from submodules
from forex_ai.agents.framework.agent import (
    BaseAgent,
    AgentTool,
    AgentMemoryEntry,
    AgentMetrics,
)

# Import specialized agents
from forex_ai.agents.fundamental_analysis import FundamentalAnalysisAgent
from forex_ai.agents.sentiment_analysis import SentimentAnalysisAgent

# These will be available after implementation
# from forex_ai.agents.risk_agent import RiskManagementAgent
# from forex_ai.agents.execution_agent import ExecutionAgent

# Import workflow
from forex_ai.agents.workflows.forex_analysis_workflow import ForexAnalysisWorkflow

__all__ = [
    # Base agent framework
    "BaseAgent",
    "AgentTool",
    "AgentMemoryEntry",
    "AgentMetrics",
    # Specialized agents
    "FundamentalAnalysisAgent",
    "SentimentAnalysisAgent",
    # 'RiskManagementAgent',
    # 'ExecutionAgent',
    # Workflows
    "ForexAnalysisWorkflow",
]

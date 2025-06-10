"""
Enhanced memory management for context-aware analysis.

This module provides an extension of the standard AgentMemory class that integrates
with AutoAgent memory capabilities and adds functionality for storing and
retrieving analysis contexts.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import json
from pathlib import Path
import uuid

from pydantic import BaseModel

from forex_ai.agents.base import AgentMemory
from AutoAgent.app_auto_agent.core.schema import Memory as AutoAgentMemory, Message


class AnalysisContext(BaseModel):
    """
    Model for storing analysis context.

    Attributes:
        analysis_id: Unique identifier for the analysis
        timestamp: When the analysis was performed
        pair: Currency pair analyzed
        timeframe: Time frame used for analysis
        analysis_type: Type of analysis (technical, fundamental, sentiment)
        findings: Analysis findings and insights
        confidence: Confidence level in the analysis (0.0 to 1.0)
        related_analyses: IDs of related analyses
        tags: Tags for categorizing the analysis
    """

    analysis_id: str
    timestamp: datetime
    pair: str
    timeframe: str
    analysis_type: str  # "technical", "fundamental", "sentiment", etc.
    findings: Dict[str, Any]
    confidence: float
    related_analyses: List[str] = []  # IDs of related analyses
    tags: List[str] = []


class EnhancedAgentMemory(AgentMemory):
    """
    Enhanced memory for context-aware analysis.
    Extends the base AgentMemory with AutoAgent capabilities.
    """

    def __init__(self, max_items: int = 1000, agent_id: str = None):
        """
        Initialize enhanced agent memory.

        Args:
            max_items: Maximum number of items to store in memory
            agent_id: ID of the agent owning this memory
        """
        super().__init__(max_items)
        self.agent_id = agent_id or str(uuid.uuid4())
        self.autoagent_memory = AutoAgentMemory()
        self.analysis_contexts: Dict[str, AnalysisContext] = {}

    def add_analysis_context(self, context: AnalysisContext) -> None:
        """
        Add analysis context to memory.

        Args:
            context: Analysis context to add
        """
        self.analysis_contexts[context.analysis_id] = context

        # Also add as a system message to AutoAgent memory for LLM context
        context_summary = (
            f"Previous analysis (ID: {context.analysis_id}) for {context.pair} on {context.timeframe} "
            f"timeframe from {context.timestamp.isoformat()}: "
            f"{json.dumps(context.findings, indent=2)}"
        )

        # Add to AutoAgent memory as system message
        system_message = Message.system_message(content=context_summary)
        self.autoagent_memory.add_message(system_message)

        # Also add to standard memory
        self.add_observation(
            {
                "type": "analysis_context",
                "context": context.dict(),
                "timestamp": context.timestamp.isoformat(),
            }
        )

    def get_related_analyses(
        self,
        pair: str,
        timeframe: str,
        analysis_type: Optional[str] = None,
        limit: int = 5,
    ) -> List[AnalysisContext]:
        """
        Get related analyses from memory.

        Args:
            pair: Currency pair
            timeframe: Time frame for analysis
            analysis_type: Type of analysis (optional filter)
            limit: Maximum number of analyses to return

        Returns:
            List of related analysis contexts
        """
        # Filter contexts by pair and timeframe
        filtered = [
            ctx
            for ctx in self.analysis_contexts.values()
            if ctx.pair == pair and ctx.timeframe == timeframe
        ]

        # Further filter by analysis type if specified
        if analysis_type:
            filtered = [ctx for ctx in filtered if ctx.analysis_type == analysis_type]

        # Sort by timestamp (newest first) and limit
        return sorted(filtered, key=lambda x: x.timestamp, reverse=True)[:limit]

    def get_sequential_context(self, limit: int = 10) -> str:
        """
        Get sequential context from recent analyses as formatted text.

        Args:
            limit: Maximum number of contexts to include

        Returns:
            Formatted context string for LLM prompting
        """
        # Get recent observations of type "analysis_context"
        contexts = [
            obs for obs in self.observations if obs.get("type") == "analysis_context"
        ]

        # Sort by timestamp (newest first) and limit
        contexts = sorted(contexts, key=lambda x: x.get("timestamp"), reverse=True)[
            :limit
        ]

        if not contexts:
            return "No previous analysis context available."

        # Format contexts for LLM consumption
        result = "Previous analyses in chronological order:\n\n"

        for i, ctx_data in enumerate(reversed(contexts), 1):
            ctx = ctx_data.get("context", {})
            result += f"{i}. {ctx.get('timestamp', 'Unknown time')}: {ctx.get('pair', 'Unknown pair')} ({ctx.get('timeframe', 'Unknown timeframe')})\n"
            result += f"   Type: {ctx.get('analysis_type', 'Unknown')}, Confidence: {ctx.get('confidence', 0)}\n"
            result += (
                f"   Findings: {json.dumps(ctx.get('findings', {}), indent=3)}\n\n"
            )

        return result

    def get_autoagent_memory(self) -> AutoAgentMemory:
        """
        Get the AutoAgent memory object.

        Returns:
            AutoAgent memory instance
        """
        return self.autoagent_memory

    def merge_with_autoagent_memory(self, messages: List[Message]) -> None:
        """
        Merge external AutoAgent messages into memory.

        Args:
            messages: List of AutoAgent messages to merge
        """
        for message in messages:
            self.autoagent_memory.add_message(message)

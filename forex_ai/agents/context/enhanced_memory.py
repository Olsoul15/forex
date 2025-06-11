"""
Enhanced memory management for context-aware analysis.

This module provides an extension of the standard AgentMemory class that integrates
with AutoAgent memory capabilities and adds functionality for storing and
retrieving analysis contexts.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
import json
from pathlib import Path
import uuid

from pydantic import BaseModel

from forex_ai.agents.base import AgentMemory
# from AutoAgent.app_auto_agent.core.schema import Memory as AutoAgentMemory, Message


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
        # self.autoagent_memory = AutoAgentMemory()
        self.analysis_contexts: Dict[str, AnalysisContext] = {}

    def add_analysis_context(self, context: AnalysisContext) -> None:
        """
        Add analysis context to memory.

        Args:
            context: Analysis context to add
        """
        self.analysis_contexts[context.analysis_id] = context

        # Also add to standard memory
        self.add_observation(
            {
                "type": "analysis_context",
                "context": context.dict(),
                "timestamp": context.timestamp.isoformat(),
            }
        )

    def get_analysis_context(self, analysis_id: str) -> Optional[AnalysisContext]:
        """
        Get analysis context from memory.

        Args:
            analysis_id: ID of the analysis context to retrieve

        Returns:
            Analysis context if found, None otherwise
        """
        return self.analysis_contexts.get(analysis_id)

    def get_related_analyses(
        self,
        pair: str,
        timeframe: str,
        limit: int = 5,
        max_days_lookback: int = 30,
    ) -> List[AnalysisContext]:
        """
        Get related analysis contexts based on pair and timeframe.

        Args:
            pair: Currency pair
            timeframe: Time frame
            limit: Maximum number of related contexts to return
            max_days_lookback: Maximum number of days to look back

        Returns:
            List of related analysis contexts
        """
        cutoff_date = datetime.now() - timedelta(days=max_days_lookback)
        related_contexts = []

        for context in self.analysis_contexts.values():
            if (
                context.pair == pair
                and context.timeframe == timeframe
                and context.timestamp >= cutoff_date
            ):
                related_contexts.append(context)

        # Sort by timestamp (most recent first) and return the top `limit`
        related_contexts.sort(key=lambda x: x.timestamp, reverse=True)
        return related_contexts[:limit]

    def get_contextual_summary(self, pair: str, timeframe: str) -> str:
        """
        Get a contextual summary from recent analyses.

        Args:
            pair: Currency pair
            timeframe: Time frame

        Returns:
            Contextual summary string
        """
        contexts = self.get_related_analyses(pair, timeframe)
        if not contexts:
            return "No recent analysis context available."

        summary = "Recent analysis summary:\n"
        for ctx in contexts:
            summary += (
                f"- {ctx.timestamp.isoformat()}: {ctx.analysis_type} analysis "
                f"with confidence {ctx.confidence:.2f}. "
                f"Findings: {json.dumps(ctx.findings)}\n"
            )

        return summary

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

    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """
        Save the memory to a file.

        Args:
            file_path: Path to the file where the memory will be saved
        """
        pass

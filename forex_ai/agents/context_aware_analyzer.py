"""
Context-aware analyzer agent for building on previous analysis findings.

This module provides a specialized agent that maintains context across analysis
sessions and can build upon previous findings to provide more nuanced insights.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime
import json
import uuid
import asyncio

from pydantic import BaseModel

from forex_ai.agents.base import BaseAgent, AgentTool
from forex_ai.agents.context.enhanced_memory import EnhancedAgentMemory, AnalysisContext
from forex_ai.agents.context.analysis_memory_manager import AnalysisMemoryManager
from forex_ai.agents.technical_analysis import TechnicalAnalysisAgent
from forex_ai.agents.framework.agent_types import AnalysisResult

# from AutoAgent.app_auto_agent.client import MantusClient
# from AutoAgent.app_auto_agent.core.schema import Message

logger = logging.getLogger(__name__)

class SimpleTool(AgentTool):
    def __init__(self, name: str, description: str, func: Callable):
        self._name = name
        self._description = description
        self.func = func

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def run(self, **kwargs) -> Any:
        return self.func(**kwargs)

class ContextAwareAnalyzer(BaseAgent):
    """
    Agent for performing context-aware analysis, building on previous findings.
    """

    def __init__(self, name: str = "ContextAnalyzer", config: Dict[str, Any] = None):
        """
        Initialize the context-aware analyzer.

        Args:
            name: Agent name
            config: Configuration dictionary
        """
        super().__init__(name, config or {})
        self.memory = EnhancedAgentMemory(agent_id=name)
        self.memory_manager = AnalysisMemoryManager()

        # Initialize AutoAgent client
        # self.mantus_client = MantusClient()

        # Initialize technical analysis agent
        self.technical_agent = TechnicalAnalysisAgent(
            name="TechnicalForContext",
            config=config.get("technical_agent", {}) if config else {},
        )

        # Set up tools for this agent
        self._initialize_tools()

    def _initialize_tools(self):
        """Initialize tools for this agent."""
        # Add tools specific to context-aware analysis
        self.add_tool(
            SimpleTool(
                name="retrieve_context",
                description="Retrieve relevant analysis context for a currency pair and timeframe",
                func=self.retrieve_context,
            )
        )

        self.add_tool(
            SimpleTool(
                name="store_context",
                description="Store analysis context for future reference",
                func=self.store_context,
            )
        )

    async def retrieve_context(
        self,
        pair: str,
        timeframe: str,
        analysis_types: Optional[List[str]] = None,
        days_ago: int = 30,
    ) -> Dict[str, Any]:
        """
        Retrieve relevant context for analysis.

        Args:
            pair: Currency pair
            timeframe: Time frame
            analysis_types: Types of analysis to include
            days_ago: Look back period in days

        Returns:
            Dictionary with context information
        """
        # Load contexts into memory
        loaded_count = await self.memory_manager.load_memory(
            self.memory, pair, timeframe, analysis_types, days_ago
        )

        # Get formatted context for LLM
        context_str = self.memory.get_sequential_context()

        return {
            "loaded_contexts": loaded_count,
            "context_summary": context_str,
            "pair": pair,
            "timeframe": timeframe,
        }

    async def store_context(self, analysis_context: Dict[str, Any]) -> str:
        """
        Store analysis context for future reference.

        Args:
            analysis_context: Analysis context dictionary

        Returns:
            ID of the stored context
        """
        # Ensure analysis_id exists
        if "analysis_id" not in analysis_context:
            analysis_context["analysis_id"] = str(uuid.uuid4())

        # Ensure timestamp exists
        if "timestamp" not in analysis_context:
            analysis_context["timestamp"] = datetime.now().isoformat()

        # Create AnalysisContext object
        context = AnalysisContext(**analysis_context)

        # Add to memory
        self.memory.add_analysis_context(context)

        # Save to database
        context_id = await self.memory_manager.save_context(context)

        return context_id

    async def analyze_with_context(
        self,
        pair: str,
        timeframe: str,
        analysis_type: str = "technical",
        include_context: bool = True,
    ) -> AnalysisResult:
        """
        Perform analysis with context from previous analyses.

        Args:
            pair: Currency pair
            timeframe: Time frame
            analysis_type: Type of analysis to perform
            include_context: Whether to include previous context

        Returns:
            Analysis result
        """
        # Step 1: Retrieve context if needed
        context_info = {}
        if include_context:
            context_info = await self.retrieve_context(
                pair, timeframe, [analysis_type], days_ago=30
            )

        # Step 2: Perform base analysis using technical agent
        analysis_result = self.technical_agent.analyze(pair, timeframe)

        # If we don't have context or the technical agent already failed, return as is
        if not include_context or not analysis_result.success:
            return analysis_result

        # Step 3: Enhance analysis with context using AutoAgent
        # For now, we will just return the technical analysis result.
        # The integration with an external analysis API will be done later.
        return analysis_result

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data and perform context-aware analysis.

        Args:
            input_data: Input data for analysis

        Returns:
            Analysis results
        """
        # Extract parameters from input
        pair = input_data.get("pair")
        timeframe = input_data.get("timeframe")
        analysis_type = input_data.get("analysis_type", "technical")
        include_context = input_data.get("include_context", True)

        if not pair or not timeframe:
            return {
                "success": False,
                "message": "Missing required parameters: pair and timeframe",
            }

        # For synchronous operation, we'll return just the technical analysis results
        result = self.technical_agent.analyze(pair, timeframe)

        return result.dict() if hasattr(result, "dict") else result

"""
Context-aware analyzer agent for building on previous analysis findings.

This module provides a specialized agent that maintains context across analysis
sessions and can build upon previous findings to provide more nuanced insights.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import json
import uuid
import asyncio

from pydantic import BaseModel

from forex_ai.agents.base import BaseAgent
from forex_ai.agents.context.enhanced_memory import EnhancedAgentMemory, AnalysisContext
from forex_ai.agents.context.analysis_memory_manager import AnalysisMemoryManager
from forex_ai.agents.technical_analysis import TechnicalAnalysisAgent
from forex_ai.agents.framework.agent_types import AnalysisResult

from AutoAgent.app_auto_agent.client import MantusClient
from AutoAgent.app_auto_agent.core.schema import Message

logger = logging.getLogger(__name__)


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
        self.mantus_client = MantusClient()

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
            {
                "name": "retrieve_context",
                "description": "Retrieve relevant analysis context for a currency pair and timeframe",
                "func": self.retrieve_context,
            }
        )

        self.add_tool(
            {
                "name": "store_context",
                "description": "Store analysis context for future reference",
                "func": self.store_context,
            }
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
        try:
            # Create messages for AutoAgent
            messages = [
                Message.system_message(
                    content=f"""You are an expert forex analysis system. 
                    Analyze the technical indicators and patterns for {pair} on {timeframe} timeframe.
                    Build upon previous analyses to identify emerging trends and changes.
                    
                    Previous context:
                    {context_info.get('context_summary', 'No previous context available.')}
                    """
                ),
                Message.user_message(
                    content=f"""
                    I need a context-aware analysis for {pair} on {timeframe} timeframe.
                    
                    Here is the current technical analysis data:
                    {json.dumps(analysis_result.data, indent=2)}
                    
                    Based on this new data AND the previous analyses, what insights can you provide?
                    Specifically highlight:
                    1. What has changed since previous analyses
                    2. Emerging patterns or trends 
                    3. Confirmation or invalidation of previous hypotheses
                    4. New predictions based on the combined context
                    
                    Format your response as JSON with the following structure:
                    {{
                        "summary": "Brief summary of analysis",
                        "key_changes": ["list of key changes from previous analysis"],
                        "confirmed_patterns": ["patterns confirmed by new data"],
                        "invalidated_patterns": ["patterns invalidated by new data"],
                        "emerging_trends": ["new trends identified"], 
                        "prediction": "prediction about price movement",
                        "confidence": 0.0 to 1.0,
                        "supporting_evidence": ["evidence supporting prediction"]
                    }}
                    """
                ),
            ]

            # Call AutoAgent for contextual analysis
            response = await self.mantus_client.complete(
                messages=messages,
                model="gpt-4",  # Can be configured based on needs
                response_format={"type": "json_object"},
            )

            # Extract contextual insights
            contextual_data = json.loads(response.content)

            # Step 4: Store the enhanced analysis for future reference
            new_context = {
                "analysis_id": str(uuid.uuid4()),
                "timestamp": datetime.now(),
                "pair": pair,
                "timeframe": timeframe,
                "analysis_type": analysis_type,
                "findings": contextual_data,
                "confidence": contextual_data.get("confidence", 0.5),
                "related_analyses": [
                    a.analysis_id
                    for a in self.memory.get_related_analyses(pair, timeframe)
                ],
                "tags": ["context-aware", analysis_type, pair, timeframe],
            }

            # Store context for future use
            await self.store_context(new_context)

            # Update the analysis result with contextual insights
            enhanced_result = AnalysisResult(
                success=True,
                pair=pair,
                timeframe=timeframe,
                data={
                    **analysis_result.data,  # Original analysis data
                    "contextual_insights": contextual_data,  # Enhanced with context
                },
                message="Context-aware analysis completed successfully",
            )

            return enhanced_result

        except Exception as e:
            logger.error(f"Error performing context-aware analysis: {str(e)}")
            # Fall back to the original analysis if context enhancement fails
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

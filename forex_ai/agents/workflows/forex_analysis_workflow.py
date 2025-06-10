"""
Forex Analysis Workflow for the AI Forex Trading System.

This module defines a LangGraph workflow that orchestrates multiple specialized agents
to provide comprehensive forex market analysis, including fundamental factors,
sentiment analysis, and technical indicators.
"""

from typing import Dict, List, Optional, Any, Union, Callable
import logging
from datetime import datetime, timedelta
from forex_ai.agents.framework.agent import BaseAgent
from ..framework.langgraph_integration import (
    LangGraphWorkflow,
    LangGraphNode,
    ConditionalEdge,
)
from ..fundamental_analysis import FundamentalAnalysisAgent
from ..sentiment_analysis import SentimentAnalysisAgent
from ..technical_analysis import TechnicalAnalysisAgent
from forex_ai.utils.logging import get_logger
from forex_ai.exceptions import WorkflowError

logger = get_logger(__name__)


class ForexAnalysisWorkflow:
    """
    Workflow that coordinates technical, fundamental, and sentiment analysis.

    This workflow orchestrates multiple specialized agents to provide
    comprehensive analysis of forex markets, combining market data analysis,
    economic factors, and news sentiment.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the forex analysis workflow.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Initialize agents
        self.fundamental_agent = FundamentalAnalysisAgent(self.config)
        self.sentiment_agent = SentimentAnalysisAgent(self.config)
        self.technical_agent = TechnicalAnalysisAgent(self.config)

        # Initialize workflow
        self.workflow = LangGraphWorkflow(
            name="forex_analysis",
            description="Comprehensive forex market analysis combining technical, fundamental, and sentiment factors",
        )

        # Configure workflow
        self._setup_workflow()

        logger.info("Forex analysis workflow initialized")

    def _setup_workflow(self) -> None:
        """Set up the workflow with nodes and edges."""
        # Create nodes
        input_node = LangGraphNode(
            id="input",
            name="Input Node",
            description="Processes input data and dispatches to appropriate analysts",
            handler=self._process_input,
        )

        fundamental_node = LangGraphNode(
            id="fundamental",
            name="Fundamental Analysis",
            description="Analyzes economic data and central bank policies",
            handler=self._run_fundamental_analysis,
        )

        sentiment_node = LangGraphNode(
            id="sentiment",
            name="Sentiment Analysis",
            description="Analyzes news sentiment and market impact",
            handler=self._run_sentiment_analysis,
        )

        technical_node = LangGraphNode(
            id="technical",
            name="Technical Analysis",
            description="Analyzes market data using technical indicators and patterns",
            handler=self._run_technical_analysis,
        )

        consolidation_node = LangGraphNode(
            id="consolidate",
            name="Analysis Consolidation",
            description="Combines results from different analysis types",
            handler=self._consolidate_analysis,
        )

        # Add nodes to workflow
        self.workflow.add_node(input_node)
        self.workflow.add_node(fundamental_node)
        self.workflow.add_node(sentiment_node)
        self.workflow.add_node(technical_node)
        self.workflow.add_node(consolidation_node)

        # Define edges
        # Input node can go to the various analysis nodes depending on the request
        self.workflow.add_edge(
            "input",
            ConditionalEdge(
                target_node="sentiment",
                condition=lambda state: state.get("run_sentiment", True),
            ),
        )

        self.workflow.add_edge(
            "input",
            ConditionalEdge(
                target_node="fundamental",
                condition=lambda state: state.get("run_fundamental", True),
            ),
        )

        self.workflow.add_edge(
            "input",
            ConditionalEdge(
                target_node="technical",
                condition=lambda state: state.get("run_technical", True),
            ),
        )

        # All analysis nodes feed into the consolidation node
        self.workflow.add_edge("sentiment", "consolidate")
        self.workflow.add_edge("fundamental", "consolidate")
        self.workflow.add_edge("technical", "consolidate")

        # Consolidation node is the final step
        self.workflow.add_edge("consolidate", "END")

        logger.info("Forex analysis workflow structure configured")

    def _process_input(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data and prepare for analysis.

        Args:
            state: Current workflow state

        Returns:
            Updated state with flags for which analyses to run
        """
        input_data = state.get("input", {})
        logger.info(f"Processing input data for {input_data.get('symbol', 'unknown')}")

        # Determine which analyses to run
        analysis_types = input_data.get(
            "analysis_types", ["fundamental", "sentiment", "technical"]
        )

        # Set flags in state
        state["run_fundamental"] = "fundamental" in analysis_types
        state["run_sentiment"] = "sentiment" in analysis_types
        state["run_technical"] = "technical" in analysis_types

        # Initialize results containers
        state["fundamental_result"] = None
        state["sentiment_result"] = None
        state["technical_result"] = None

        return state

    def _run_fundamental_analysis(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run fundamental analysis.

        Args:
            state: Current workflow state

        Returns:
            Updated state with fundamental analysis results
        """
        input_data = state.get("input", {})
        logger.info(
            f"Running fundamental analysis for {input_data.get('symbol', 'unknown')}"
        )

        try:
            # Process with fundamental agent
            result = self.fundamental_agent.process(input_data)
            state["fundamental_result"] = result
            logger.info("Fundamental analysis completed successfully")
        except Exception as e:
            logger.error(f"Fundamental analysis failed: {str(e)}")
            state["fundamental_result"] = {"error": str(e)}

        return state

    def _run_sentiment_analysis(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run sentiment analysis.

        Args:
            state: Current workflow state

        Returns:
            Updated state with sentiment analysis results
        """
        input_data = state.get("input", {})
        logger.info(
            f"Running sentiment analysis for {input_data.get('symbol', 'unknown')}"
        )

        try:
            # Process with sentiment agent
            result = self.sentiment_agent.process(input_data)
            state["sentiment_result"] = result
            logger.info("Sentiment analysis completed successfully")
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {str(e)}")
            state["sentiment_result"] = {"error": str(e)}

        return state

    def _run_technical_analysis(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run technical analysis.

        Args:
            state: Current workflow state

        Returns:
            Updated state with technical analysis results
        """
        input_data = state.get("input", {})
        logger.info(
            f"Running technical analysis for {input_data.get('symbol', 'unknown')}"
        )

        try:
            # Process with technical agent - note it's synchronous, not async
            result = self.technical_agent.process(input_data)
            state["technical_result"] = result
            logger.info("Technical analysis completed successfully")
        except Exception as e:
            logger.error(f"Technical analysis failed: {str(e)}")
            state["technical_result"] = {"error": str(e)}

        return state

    def _consolidate_analysis(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Consolidate analysis results from different sources.

        Args:
            state: Current workflow state

        Returns:
            Updated state with consolidated analysis
        """
        symbol = state.get("input", {}).get("symbol", "unknown")
        logger.info(f"Consolidating analysis results for {symbol}")

        fundamental_result = state.get("fundamental_result", {})
        sentiment_result = state.get("sentiment_result", {})

        # Create consolidated result
        consolidated = {
            "symbol": symbol,
            "analysis_time": datetime.now().isoformat(),
            "analysis_types": [],
        }

        # Add fundamental analysis if available
        if fundamental_result and "error" not in fundamental_result:
            consolidated["fundamental"] = fundamental_result
            consolidated["analysis_types"].append("fundamental")

        # Add sentiment analysis if available
        if sentiment_result and "error" not in sentiment_result:
            consolidated["sentiment"] = sentiment_result
            consolidated["analysis_types"].append("sentiment")

        # Add technical analysis if available
        if (
            state.get("technical_result", {})
            and "error" not in state["technical_result"]
        ):
            consolidated["technical"] = state["technical_result"]
            consolidated["analysis_types"].append("technical")

        # Calculate overall market view
        # Initialize weights for each analysis type (will adjust based on confidence later)
        weights = {"fundamental": 0.5, "sentiment": 0.5}

        # Track available analyses and their confidence levels
        analysis_confidence = {}

        # Get outlook and confidence from each analysis type
        if "fundamental" in consolidated["analysis_types"]:
            # Get fundamental outlook and confidence
            fundamental_outlook = fundamental_result.get("outlook", "neutral").lower()
            fundamental_confidence = fundamental_result.get("confidence", 0.5)
            analysis_confidence["fundamental"] = fundamental_confidence

        if "sentiment" in consolidated["analysis_types"]:
            # Map sentiment to outlook format
            sentiment_label = sentiment_result.get("sentiment", "neutral").lower()
            sentiment_mapping = {
                "bullish": "bullish",
                "bearish": "bearish",
                "neutral": "neutral",
            }
            sentiment_outlook = sentiment_mapping.get(sentiment_label, "neutral")
            sentiment_confidence = sentiment_result.get("confidence", 0.5)
            analysis_confidence["sentiment"] = sentiment_confidence

        # Adjust weights based on confidence if we have multiple analyses
        if len(analysis_confidence) > 1:
            # Normalize weights based on confidence
            total_confidence = sum(analysis_confidence.values())
            if total_confidence > 0:
                for analysis_type, confidence in analysis_confidence.items():
                    weights[analysis_type] = confidence / total_confidence

        # Store the weights in the result
        for analysis_type, weight in weights.items():
            if analysis_type in consolidated["analysis_types"]:
                consolidated[f"{analysis_type}_weight"] = weight

        # Calculate weighted score for overall outlook
        if consolidated["analysis_types"]:
            # Convert outlooks to numeric scale (-1 to 1)
            outlook_scores = {"bearish": -0.5, "neutral": 0.0, "bullish": 0.5}

            # Calculate score for each analysis type
            weighted_scores = {}

            if "fundamental" in consolidated["analysis_types"]:
                fundamental_score = outlook_scores.get(fundamental_outlook, 0)
                weighted_scores["fundamental"] = (
                    fundamental_score * weights["fundamental"]
                )

            if "sentiment" in consolidated["analysis_types"]:
                sentiment_score = outlook_scores.get(sentiment_outlook, 0)
                weighted_scores["sentiment"] = sentiment_score * weights["sentiment"]

            # Calculate combined score
            combined_score = sum(weighted_scores.values())

            # Convert back to text outlook
            if combined_score <= -0.25:
                overall_outlook = "bearish"
            elif combined_score <= 0.25:
                overall_outlook = "neutral"
            else:
                overall_outlook = "bullish"
        else:
            # No analysis available
            overall_outlook = "neutral"

        # Add overall outlook to consolidated result
        consolidated["overall_outlook"] = overall_outlook

        # Add score for reference
        if "analysis_types" in consolidated and consolidated["analysis_types"]:
            consolidated["outlook_score"] = combined_score

        # Generate summary
        consolidated["summary"] = self._generate_summary(consolidated)

        # Set result in state
        state["result"] = consolidated

        return state

    def _generate_summary(self, consolidated: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary of the consolidated analysis.

        Args:
            consolidated: Consolidated analysis result

        Returns:
            Summary text
        """
        symbol = consolidated.get("symbol", "unknown")
        overall_outlook = consolidated.get("overall_outlook", "neutral").title()
        analysis_types = consolidated.get("analysis_types", [])

        # Create summary
        summary = f"Forex Analysis Summary for {symbol}\n"
        summary += f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        summary += f"Overall Market Outlook: {overall_outlook}\n"
        summary += f"Analysis Types: {', '.join(analysis_types)}\n\n"

        # Add fundamental analysis
        if "fundamental" in consolidated:
            fundamental = consolidated["fundamental"]

            summary += "Fundamental Analysis:\n"

            # Add economic indicators if available
            if "economic_indicators" in fundamental:
                summary += "- Key Economic Indicators:\n"
                for indicator in fundamental["economic_indicators"][
                    :3
                ]:  # Top 3 indicators
                    name = indicator.get("name", "Unknown")
                    value = indicator.get("value", "N/A")
                    impact = indicator.get("impact", "neutral").title()
                    summary += f"  * {name}: {value} (Impact: {impact})\n"

            # Add central bank policy if available
            if "central_bank_policy" in fundamental:
                policy = fundamental["central_bank_policy"]
                summary += f"- Central Bank Policy: {policy.get('stance', 'Unknown')}\n"
                if "next_meeting" in policy:
                    summary += f"- Next Meeting: {policy['next_meeting']}\n"

            summary += "\n"

        # Add sentiment analysis
        if "sentiment" in consolidated:
            sentiment = consolidated["sentiment"]
            sentiment_outlook = sentiment.get("sentiment", "neutral").title()
            confidence = sentiment.get("confidence", 0.0)

            summary += "Sentiment Analysis:\n"
            summary += (
                f"- Outlook: {sentiment_outlook} (Confidence: {confidence:.2f})\n"
            )
            summary += f"- Based on {sentiment.get('news_count', 0)} news articles\n"

            # Add top news if available
            top_news = sentiment.get("top_news", [])
            if top_news:
                summary += "- Top News:\n"
                for i, news in enumerate(top_news[:2], 1):  # Top 2 news items
                    title = news.get("title", "No title")
                    summary += f"  {i}. {title}\n"

            # Add impact prediction if available
            if "impact" in sentiment:
                impact = sentiment["impact"]
                summary += "- Market Impact Prediction:\n"
                summary += f"  * Probability: {impact.get('probability', 0):.2f}\n"
                summary += f"  * Magnitude: {impact.get('magnitude', 0):.2f}\n"

                affected_pairs = impact.get("affected_pairs", [])
                if affected_pairs:
                    summary += f"  * Affected Pairs: {', '.join(affected_pairs[:3])}\n"

            summary += "\n"

        # Add technical analysis
        if "technical" in consolidated:
            technical = consolidated["technical"]

            summary += "Technical Analysis:\n"

            # Add technical indicators if available
            if "technical_indicators" in technical:
                summary += "- Key Technical Indicators:\n"
                for indicator in technical["technical_indicators"][
                    :3
                ]:  # Top 3 indicators
                    name = indicator.get("name", "Unknown")
                    value = indicator.get("value", "N/A")
                    summary += f"  * {name}: {value}\n"

            summary += "\n"

        # Analysis weights if available
        weights = {}
        for analysis_type in ["fundamental", "sentiment", "technical"]:
            weight_key = f"{analysis_type}_weight"
            if weight_key in consolidated:
                weights[analysis_type] = consolidated[weight_key]

        if weights:
            summary += "Analysis Weighting:\n"
            for analysis_type, weight in weights.items():
                summary += f"- {analysis_type.title()}: {weight:.2f}\n"
            summary += "\n"

        # Add trading recommendation
        summary += "Trading Recommendation:\n"

        if overall_outlook == "Bullish":
            summary += "- Consider LONG positions with appropriate risk management\n"
        elif overall_outlook == "Neutral":
            summary += (
                "- Consider ranging market strategies or staying on the sidelines\n"
            )
        elif overall_outlook == "Bearish":
            summary += (
                "- Bias toward SHORT positions, but carefully assess entry points\n"
            )

        summary += "\nNote: This analysis is for informational purposes only and should not be considered financial advice."

        return summary

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the forex analysis workflow.

        Args:
            input_data: Input data for the workflow

        Returns:
            Consolidated analysis result
        """
        try:
            # Validate input
            if "symbol" not in input_data:
                raise WorkflowError("Symbol is required for forex analysis")

            # Initialize workflow state
            state = {"input": input_data}

            # Execute workflow
            final_state = self.workflow.execute(state)

            # Return result
            return final_state.get("result", {})

        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            return {
                "error": str(e),
                "symbol": input_data.get("symbol", "unknown"),
                "analysis_time": datetime.now().isoformat(),
            }

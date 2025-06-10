"""
AutoAgent tool wrappers for fundamental analysis modules in the AI Forex system.

This module provides AutoAgent tool wrappers around existing fundamental
analysis modules in the AI Forex system, allowing them to be used within
AutoAgent workflows for cross-analysis coordination.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import json
import logging

from AutoAgent.app_auto_agent.tool.base import (
    Tool,
    ToolDescription,
    ToolParameter,
    ToolContext,
    ToolResult,
)

from forex_ai.agents.fundamental_analysis import (
    FundamentalAnalysisAgent,
    EconomicCalendarAnalyzer,
)
from forex_ai.agents.news import NewsAnalysisAgent
from forex_ai.utils.logging import get_logger

logger = get_logger(__name__)


class FundamentalAnalysisTool(Tool):
    """Tool wrapper for fundamental analysis capabilities."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the fundamental analysis tool.

        Args:
            config: Optional configuration dictionary
        """
        super().__init__()
        self.config = config or {}
        self.fundamental_agent = FundamentalAnalysisAgent(self.config)

        # Define tool metadata
        self.description = ToolDescription(
            name="fundamental_analysis",
            description="Performs fundamental analysis on forex pairs",
            version="1.0.0",
        )

        # Define parameters
        self.parameters = [
            ToolParameter(
                name="pair",
                description="Currency pair to analyze (e.g., 'EUR/USD')",
                required=True,
                type="string",
            ),
            ToolParameter(
                name="timeframe",
                description="Analysis timeframe (e.g., 'daily', 'weekly')",
                required=True,
                type="string",
            ),
            ToolParameter(
                name="analysis_types",
                description="Types of fundamental analysis to perform",
                required=False,
                type="array",
                default=["economic_indicators", "central_bank", "geopolitical"],
            ),
            ToolParameter(
                name="start_date",
                description="Start date for analysis (ISO format)",
                required=False,
                type="string",
                default=None,
            ),
            ToolParameter(
                name="end_date",
                description="End date for analysis (ISO format)",
                required=False,
                type="string",
                default=None,
            ),
        ]

    async def execute(self, context: ToolContext) -> ToolResult:
        """
        Execute fundamental analysis.

        Args:
            context: Tool execution context with parameters

        Returns:
            Analysis results
        """
        # Extract parameters
        pair = context.get_parameter("pair")
        timeframe = context.get_parameter("timeframe")
        analysis_types = context.get_parameter(
            "analysis_types", ["economic_indicators", "central_bank", "geopolitical"]
        )
        start_date = context.get_parameter("start_date")
        end_date = context.get_parameter("end_date")

        # Set default dates if not provided
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")

        if not start_date:
            # Default to last 30 days
            start_date = (
                datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=30)
            ).strftime("%Y-%m-%d")

        try:
            logger.info(
                f"Performing fundamental analysis for {pair} on {timeframe} timeframe"
            )

            # Execute analysis
            analysis_result = await self.fundamental_agent.analyze(
                pair=pair,
                timeframe=timeframe,
                analysis_types=analysis_types,
                start_date=start_date,
                end_date=end_date,
            )

            # Prepare response
            return ToolResult(
                success=True,
                result={
                    "pair": pair,
                    "timeframe": timeframe,
                    "analysis_types": analysis_types,
                    "start_date": start_date,
                    "end_date": end_date,
                    "fundamental_indicators": analysis_result.get("indicators", {}),
                    "economic_events": analysis_result.get("economic_events", []),
                    "central_bank_analysis": analysis_result.get("central_bank", {}),
                    "geopolitical_factors": analysis_result.get("geopolitical", {}),
                    "strength_indicators": analysis_result.get(
                        "strength_indicators", {}
                    ),
                    "outlook": analysis_result.get("outlook", {}),
                },
            )

        except Exception as e:
            logger.error(f"Error performing fundamental analysis: {str(e)}")

            return ToolResult(
                success=False, error=f"Error performing fundamental analysis: {str(e)}"
            )


class EconomicCalendarTool(Tool):
    """Tool wrapper for economic calendar analysis capabilities."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the economic calendar tool.

        Args:
            config: Optional configuration dictionary
        """
        super().__init__()
        self.config = config or {}
        self.calendar_analyzer = EconomicCalendarAnalyzer(self.config)

        # Define tool metadata
        self.description = ToolDescription(
            name="economic_calendar",
            description="Analyzes upcoming economic events from economic calendars",
            version="1.0.0",
        )

        # Define parameters
        self.parameters = [
            ToolParameter(
                name="currencies",
                description="List of currencies to analyze (e.g., ['EUR', 'USD'])",
                required=True,
                type="array",
            ),
            ToolParameter(
                name="days_ahead",
                description="Number of days to look ahead",
                required=False,
                type="integer",
                default=7,
            ),
            ToolParameter(
                name="impact_threshold",
                description="Minimum impact threshold (1-3, where 3 is highest impact)",
                required=False,
                type="integer",
                default=2,
            ),
        ]

    async def execute(self, context: ToolContext) -> ToolResult:
        """
        Execute economic calendar analysis.

        Args:
            context: Tool execution context with parameters

        Returns:
            Economic calendar analysis results
        """
        # Extract parameters
        currencies = context.get_parameter("currencies")
        days_ahead = context.get_parameter("days_ahead", 7)
        impact_threshold = context.get_parameter("impact_threshold", 2)

        try:
            logger.info(f"Analyzing economic calendar for currencies: {currencies}")

            # Execute analysis
            calendar_results = await self.calendar_analyzer.analyze_upcoming_events(
                currencies=currencies,
                days_ahead=days_ahead,
                impact_threshold=impact_threshold,
            )

            # Prepare response
            return ToolResult(
                success=True,
                result={
                    "currencies": currencies,
                    "days_ahead": days_ahead,
                    "impact_threshold": impact_threshold,
                    "events": calendar_results.get("events", []),
                    "high_impact_count": calendar_results.get("high_impact_count", 0),
                    "event_by_currency": calendar_results.get("events_by_currency", {}),
                    "potential_volatility": calendar_results.get(
                        "potential_volatility", {}
                    ),
                    "trading_recommendations": calendar_results.get(
                        "trading_recommendations", []
                    ),
                },
            )

        except Exception as e:
            logger.error(f"Error analyzing economic calendar: {str(e)}")

            return ToolResult(
                success=False, error=f"Error analyzing economic calendar: {str(e)}"
            )


class NewsAnalysisTool(Tool):
    """Tool wrapper for news analysis capabilities."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the news analysis tool.

        Args:
            config: Optional configuration dictionary
        """
        super().__init__()
        self.config = config or {}
        self.news_agent = NewsAnalysisAgent(self.config)

        # Define tool metadata
        self.description = ToolDescription(
            name="news_analysis",
            description="Analyzes financial news for market sentiment and impact",
            version="1.0.0",
        )

        # Define parameters
        self.parameters = [
            ToolParameter(
                name="pair",
                description="Currency pair to analyze news for (e.g., 'EUR/USD')",
                required=True,
                type="string",
            ),
            ToolParameter(
                name="hours_ago",
                description="How many hours back to analyze news",
                required=False,
                type="integer",
                default=24,
            ),
            ToolParameter(
                name="max_articles",
                description="Maximum number of articles to analyze",
                required=False,
                type="integer",
                default=20,
            ),
            ToolParameter(
                name="include_social",
                description="Whether to include social media in analysis",
                required=False,
                type="boolean",
                default=True,
            ),
        ]

    async def execute(self, context: ToolContext) -> ToolResult:
        """
        Execute news analysis.

        Args:
            context: Tool execution context with parameters

        Returns:
            News analysis results
        """
        # Extract parameters
        pair = context.get_parameter("pair")
        hours_ago = context.get_parameter("hours_ago", 24)
        max_articles = context.get_parameter("max_articles", 20)
        include_social = context.get_parameter("include_social", True)

        try:
            logger.info(f"Analyzing news for {pair} from the past {hours_ago} hours")

            # Extract currencies from pair
            currencies = pair.replace("/", "").split()
            if len(currencies) == 1 and "/" in pair:
                currencies = pair.split("/")

            # Execute analysis
            news_results = await self.news_agent.analyze_recent_news(
                currencies=currencies,
                hours_ago=hours_ago,
                max_articles=max_articles,
                include_social=include_social,
            )

            # Prepare response
            return ToolResult(
                success=True,
                result={
                    "pair": pair,
                    "hours_ago": hours_ago,
                    "article_count": news_results.get("article_count", 0),
                    "overall_sentiment": news_results.get("overall_sentiment", {}),
                    "sentiment_breakdown": news_results.get("sentiment_breakdown", {}),
                    "key_topics": news_results.get("key_topics", []),
                    "trending_keywords": news_results.get("trending_keywords", []),
                    "impactful_articles": news_results.get("impactful_articles", []),
                    "social_sentiment": (
                        news_results.get("social_sentiment", {})
                        if include_social
                        else None
                    ),
                    "trading_implications": news_results.get(
                        "trading_implications", {}
                    ),
                },
            )

        except Exception as e:
            logger.error(f"Error analyzing news: {str(e)}")

            return ToolResult(success=False, error=f"Error analyzing news: {str(e)}")


def get_fundamental_tools(config: Optional[Dict[str, Any]] = None) -> List[Tool]:
    """
    Get all fundamental analysis tools.

    Args:
        config: Optional configuration dictionary

    Returns:
        List of fundamental analysis tools
    """
    return [
        FundamentalAnalysisTool(config),
        EconomicCalendarTool(config),
        NewsAnalysisTool(config),
    ]

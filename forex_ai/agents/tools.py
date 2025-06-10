"""
Agent Tool Implementations for the AI Forex Trading System.

This module provides concrete implementations of tools that agents
can use to interact with the system and external resources.
"""

from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
from pathlib import Path

from .base import AgentTool
from forex_ai.utils.logging import get_logger
from forex_ai.exceptions import AgentToolError
from forex_ai.analysis.technical.indicators import apply_indicators
from forex_ai.analysis.technical.patterns import (
    detect_candlestick_patterns,
    PatternType,
)

logger = get_logger(__name__)


class MarketDataTool(AgentTool):
    """Tool for fetching and querying market data."""

    def __init__(self, data_connector):
        """
        Initialize the market data tool.

        Args:
            data_connector: Connector to market data source
        """
        self.data_connector = data_connector

    @property
    def name(self) -> str:
        """Get the name of the tool."""
        return "market_data"

    @property
    def description(self) -> str:
        """Get a description of what the tool does."""
        return "Fetches historical and real-time market data for analysis"

    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        """Get the parameters the tool accepts."""
        return {
            "symbol": {
                "type": "string",
                "description": "Trading symbol (e.g., 'EUR/USD')",
                "required": True,
            },
            "timeframe": {
                "type": "string",
                "description": "Timeframe (e.g., '1h', '4h', '1d')",
                "required": True,
            },
            "start_date": {
                "type": "string",
                "description": "Start date in ISO format",
                "required": False,
            },
            "end_date": {
                "type": "string",
                "description": "End date in ISO format",
                "required": False,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of candles to fetch",
                "required": False,
            },
        }

    def run(
        self,
        symbol: str,
        timeframe: str,
        start_date: str = None,
        end_date: str = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch market data.

        Args:
            symbol: Trading symbol (e.g., 'EUR/USD')
            timeframe: Timeframe (e.g., '1h', '4h', '1d')
            start_date: Start date in ISO format
            end_date: End date in ISO format
            limit: Maximum number of candles to fetch

        Returns:
            DataFrame with OHLCV data

        Raises:
            AgentToolError: If data fetching fails
        """
        try:
            # Convert dates if provided
            start = datetime.fromisoformat(start_date) if start_date else None
            end = datetime.fromisoformat(end_date) if end_date else None

            # Fetch data from connector
            data = self.data_connector.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start,
                end_date=end,
                limit=limit,
            )

            return data
        except Exception as e:
            logger.error(f"Error fetching market data: {str(e)}")
            raise AgentToolError(f"Failed to fetch market data: {str(e)}") from e


class IndicatorTool(AgentTool):
    """Tool for calculating technical indicators."""

    @property
    def name(self) -> str:
        """Get the name of the tool."""
        return "indicators"

    @property
    def description(self) -> str:
        """Get a description of what the tool does."""
        return "Calculates technical indicators on market data"

    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        """Get the parameters the tool accepts."""
        return {
            "data": {
                "type": "dataframe",
                "description": "Market data as a pandas DataFrame",
                "required": True,
            },
            "indicators": {
                "type": "list",
                "description": "List of indicators to calculate",
                "required": True,
            },
            "params": {
                "type": "dict",
                "description": "Parameters for indicators",
                "required": False,
            },
        }

    def run(
        self,
        data: pd.DataFrame,
        indicators: List[str],
        params: Dict[str, Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Calculate technical indicators.

        Args:
            data: Market data as a pandas DataFrame
            indicators: List of indicators to calculate
            params: Parameters for indicators

        Returns:
            DataFrame with calculated indicators

        Raises:
            AgentToolError: If indicator calculation fails
        """
        try:
            # Use the indicators module to calculate indicators
            result = apply_indicators(data, indicators, params or {})
            return result
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            raise AgentToolError(f"Failed to calculate indicators: {str(e)}") from e


class PatternRecognitionTool(AgentTool):
    """Tool for detecting patterns in market data."""

    @property
    def name(self) -> str:
        """Get the name of the tool."""
        return "pattern_recognition"

    @property
    def description(self) -> str:
        """Get a description of what the tool does."""
        return "Detects candlestick and chart patterns in market data"

    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        """Get the parameters the tool accepts."""
        return {
            "data": {
                "type": "dataframe",
                "description": "Market data as a pandas DataFrame",
                "required": True,
            },
            "patterns": {
                "type": "list",
                "description": "List of pattern types to detect",
                "required": False,
            },
            "min_confidence": {
                "type": "float",
                "description": "Minimum confidence threshold",
                "required": False,
            },
        }

    def run(
        self,
        data: pd.DataFrame,
        patterns: List[str] = None,
        min_confidence: float = 0.6,
    ) -> List[Dict[str, Any]]:
        """
        Detect patterns in market data.

        Args:
            data: Market data as a pandas DataFrame
            patterns: List of pattern types to detect
            min_confidence: Minimum confidence threshold

        Returns:
            List of detected patterns

        Raises:
            AgentToolError: If pattern detection fails
        """
        try:
            # Convert string pattern names to PatternType enums
            pattern_types = None
            if patterns:
                pattern_types = []
                for p in patterns:
                    try:
                        pattern_types.append(PatternType[p.upper()])
                    except KeyError:
                        logger.warning(f"Invalid pattern type: {p}")

            # Detect patterns
            detected_patterns = detect_candlestick_patterns(data, pattern_types)

            # Filter by confidence
            filtered_patterns = [
                {
                    "type": p.pattern_type.value,
                    "direction": p.direction.value,
                    "confidence": p.confidence,
                    "index": p.index,
                    "date": (
                        data.index[p.index]
                        if hasattr(data.index, "__getitem__")
                        else None
                    ),
                    "price": data.iloc[p.index]["close"],
                    "additional_info": p.additional_info,
                }
                for p in detected_patterns
                if p.confidence >= min_confidence
            ]

            return filtered_patterns
        except Exception as e:
            logger.error(f"Error detecting patterns: {str(e)}")
            raise AgentToolError(f"Failed to detect patterns: {str(e)}") from e


class StrategyTool(AgentTool):
    """Tool for executing trading strategies."""

    def __init__(self, strategy_repository):
        """
        Initialize the strategy tool.

        Args:
            strategy_repository: Repository of trading strategies
        """
        self.strategy_repository = strategy_repository

    @property
    def name(self) -> str:
        """Get the name of the tool."""
        return "strategy"

    @property
    def description(self) -> str:
        """Get a description of what the tool does."""
        return "Executes trading strategies on market data"

    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        """Get the parameters the tool accepts."""
        return {
            "strategy_id": {
                "type": "string",
                "description": "ID of the strategy to execute",
                "required": True,
            },
            "data": {
                "type": "dataframe",
                "description": "Market data as a pandas DataFrame",
                "required": True,
            },
            "parameters": {
                "type": "dict",
                "description": "Strategy parameters",
                "required": False,
            },
        }

    def run(
        self, strategy_id: str, data: pd.DataFrame, parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Execute a trading strategy.

        Args:
            strategy_id: ID of the strategy to execute
            data: Market data as a pandas DataFrame
            parameters: Strategy parameters

        Returns:
            Strategy execution results

        Raises:
            AgentToolError: If strategy execution fails
        """
        try:
            # Moved import inside try block and indented
            from forex_ai.analysis.pine_script import PineScriptExecutor

            # Get strategy from repository
            strategy = self.strategy_repository.get_strategy(strategy_id)
            if not strategy:
                raise AgentToolError(f"Strategy not found: {strategy_id}")

            # Create executor
            executor = PineScriptExecutor(strategy, parameters)

            # Execute strategy
            result = executor.execute(data)

            return result
        except Exception as e:
            logger.error(f"Error executing strategy: {str(e)}")
            raise AgentToolError(f"Failed to execute strategy: {str(e)}") from e


class OptimizationTool(AgentTool):
    """Tool for optimizing trading strategies."""

    def __init__(self, strategy_repository):
        """
        Initialize the optimization tool.

        Args:
            strategy_repository: Repository of trading strategies
        """
        self.strategy_repository = strategy_repository

    @property
    def name(self) -> str:
        """Get the name of the tool."""
        return "optimization"

    @property
    def description(self) -> str:
        """Get a description of what the tool does."""
        return "Optimizes trading strategy parameters"

    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        """Get the parameters the tool accepts."""
        return {
            "strategy_id": {
                "type": "string",
                "description": "ID of the strategy to optimize",
                "required": True,
            },
            "data": {
                "type": "dataframe",
                "description": "Market data as a pandas DataFrame",
                "required": True,
            },
            "parameter_spaces": {
                "type": "list",
                "description": "Parameter spaces to search",
                "required": True,
            },
            "method": {
                "type": "string",
                "description": "Optimization method (grid_search, random_search)",
                "required": False,
            },
            "metric": {
                "type": "string",
                "description": "Performance metric to optimize",
                "required": False,
            },
        }

    def run(
        self,
        strategy_id: str,
        data: pd.DataFrame,
        parameter_spaces: List[Dict[str, Any]],
        method: str = "random_search",
        metric: str = "sharpe_ratio",
    ) -> Dict[str, Any]:
        """
        Optimize a trading strategy.

        Args:
            strategy_id: ID of the strategy to optimize
            data: Market data as a pandas DataFrame
            parameter_spaces: Parameter spaces to search
            method: Optimization method
            metric: Performance metric to optimize

        Returns:
            Optimization results

        Raises:
            AgentToolError: If optimization fails
        """
        try:
            # Moved import inside try block and indented
            from forex_ai.analysis.strategy_optimization import (
                StrategyOptimizer,
                ParameterSpace,
            )

            # Get strategy from repository
            strategy = self.strategy_repository.get_strategy(strategy_id)
            if not strategy:
                raise AgentToolError(f"Strategy not found: {strategy_id}")

            # Convert parameter spaces to ParameterSpace objects
            param_spaces = []
            for ps in parameter_spaces:
                param_spaces.append(
                    ParameterSpace(
                        name=ps["name"],
                        min_value=ps.get("min_value", 0),
                        max_value=ps.get("max_value", 0),
                        step=ps.get("step", 1),
                        is_categorical=ps.get("is_categorical", False),
                        categories=ps.get("categories", []),
                    )
                )

            # Create optimizer
            optimizer = StrategyOptimizer(data=data, performance_metric=metric)

            # Run optimization
            if method == "grid_search":
                result = optimizer.optimize_grid_search(strategy, param_spaces)
            else:
                result = optimizer.optimize_random_search(strategy, param_spaces)

            # Convert result to dictionary
            return {
                "strategy_id": result.strategy_id,
                "strategy_name": result.strategy_name,
                "best_parameters": result.best_parameters,
                "performance_metrics": result.performance_metrics,
                "optimization_time": result.optimization_time,
                "created_at": result.created_at.isoformat(),
                "unique_id": result.unique_id,
            }
        except Exception as e:
            logger.error(f"Error optimizing strategy: {str(e)}")
            raise AgentToolError(f"Failed to optimize strategy: {str(e)}") from e


class NewsFetcherTool(AgentTool):
    """Tool for fetching financial news."""

    def __init__(self, api_key: str = None, base_url: str = None):
        """
        Initialize the news fetcher tool.

        Args:
            api_key: API key for news service
            base_url: Base URL for news API
        """
        self.api_key = api_key
        self.base_url = base_url or "https://newsapi.org/v2"

    @property
    def name(self) -> str:
        """Get the name of the tool."""
        return "news_fetcher"

    @property
    def description(self) -> str:
        """Get a description of what the tool does."""
        return "Fetches financial news articles"

    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        """Get the parameters the tool accepts."""
        return {
            "query": {
                "type": "string",
                "description": "Search query for news",
                "required": True,
            },
            "from_date": {
                "type": "string",
                "description": "Start date in ISO format",
                "required": False,
            },
            "to_date": {
                "type": "string",
                "description": "End date in ISO format",
                "required": False,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of articles to fetch",
                "required": False,
            },
        }

    def run(
        self, query: str, from_date: str = None, to_date: str = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Fetch financial news articles.

        Args:
            query: Search query for news
            from_date: Start date in ISO format
            to_date: End date in ISO format
            limit: Maximum number of articles to fetch

        Returns:
            List of news articles

        Raises:
            AgentToolError: If news fetching fails
        """
        try:
            # This is a simplified implementation
            # In a real system, you'd integrate with a news API

            if not self.api_key:
                # Return mock data if no API key
                logger.warning("No API key provided, returning mock news data")
                return self._get_mock_news(query, limit)

            # Prepare request parameters
            params = {
                "q": query,
                "apiKey": self.api_key,
                "pageSize": limit,
                "language": "en",
                "sortBy": "publishedAt",
            }

            # Add date filters if provided
            if from_date:
                params["from"] = from_date
            if to_date:
                params["to"] = to_date

            # Make API request
            response = requests.get(f"{self.base_url}/everything", params=params)
            response.raise_for_status()

            # Parse response
            data = response.json()

            # Extract articles
            articles = data.get("articles", [])

            # Process and return articles
            return [
                {
                    "title": article.get("title"),
                    "description": article.get("description"),
                    "content": article.get("content"),
                    "url": article.get("url"),
                    "published_at": article.get("publishedAt"),
                    "source": article.get("source", {}).get("name"),
                }
                for article in articles[:limit]
            ]
        except Exception as e:
            logger.error(f"Error fetching news: {str(e)}")
            # Fall back to mock data on error
            logger.info("Falling back to mock news data")
            return self._get_mock_news(query, limit)

    def _get_mock_news(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """
        Generate mock news data for testing.

        Args:
            query: Search query
            limit: Maximum number of articles

        Returns:
            List of mock news articles
        """
        # Generate mock articles
        currencies = ["EUR", "USD", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]
        articles = []

        # Extract currency from query if possible
        query_currencies = [c for c in currencies if c in query]

        for i in range(min(limit, 5)):
            # Use query currencies or pick random ones
            if query_currencies:
                curr = query_currencies[0]
            else:
                curr = np.random.choice(currencies)

            # Generate article with current date
            date = (datetime.now() - timedelta(days=i)).isoformat()

            articles.append(
                {
                    "title": f"{curr} Analysis: Technical and Fundamental Outlook",
                    "description": f"Latest analysis of {curr} trading patterns and economic factors.",
                    "content": f"The {curr} has shown significant movement in recent trading sessions...",
                    "url": f"https://example.com/forex/{curr.lower()}-analysis",
                    "published_at": date,
                    "source": "Mock Financial News",
                }
            )

        return articles

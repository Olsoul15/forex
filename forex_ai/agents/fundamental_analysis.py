"""
Fundamental Analysis Agent Implementation for the AI Forex Trading System.

This module provides a specialized agent for performing fundamental analysis
on forex markets, including economic data, news sentiment, and central bank policies.
"""

from typing import Dict, List, Optional, Any, Union
import pandas as pd
from datetime import datetime, timedelta
import json

from .base import BaseAgent
from .tools import NewsFetcherTool
from forex_ai.utils.logging import get_logger
from forex_ai.exceptions import AgentError

logger = get_logger(__name__)


class FundamentalAnalysisAgent(BaseAgent):
    """
    Agent specialized in fundamental analysis of forex markets.

    This agent analyzes economic indicators, news sentiment, and central bank
    policies to provide insights on fundamental market drivers.
    """

    def initialize(self) -> None:
        """Initialize the fundamental analysis agent with required tools."""
        # Add news fetcher tool
        api_key = self.config.get("news_api_key")
        self.add_tool(NewsFetcherTool(api_key=api_key))

        # Initialize state
        self.memory.update_state("last_analysis_time", None)
        self.memory.update_state("economic_calendar", {})

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process fundamental data and perform analysis.

        Args:
            input_data: Dictionary containing analysis request parameters
                Required keys:
                - symbol: Trading symbol
                Optional keys:
                - start_date: Start date for analysis
                - end_date: End date for analysis
                - include_news: Whether to include news analysis

        Returns:
            Dictionary containing analysis results

        Raises:
            AgentError: If analysis fails
        """
        try:
            self.metrics.start_execution()

            # Extract parameters
            symbol = input_data.get("symbol")

            if not symbol:
                raise AgentError("Symbol is required")

            # Get currency pair components
            currencies = self._extract_currencies(symbol)

            # Analyze economic data
            economic_analysis = self._analyze_economic_data(currencies)

            # Fetch and analyze news if requested
            news_analysis = None
            if input_data.get("include_news", True):
                news_analysis = self._analyze_news(currencies)

            # Analyze central bank policies
            policy_analysis = self._analyze_central_bank_policies(currencies)

            # Create analysis summary
            summary = self._create_analysis_summary(
                symbol, economic_analysis, news_analysis, policy_analysis
            )

            # Record analysis in memory
            self.memory.add_observation(
                {
                    "type": "fundamental_analysis",
                    "symbol": symbol,
                    "economic_sentiment": economic_analysis.get("sentiment"),
                    "news_sentiment": (
                        news_analysis.get("overall_sentiment")
                        if news_analysis
                        else "unknown"
                    ),
                    "policy_bias": policy_analysis.get("policy_bias"),
                    "timestamp": datetime.now().isoformat(),
                }
            )

            self.memory.update_state("last_analysis_time", datetime.now().isoformat())

            # Record metrics
            execution_time = self.metrics.end_execution("fundamental_analysis")
            self.metrics.track_metric("analysis_time", execution_time)

            return {
                "symbol": symbol,
                "currencies": currencies,
                "analysis_time": datetime.now().isoformat(),
                "economic_analysis": economic_analysis,
                "news_analysis": news_analysis,
                "policy_analysis": policy_analysis,
                "summary": summary,
            }

        except Exception as e:
            logger.error(f"Error in fundamental analysis: {str(e)}")
            self.metrics.record_error("fundamental_analysis_error")
            raise AgentError(f"Failed to perform fundamental analysis: {str(e)}") from e

    def _extract_currencies(self, symbol: str) -> List[str]:
        """
        Extract individual currencies from a forex pair symbol.

        Args:
            symbol: Trading symbol (e.g., 'EUR/USD', 'GBPJPY')

        Returns:
            List of currency codes
        """
        # Remove common separators and split
        clean_symbol = symbol.replace("/", "").replace("-", "").replace("_", "")

        # Split into 3-character currency codes
        currencies = []
        for i in range(0, len(clean_symbol), 3):
            if i + 3 <= len(clean_symbol):
                currencies.append(clean_symbol[i : i + 3])

        return currencies

    def _analyze_economic_data(self, currencies: List[str]) -> Dict[str, Any]:
        """
        Analyze economic data for the given currencies.

        Args:
            currencies: List of currency codes

        Returns:
            Dictionary containing economic analysis
        """
        # This is a simplified implementation
        # In a real system, you'd fetch data from economic APIs

        # Get or initialize economic calendar
        economic_calendar = self.memory.get_state("economic_calendar", {})

        # Mock economic indicators for demonstration
        economic_indicators = {
            "USD": {
                "gdp_growth": 2.1,
                "inflation": 3.1,
                "unemployment": 3.8,
                "interest_rate": 5.25,
                "sentiment": "neutral",
            },
            "EUR": {
                "gdp_growth": 0.3,
                "inflation": 2.9,
                "unemployment": 6.5,
                "interest_rate": 4.50,
                "sentiment": "bearish",
            },
            "GBP": {
                "gdp_growth": 0.6,
                "inflation": 4.0,
                "unemployment": 4.2,
                "interest_rate": 5.25,
                "sentiment": "neutral",
            },
            "JPY": {
                "gdp_growth": 1.5,
                "inflation": 2.8,
                "unemployment": 2.6,
                "interest_rate": 0.1,
                "sentiment": "bullish",
            },
            "AUD": {
                "gdp_growth": 1.8,
                "inflation": 3.4,
                "unemployment": 4.1,
                "interest_rate": 4.35,
                "sentiment": "neutral",
            },
            "CAD": {
                "gdp_growth": 1.0,
                "inflation": 3.0,
                "unemployment": 5.7,
                "interest_rate": 5.0,
                "sentiment": "bearish",
            },
            "CHF": {
                "gdp_growth": 0.3,
                "inflation": 1.4,
                "unemployment": 2.3,
                "interest_rate": 1.75,
                "sentiment": "bullish",
            },
            "NZD": {
                "gdp_growth": 0.8,
                "inflation": 4.0,
                "unemployment": 4.0,
                "interest_rate": 5.5,
                "sentiment": "neutral",
            },
        }

        # Check if all currencies are recognized
        for currency in currencies:
            if currency not in economic_indicators:
                economic_indicators[currency] = {
                    "gdp_growth": 0.0,
                    "inflation": 0.0,
                    "unemployment": 0.0,
                    "interest_rate": 0.0,
                    "sentiment": "unknown",
                }

        # Extract relevant data
        analysis = {}
        for currency in currencies:
            analysis[currency] = economic_indicators.get(currency, {})

        # Compare currencies if it's a pair
        if len(currencies) == 2:
            base_currency = currencies[0]
            quote_currency = currencies[1]

            # Calculate differentials
            interest_rate_differential = economic_indicators.get(base_currency, {}).get(
                "interest_rate", 0
            ) - economic_indicators.get(quote_currency, {}).get("interest_rate", 0)

            inflation_differential = economic_indicators.get(base_currency, {}).get(
                "inflation", 0
            ) - economic_indicators.get(quote_currency, {}).get("inflation", 0)

            growth_differential = economic_indicators.get(base_currency, {}).get(
                "gdp_growth", 0
            ) - economic_indicators.get(quote_currency, {}).get("gdp_growth", 0)

            # Determine fundamental bias
            bias = "neutral"
            if interest_rate_differential > 0.5:
                bias = "bullish"  # Higher interest rates tend to strengthen currency
            elif interest_rate_differential < -0.5:
                bias = "bearish"

            # Adjust bias based on other factors
            if bias == "bullish" and growth_differential < -1.0:
                bias = "neutral"  # Growth concerns offset interest rate advantage
            elif bias == "bearish" and growth_differential > 1.0:
                bias = "neutral"  # Growth advantage offsets interest rate disadvantage

            analysis["differentials"] = {
                "interest_rate": interest_rate_differential,
                "inflation": inflation_differential,
                "growth": growth_differential,
            }

            analysis["sentiment"] = bias

        return analysis

    def _analyze_news(self, currencies: List[str]) -> Dict[str, Any]:
        """
        Analyze news for the given currencies.

        Args:
            currencies: List of currency codes

        Returns:
            Dictionary containing news analysis
        """
        try:
            # Create currency-specific queries
            all_news = []
            for currency in currencies:
                # Create search query
                query = f"{currency} forex currency"

                # Fetch news
                news = self.use_tool(
                    "news_fetcher",
                    query=query,
                    from_date=(datetime.now() - timedelta(days=3)).isoformat(),
                    limit=5,
                )

                # Add currency to each article
                for article in news:
                    article["currency"] = currency

                all_news.extend(news)

            # Sort by published date
            all_news.sort(key=lambda x: x.get("published_at", ""), reverse=True)

            # Try to use advanced news processing with RAG
            try:
                from forex_ai.analysis.sentiment_analysis import RAGNewsProcessor

                # Initialize the RAG processor
                processor = RAGNewsProcessor()
                logger.info("Using advanced RAG-enhanced news processing")

                # Process all news items
                processed_news = processor.batch_process(all_news)

                # Extract sentiment and impact predictions
                currency_sentiment = {}
                news_impact = {}

                for article in processed_news:
                    currency = article.get("currency")
                    analysis = article.get("analysis", {})

                    # Extract sentiment
                    sentiment_result = analysis.get("sentiment", {})
                    sentiment_score = sentiment_result.get("score", 0)
                    sentiment_label = (
                        "bullish"
                        if sentiment_score > 0.2
                        else ("bearish" if sentiment_score < -0.2 else "neutral")
                    )

                    # Extract impact
                    impact = analysis.get("impact_prediction", {})

                    # Store sentiment for each currency
                    if currency and currency not in currency_sentiment:
                        currency_sentiment[currency] = []

                    if currency:
                        currency_sentiment[currency].append(sentiment_label)

                    # Store impact predictions
                    affected_pairs = impact.get("affected_pairs", [])
                    impact_probability = impact.get("impact_probability", 0)

                    for pair in affected_pairs:
                        if pair not in news_impact:
                            news_impact[pair] = []

                        news_impact[pair].append(
                            {
                                "title": article.get("title"),
                                "impact_probability": impact_probability,
                                "sentiment": sentiment_label,
                            }
                        )

                # Calculate overall sentiment per currency
                currency_sentiment_summary = {}
                for currency, sentiment_list in currency_sentiment.items():
                    if not sentiment_list:
                        currency_sentiment_summary[currency] = "neutral"
                        continue

                    bullish_count = sentiment_list.count("bullish")
                    bearish_count = sentiment_list.count("bearish")

                    if bullish_count > bearish_count:
                        currency_sentiment_summary[currency] = "bullish"
                    elif bearish_count > bullish_count:
                        currency_sentiment_summary[currency] = "bearish"
                    else:
                        currency_sentiment_summary[currency] = "neutral"

                # Determine overall pair sentiment if applicable
                overall_sentiment = "neutral"
                if len(currencies) == 2:
                    base_currency = currencies[0]
                    quote_currency = currencies[1]

                    base_sentiment = currency_sentiment_summary.get(
                        base_currency, "neutral"
                    )
                    quote_sentiment = currency_sentiment_summary.get(
                        quote_currency, "neutral"
                    )

                    # Enhanced sentiment determination using direct pair info when available
                    pair = f"{base_currency}/{quote_currency}"
                    if pair in news_impact and news_impact[pair]:
                        # Get sentiment from the highest impact news for this pair
                        high_impact_news = sorted(
                            news_impact[pair],
                            key=lambda x: x.get("impact_probability", 0),
                            reverse=True,
                        )

                        # Use sentiment from high impact news if available
                        if high_impact_news:
                            overall_sentiment = high_impact_news[0].get(
                                "sentiment", "neutral"
                            )
                    else:
                        # Fallback to relative currency sentiment
                        if base_sentiment == "bullish" and quote_sentiment in [
                            "neutral",
                            "bearish",
                        ]:
                            overall_sentiment = "bullish"
                        elif base_sentiment == "bearish" and quote_sentiment in [
                            "neutral",
                            "bullish",
                        ]:
                            overall_sentiment = "bearish"
                        elif (
                            base_sentiment == "neutral" and quote_sentiment == "bullish"
                        ):
                            overall_sentiment = "bearish"
                        elif (
                            base_sentiment == "neutral" and quote_sentiment == "bearish"
                        ):
                            overall_sentiment = "bullish"

                # Add enhanced analysis
                return {
                    "articles": processed_news,
                    "currency_sentiment": currency_sentiment_summary,
                    "overall_sentiment": overall_sentiment,
                    "news_impact": news_impact,
                    "analysis_method": "advanced",
                }

            except (ImportError, Exception) as e:
                logger.warning(f"Unable to use advanced news processing: {str(e)}")
                logger.info("Falling back to simple sentiment analysis")

            # Simple fallback sentiment analysis
            # In a real system, you would use a more sophisticated sentiment analyzer
            sentiments = {}
            for article in all_news:
                currency = article.get("currency")
                title = article.get("title", "").lower()

                sentiment = "neutral"
                if any(
                    word in title
                    for word in [
                        "rise",
                        "gain",
                        "jump",
                        "surge",
                        "strong",
                        "strengthen",
                    ]
                ):
                    sentiment = "bullish"
                elif any(
                    word in title
                    for word in ["fall", "drop", "decline", "weak", "worsen", "plunge"]
                ):
                    sentiment = "bearish"

                if currency not in sentiments:
                    sentiments[currency] = []

                sentiments[currency].append(sentiment)

            # Calculate overall sentiment per currency
            currency_sentiment = {}
            for currency, sentiment_list in sentiments.items():
                if not sentiment_list:
                    currency_sentiment[currency] = "neutral"
                    continue

                bullish_count = sentiment_list.count("bullish")
                bearish_count = sentiment_list.count("bearish")

                if bullish_count > bearish_count:
                    currency_sentiment[currency] = "bullish"
                elif bearish_count > bullish_count:
                    currency_sentiment[currency] = "bearish"
                else:
                    currency_sentiment[currency] = "neutral"

            # Determine overall pair sentiment if applicable
            overall_sentiment = "neutral"
            if len(currencies) == 2:
                base_currency = currencies[0]
                quote_currency = currencies[1]

                base_sentiment = currency_sentiment.get(base_currency, "neutral")
                quote_sentiment = currency_sentiment.get(quote_currency, "neutral")

                if base_sentiment == "bullish" and quote_sentiment in [
                    "neutral",
                    "bearish",
                ]:
                    overall_sentiment = "bullish"
                elif base_sentiment == "bearish" and quote_sentiment in [
                    "neutral",
                    "bullish",
                ]:
                    overall_sentiment = "bearish"
                elif base_sentiment == "neutral" and quote_sentiment == "bullish":
                    overall_sentiment = "bearish"
                elif base_sentiment == "neutral" and quote_sentiment == "bearish":
                    overall_sentiment = "bullish"

            return {
                "articles": all_news,
                "currency_sentiment": currency_sentiment,
                "overall_sentiment": overall_sentiment,
                "analysis_method": "simple",
            }

        except Exception as e:
            logger.error(f"Error analyzing news: {str(e)}")
            return {
                "articles": [],
                "currency_sentiment": {},
                "overall_sentiment": "unknown",
                "error": str(e),
                "analysis_method": "failed",
            }

    def _analyze_central_bank_policies(self, currencies: List[str]) -> Dict[str, Any]:
        """
        Analyze central bank policies for the given currencies.

        Args:
            currencies: List of currency codes

        Returns:
            Dictionary containing policy analysis
        """
        # This is a simplified implementation
        # In a real system, you'd fetch data from central bank sources

        # Mock central bank data
        central_banks = {
            "USD": {
                "name": "Federal Reserve",
                "current_policy": "tightening",
                "last_change": "2023-07-26",
                "next_meeting": "2023-12-13",
                "forward_guidance": "Considering rate cuts in 2024",
            },
            "EUR": {
                "name": "European Central Bank",
                "current_policy": "neutral",
                "last_change": "2023-09-14",
                "next_meeting": "2023-12-14",
                "forward_guidance": "Monitoring inflation developments",
            },
            "GBP": {
                "name": "Bank of England",
                "current_policy": "tightening",
                "last_change": "2023-08-03",
                "next_meeting": "2023-12-14",
                "forward_guidance": "Expects rates to remain high for an extended period",
            },
            "JPY": {
                "name": "Bank of Japan",
                "current_policy": "accommodative",
                "last_change": "2023-07-28",
                "next_meeting": "2023-12-19",
                "forward_guidance": "Considering policy normalization",
            },
            "AUD": {
                "name": "Reserve Bank of Australia",
                "current_policy": "neutral",
                "last_change": "2023-11-07",
                "next_meeting": "2023-12-05",
                "forward_guidance": "May need further tightening",
            },
            "CAD": {
                "name": "Bank of Canada",
                "current_policy": "tightening",
                "last_change": "2023-07-12",
                "next_meeting": "2023-12-06",
                "forward_guidance": "Ready to raise rates further if needed",
            },
            "CHF": {
                "name": "Swiss National Bank",
                "current_policy": "neutral",
                "last_change": "2023-06-22",
                "next_meeting": "2023-12-14",
                "forward_guidance": "Willing to intervene in FX markets",
            },
            "NZD": {
                "name": "Reserve Bank of New Zealand",
                "current_policy": "tightening",
                "last_change": "2023-10-04",
                "next_meeting": "2024-02-28",
                "forward_guidance": "Expects rates to remain restrictive",
            },
        }

        # Extract relevant data
        policy_data = {}
        for currency in currencies:
            policy_data[currency] = central_banks.get(
                currency,
                {
                    "name": "Unknown Central Bank",
                    "current_policy": "unknown",
                    "last_change": "unknown",
                    "next_meeting": "unknown",
                    "forward_guidance": "No data available",
                },
            )

        # Determine policy bias for the pair
        policy_bias = "neutral"
        if len(currencies) == 2:
            base_currency = currencies[0]
            quote_currency = currencies[1]

            base_policy = central_banks.get(base_currency, {}).get(
                "current_policy", "unknown"
            )
            quote_policy = central_banks.get(quote_currency, {}).get(
                "current_policy", "unknown"
            )

            # Simple policy comparison
            if base_policy == "tightening" and quote_policy in [
                "neutral",
                "accommodative",
            ]:
                policy_bias = "bullish"  # Base currency may strengthen
            elif base_policy == "accommodative" and quote_policy in [
                "neutral",
                "tightening",
            ]:
                policy_bias = "bearish"  # Base currency may weaken
            elif base_policy == "neutral" and quote_policy == "tightening":
                policy_bias = "bearish"  # Base currency may weaken
            elif base_policy == "neutral" and quote_policy == "accommodative":
                policy_bias = "bullish"  # Base currency may strengthen

        return {"central_banks": policy_data, "policy_bias": policy_bias}

    def _create_analysis_summary(
        self,
        symbol: str,
        economic_analysis: Dict[str, Any],
        news_analysis: Optional[Dict[str, Any]],
        policy_analysis: Dict[str, Any],
    ) -> str:
        """
        Create a human-readable summary of the fundamental analysis.

        Args:
            symbol: Trading symbol
            economic_analysis: Economic data analysis
            news_analysis: News analysis results
            policy_analysis: Central bank policy analysis

        Returns:
            String containing analysis summary
        """
        try:
            currencies = self._extract_currencies(symbol)

            # Ensure we have at least one currency
            if not currencies:
                return "Error: Could not extract currencies from symbol"

            # Compose summary
            summary = f"Fundamental Analysis Summary for {symbol}:\n\n"

            # Add economic analysis
            summary += "Economic Overview:\n"
            for currency in currencies:
                econ_data = economic_analysis.get(currency, {})
                summary += f"- {currency}: "
                summary += f"GDP Growth: {econ_data.get('gdp_growth', 0):.1f}%, "
                summary += f"Inflation: {econ_data.get('inflation', 0):.1f}%, "
                summary += f"Interest Rate: {econ_data.get('interest_rate', 0):.2f}%\n"

            # Add differentials if available
            if "differentials" in economic_analysis:
                diff = economic_analysis["differentials"]
                summary += f"\nDifferentials ({currencies[0]}-{currencies[1]}):\n"
                summary += f"- Interest Rate: {diff.get('interest_rate', 0):.2f}%\n"
                summary += f"- Inflation: {diff.get('inflation', 0):.1f}%\n"
                summary += f"- GDP Growth: {diff.get('growth', 0):.1f}%\n"

            # Add central bank analysis
            summary += "\nCentral Bank Policies:\n"
            for currency in currencies:
                cb_data = policy_analysis.get("central_banks", {}).get(currency, {})
                summary += f"- {cb_data.get('name', 'Unknown')}: "
                summary += f"Current Stance: {cb_data.get('current_policy', 'Unknown').title()}, "
                summary += f"Next Meeting: {cb_data.get('next_meeting', 'Unknown')}\n"
                summary += f"  Guidance: {cb_data.get('forward_guidance', 'None')}\n"

            # Add news analysis if available
            if news_analysis:
                summary += "\nRecent News Sentiment:\n"
                for currency in currencies:
                    sentiment = news_analysis.get("currency_sentiment", {}).get(
                        currency, "neutral"
                    )
                    summary += f"- {currency}: {sentiment.title()}\n"

                # Add recent headlines
                summary += "\nRecent Headlines:\n"
                for article in news_analysis.get("articles", [])[:3]:
                    summary += f"- {article.get('title')}\n"

            # Overall outlook
            economic_sentiment = economic_analysis.get("sentiment", "neutral")
            news_sentiment = (
                news_analysis.get("overall_sentiment", "neutral")
                if news_analysis
                else "neutral"
            )
            policy_bias = policy_analysis.get("policy_bias", "neutral")

            # Count sentiments
            sentiment_count = {
                "bullish": sum(
                    1
                    for s in [economic_sentiment, news_sentiment, policy_bias]
                    if s == "bullish"
                ),
                "bearish": sum(
                    1
                    for s in [economic_sentiment, news_sentiment, policy_bias]
                    if s == "bearish"
                ),
                "neutral": sum(
                    1
                    for s in [economic_sentiment, news_sentiment, policy_bias]
                    if s == "neutral"
                ),
            }

            # Determine overall outlook
            if sentiment_count["bullish"] > sentiment_count["bearish"]:
                overall_outlook = "Bullish"
            elif sentiment_count["bearish"] > sentiment_count["bullish"]:
                overall_outlook = "Bearish"
            else:
                overall_outlook = "Neutral"

            # Add confidence based on agreement
            if sentiment_count["bullish"] == 3 or sentiment_count["bearish"] == 3:
                confidence = "High"
            elif sentiment_count["bullish"] == 2 or sentiment_count["bearish"] == 2:
                confidence = "Moderate"
            else:
                confidence = "Low"

            summary += f"\nOverall Fundamental Outlook: {overall_outlook} (Confidence: {confidence})\n"
            summary += f"- Economic Sentiment: {economic_sentiment.title()}\n"
            summary += f"- News Sentiment: {news_sentiment.title()}\n"
            summary += f"- Policy Bias: {policy_bias.title()}"

            return summary
        except Exception as e:
            logger.error(f"Error creating analysis summary: {str(e)}")
            return f"Error creating analysis summary: {str(e)}"


class EconomicCalendarAnalyzer:
    """
    Specialized component for analyzing economic calendar events and their impact
    on forex markets.

    This class focuses on:
    - Retrieving economic calendar events
    - Assessing their importance and potential market impact
    - Forecasting market reactions based on historical patterns
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the economic calendar analyzer.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.logger.info("Initialized EconomicCalendarAnalyzer")

    def get_upcoming_events(
        self, currencies: List[str], days_ahead: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Get upcoming economic calendar events for the specified currencies.

        Args:
            currencies: List of currency codes
            days_ahead: Number of days to look ahead

        Returns:
            List of economic events
        """
        try:
            self.logger.info(
                f"Fetching upcoming events for {currencies} ({days_ahead} days ahead)"
            )

            end_date = datetime.now() + timedelta(days=days_ahead)

            # In a real implementation, this would fetch data from an economic calendar API
            # Here we're mocking the data for demonstration purposes

            events = []

            # Mock data for demonstration
            mock_events = {
                "USD": [
                    {
                        "date": (datetime.now() + timedelta(days=1)).strftime(
                            "%Y-%m-%d"
                        ),
                        "time": "14:30",
                        "currency": "USD",
                        "event": "Non-Farm Payrolls",
                        "importance": "high",
                        "forecast": "175K",
                        "previous": "187K",
                    },
                    {
                        "date": (datetime.now() + timedelta(days=2)).strftime(
                            "%Y-%m-%d"
                        ),
                        "time": "20:00",
                        "currency": "USD",
                        "event": "FOMC Meeting Minutes",
                        "importance": "high",
                        "forecast": "",
                        "previous": "",
                    },
                ],
                "EUR": [
                    {
                        "date": (datetime.now() + timedelta(days=3)).strftime(
                            "%Y-%m-%d"
                        ),
                        "time": "10:00",
                        "currency": "EUR",
                        "event": "CPI y/y",
                        "importance": "high",
                        "forecast": "2.9%",
                        "previous": "3.1%",
                    }
                ],
                "GBP": [
                    {
                        "date": (datetime.now() + timedelta(days=4)).strftime(
                            "%Y-%m-%d"
                        ),
                        "time": "12:00",
                        "currency": "GBP",
                        "event": "BOE Interest Rate Decision",
                        "importance": "high",
                        "forecast": "5.25%",
                        "previous": "5.25%",
                    }
                ],
            }

            # Filter events by specified currencies
            for currency in currencies:
                if currency in mock_events:
                    events.extend(mock_events[currency])

            # Sort by date and time
            events.sort(key=lambda x: x["date"] + " " + x["time"])

            return events

        except Exception as e:
            self.logger.error(f"Error fetching upcoming events: {str(e)}")
            return []

    def evaluate_event_impact(
        self, event: Dict[str, Any], currency_pair: str
    ) -> Dict[str, Any]:
        """
        Evaluate the potential impact of an economic event on a currency pair.

        Args:
            event: Economic calendar event
            currency_pair: Currency pair (e.g., "EUR/USD")

        Returns:
            Impact evaluation
        """
        try:
            self.logger.info(
                f"Evaluating impact of {event['event']} on {currency_pair}"
            )

            # Extract currencies from the pair
            base_currency = currency_pair[:3]
            quote_currency = (
                currency_pair[3:] if len(currency_pair) == 6 else currency_pair[4:]
            )

            # Default impact assessment
            impact = {
                "event": event["event"],
                "currency": event["currency"],
                "date": event["date"],
                "time": event["time"],
                "impact": "neutral",
                "magnitude": 0.0,
                "confidence": 0.0,
                "reasoning": "",
            }

            # Check if the event is relevant to the currency pair
            if event["currency"] not in [base_currency, quote_currency]:
                impact["reasoning"] = f"Event not directly relevant to {currency_pair}"
                return impact

            # Determine if this is a high, medium, or low importance event
            importance = event.get("importance", "low").lower()

            # Map importance to impact magnitude
            magnitude_map = {"high": 0.8, "medium": 0.5, "low": 0.2}
            impact["magnitude"] = magnitude_map.get(importance, 0.2)

            # Determine direction (positive/negative for the currency)
            # This would require a more sophisticated analysis based on the type of event,
            # expected vs. actual values, etc.

            # For demonstration, we're using a simplified logic
            if event["currency"] == base_currency:
                # For base currency, positive impact strengthens the pair
                impact["impact"] = "bullish"
            else:
                # For quote currency, positive impact weakens the pair
                impact["impact"] = "bearish"

            # Assign confidence based on event importance
            confidence_map = {"high": 0.7, "medium": 0.5, "low": 0.3}
            impact["confidence"] = confidence_map.get(importance, 0.3)

            # Generate reasoning
            impact["reasoning"] = (
                f"{event['event']} typically has a {importance} impact on {event['currency']}. "
                f"As this affects the {base_currency if event['currency'] == base_currency else quote_currency} "
                f"in the {currency_pair} pair, expected impact is {impact['impact']}."
            )

            return impact

        except Exception as e:
            self.logger.error(f"Error evaluating event impact: {str(e)}")
            return {
                "event": event.get("event", "Unknown"),
                "impact": "neutral",
                "magnitude": 0.0,
                "confidence": 0.0,
                "reasoning": f"Error evaluating impact: {str(e)}",
            }

    def get_calendar_analysis(
        self, currency_pair: str, days_ahead: int = 7
    ) -> Dict[str, Any]:
        """
        Get a comprehensive analysis of upcoming economic events for a currency pair.

        Args:
            currency_pair: Currency pair (e.g., "EUR/USD")
            days_ahead: Number of days to look ahead

        Returns:
            Calendar analysis
        """
        try:
            # Extract currencies from the pair
            base_currency = currency_pair[:3]
            quote_currency = (
                currency_pair[3:] if len(currency_pair) == 6 else currency_pair[4:]
            )

            # Get upcoming events for both currencies
            events = self.get_upcoming_events(
                [base_currency, quote_currency], days_ahead
            )

            # Evaluate impact of each event
            event_impacts = []
            for event in events:
                impact = self.evaluate_event_impact(event, currency_pair)
                event_impacts.append(impact)

            # Separate events by currency
            base_events = [e for e in event_impacts if e["currency"] == base_currency]
            quote_events = [e for e in event_impacts if e["currency"] == quote_currency]

            # Calculate overall sentiment based on event impacts
            base_sentiment = self._calculate_overall_sentiment(base_events)
            quote_sentiment = self._calculate_overall_sentiment(quote_events)

            # Determine pair sentiment (relative strength)
            if base_sentiment["score"] > quote_sentiment["score"]:
                pair_sentiment = "bullish"
                strength = min(
                    1.0, (base_sentiment["score"] - quote_sentiment["score"]) * 2
                )
            elif quote_sentiment["score"] > base_sentiment["score"]:
                pair_sentiment = "bearish"
                strength = min(
                    1.0, (quote_sentiment["score"] - base_sentiment["score"]) * 2
                )
            else:
                pair_sentiment = "neutral"
                strength = 0.0

            # Find key events (highest impact)
            key_events = sorted(
                event_impacts,
                key=lambda x: x["magnitude"] * x["confidence"],
                reverse=True,
            )[:3]

            return {
                "currency_pair": currency_pair,
                "analysis_period": f"{datetime.now().strftime('%Y-%m-%d')} to {(datetime.now() + timedelta(days=days_ahead)).strftime('%Y-%m-%d')}",
                "total_events": len(events),
                "base_currency_events": len(base_events),
                "quote_currency_events": len(quote_events),
                "base_currency_sentiment": base_sentiment,
                "quote_currency_sentiment": quote_sentiment,
                "pair_sentiment": pair_sentiment,
                "sentiment_strength": strength,
                "key_events": key_events,
                "all_events": event_impacts,
            }

        except Exception as e:
            self.logger.error(f"Error in calendar analysis: {str(e)}")
            return {
                "currency_pair": currency_pair,
                "error": str(e),
                "pair_sentiment": "neutral",
                "sentiment_strength": 0.0,
                "key_events": [],
                "all_events": [],
            }

    def _calculate_overall_sentiment(
        self, events: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate overall sentiment from a list of event impacts.

        Args:
            events: List of event impact evaluations

        Returns:
            Sentiment assessment
        """
        if not events:
            return {"score": 0.0, "label": "neutral", "confidence": 0.0}

        # Sum weighted impacts
        weighted_sum = 0.0
        total_weight = 0.0

        for event in events:
            impact_score = event["magnitude"] * event["confidence"]

            # Convert impact direction to score
            if event["impact"] == "bullish":
                direction = 1.0
            elif event["impact"] == "bearish":
                direction = -1.0
            else:
                direction = 0.0

            weighted_sum += impact_score * direction
            total_weight += impact_score

        # Calculate normalized score
        if total_weight > 0:
            score = weighted_sum / total_weight
        else:
            score = 0.0

        # Convert score to label
        if score > 0.3:
            label = "bullish"
        elif score < -0.3:
            label = "bearish"
        else:
            label = "neutral"

        # Calculate confidence
        confidence = min(1.0, total_weight / 2.0)

        return {"score": score, "label": label, "confidence": confidence}

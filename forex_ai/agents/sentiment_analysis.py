"""
Sentiment Analysis Agent Implementation for the AI Forex Trading System.

This module provides a specialized agent for sentiment analysis of forex-related news
and market sentiment. It leverages advanced NLP models to analyze news articles,
extract forex entities, and predict market impact.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import json
import logging

from .base import BaseAgent
from .tools import NewsFetcherTool
from forex_ai.utils.logging import get_logger
from forex_ai.exceptions import AgentError

logger = get_logger(__name__)


class SentimentAnalysisAgent(BaseAgent):
    """
    Agent specialized in sentiment analysis of forex-related news.

    This agent analyzes news sentiment, extracts forex entities, and predicts
    market impact of news events using advanced NLP models.
    """

    def initialize(self) -> None:
        """Initialize the sentiment analysis agent with required tools."""
        # Add news fetcher tool
        api_key = self.config.get("news_api_key")
        self.add_tool(NewsFetcherTool(api_key=api_key))

        # Initialize state
        self.memory.update_state("last_analysis_time", None)
        self.memory.update_state("sentiment_history", {})

        # Initialize sentiment analysis components
        try:
            from forex_ai.analysis.sentiment_analysis import (
                SentimentAnalyzer,
                ForexEntityExtractor,
                NewsImpactPredictor,
                RAGNewsProcessor,
            )

            self.processor = RAGNewsProcessor()
            self.analyzer = SentimentAnalyzer()
            self.impact_predictor = NewsImpactPredictor()

            logger.info("Initialized advanced sentiment analysis components")
            self.has_advanced_components = True
        except ImportError as e:
            logger.warning(
                f"Advanced sentiment analysis components not available: {str(e)}"
            )
            self.has_advanced_components = False

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process news data and perform sentiment analysis.

        Args:
            input_data: Dictionary containing analysis request parameters
                Required keys:
                - symbol: Trading symbol or "all" for market-wide sentiment
                Optional keys:
                - from_date: Start date for news analysis (ISO format)
                - limit: Maximum number of news items to analyze
                - include_impact: Whether to include impact prediction
                - min_relevance: Minimum relevance score (0-1)

        Returns:
            Dictionary containing sentiment analysis results
        """
        self.metrics.start_execution()

        try:
            # Extract parameters
            symbol = input_data.get("symbol")
            if not symbol:
                raise AgentError("Symbol is required for sentiment analysis")

            # Extract additional parameters with defaults
            from_date = input_data.get(
                "from_date", (datetime.now() - timedelta(days=3)).isoformat()
            )
            limit = input_data.get("limit", 10)
            include_impact = input_data.get("include_impact", True)
            min_relevance = input_data.get("min_relevance", 0.3)

            logger.info(f"Performing sentiment analysis for {symbol}")

            # Extract currencies from symbol
            currencies = self._extract_currencies(symbol)

            # Fetch and analyze news
            news_analysis = self._analyze_news(
                currencies=currencies,
                from_date=from_date,
                limit=limit,
                min_relevance=min_relevance,
            )

            # Add impact prediction if requested
            if include_impact and self.has_advanced_components:
                impact_analysis = self._predict_impact(news_analysis["articles"])
                news_analysis["impact_prediction"] = impact_analysis

            # Update sentiment history in memory
            self._update_sentiment_history(symbol, news_analysis)

            # Create summarized result
            result = {
                "symbol": symbol,
                "analysis_time": datetime.now().isoformat(),
                "currencies": currencies,
                "sentiment": news_analysis.get("overall_sentiment", "neutral"),
                "confidence": news_analysis.get("sentiment_confidence", 0.5),
                "news_count": len(news_analysis.get("articles", [])),
                "sentiment_by_currency": news_analysis.get("currency_sentiment", {}),
                "top_news": news_analysis.get("articles", [])[
                    :3
                ],  # Include top 3 news items
                "analysis_method": news_analysis.get("analysis_method", "simple"),
            }

            # Add impact prediction summary if available
            if "impact_prediction" in news_analysis:
                result["impact"] = {
                    "probability": news_analysis["impact_prediction"].get(
                        "avg_probability", 0
                    ),
                    "magnitude": news_analysis["impact_prediction"].get(
                        "avg_magnitude", 0
                    ),
                    "affected_pairs": news_analysis["impact_prediction"].get(
                        "affected_pairs", []
                    ),
                    "direction": news_analysis["impact_prediction"].get(
                        "direction", "neutral"
                    ),
                }

            # Generate summary text
            result["summary"] = self._create_analysis_summary(result)

            # Record successful execution
            self.metrics.end_execution("sentiment_analysis")
            self.memory.update_state("last_analysis_time", datetime.now().isoformat())

            return result

        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            self.metrics.record_error("sentiment_analysis_error")
            raise AgentError(f"Failed to perform sentiment analysis: {str(e)}") from e

    def _extract_currencies(self, symbol: str) -> List[str]:
        """
        Extract currency codes from a symbol.

        Args:
            symbol: Symbol to extract currencies from (e.g., "EUR/USD" or "all")

        Returns:
            List of currency codes
        """
        if symbol.lower() == "all":
            # Return major currencies for market-wide analysis
            return ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]

        # Handle currency pairs
        if "/" in symbol:
            base, quote = symbol.split("/")
            return [base, quote]

        # Handle indices or single currencies
        return [symbol]

    def _analyze_news(
        self, currencies: List[str], from_date: str, limit: int, min_relevance: float
    ) -> Dict[str, Any]:
        """
        Analyze news for the given currencies.

        Args:
            currencies: List of currency codes
            from_date: Start date for news (ISO format)
            limit: Maximum number of news items to analyze
            min_relevance: Minimum relevance score (0-1)

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
                    "news_fetcher", query=query, from_date=from_date, limit=limit
                )

                # Add currency to each article
                for article in news:
                    article["currency"] = currency

                all_news.extend(news)

            # Sort by published date and relevance
            all_news.sort(
                key=lambda x: (x.get("forex_relevance", 0), x.get("published_at", "")),
                reverse=True,
            )

            # Filter by minimum relevance
            if min_relevance > 0:
                all_news = [
                    item
                    for item in all_news
                    if item.get("forex_relevance", 0) >= min_relevance
                ]

            # If no news found, return empty result
            if not all_news:
                return {
                    "articles": [],
                    "overall_sentiment": "neutral",
                    "sentiment_confidence": 0.0,
                    "currency_sentiment": {},
                    "analysis_method": "none",
                }

            # Use RAG-enhanced news processing if available
            if self.has_advanced_components:
                try:
                    # Process all news items with the RAG processor
                    processed_news = self.processor.batch_process(all_news)

                    # Extract sentiment and impact predictions
                    currency_sentiment = {}
                    sentiment_scores = {}

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

                        # Store sentiment for each currency
                        if currency not in currency_sentiment:
                            currency_sentiment[currency] = []
                            sentiment_scores[currency] = []

                        currency_sentiment[currency].append(sentiment_label)
                        sentiment_scores[currency].append(sentiment_score)

                    # Calculate overall sentiment per currency
                    currency_sentiment_summary = {}
                    for currency, sentiment_list in currency_sentiment.items():
                        if not sentiment_list:
                            currency_sentiment_summary[currency] = "neutral"
                            continue

                        # Calculate average sentiment score
                        avg_score = sum(sentiment_scores[currency]) / len(
                            sentiment_scores[currency]
                        )

                        if avg_score > 0.2:
                            currency_sentiment_summary[currency] = "bullish"
                        elif avg_score < -0.2:
                            currency_sentiment_summary[currency] = "bearish"
                        else:
                            currency_sentiment_summary[currency] = "neutral"

                    # Determine overall sentiment
                    overall_sentiment = "neutral"
                    if len(currencies) == 2:
                        base_currency = currencies[0]
                        quote_currency = currencies[1]

                        # Calculate relative sentiment between base and quote
                        base_score = sum(
                            sentiment_scores.get(base_currency, [0])
                        ) / max(len(sentiment_scores.get(base_currency, [])), 1)
                        quote_score = sum(
                            sentiment_scores.get(quote_currency, [0])
                        ) / max(len(sentiment_scores.get(quote_currency, [])), 1)

                        # For pairs, sentiment is positive when base strengthens vs quote
                        relative_score = base_score - quote_score

                        if relative_score > 0.2:
                            overall_sentiment = "bullish"
                        elif relative_score < -0.2:
                            overall_sentiment = "bearish"
                        else:
                            overall_sentiment = "neutral"

                    # Calculate confidence based on news volume and sentiment consistency
                    total_articles = len(processed_news)

                    # Higher confidence with more articles and consistent sentiment
                    base_confidence = min(
                        0.5 + (total_articles / 20), 0.9
                    )  # Up to 0.9 for many articles

                    # Adjust based on sentiment consistency
                    all_scores = []
                    for scores in sentiment_scores.values():
                        all_scores.extend(scores)

                    if all_scores:
                        # Calculate standard deviation as a measure of consistency
                        import numpy as np

                        score_std = np.std(all_scores) if len(all_scores) > 1 else 0
                        consistency_factor = max(
                            0, 1 - (score_std * 2)
                        )  # Lower std = higher consistency

                        sentiment_confidence = base_confidence * consistency_factor
                    else:
                        sentiment_confidence = 0.5

                    return {
                        "articles": processed_news,
                        "overall_sentiment": overall_sentiment,
                        "sentiment_confidence": sentiment_confidence,
                        "currency_sentiment": currency_sentiment_summary,
                        "analysis_method": "advanced",
                    }

                except Exception as e:
                    logger.warning(f"Advanced news processing failed: {str(e)}")
                    logger.info("Falling back to simple sentiment analysis")

            # Simple fallback sentiment analysis
            sentiments = {}
            for article in all_news:
                currency = article.get("currency")

                # Get sentiment from article if available, otherwise use title
                sentiment_score = article.get("sentiment", 0)
                sentiment = "neutral"
                if sentiment_score > 0.3:
                    sentiment = "bullish"
                elif sentiment_score < -0.3:
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

            # Calculate confidence based on news volume
            sentiment_confidence = min(
                0.5 + (len(all_news) / 20), 0.8
            )  # Simpler confidence calculation

            return {
                "articles": all_news,
                "overall_sentiment": overall_sentiment,
                "sentiment_confidence": sentiment_confidence,
                "currency_sentiment": currency_sentiment,
                "analysis_method": "simple",
            }

        except Exception as e:
            logger.error(f"Error analyzing news: {str(e)}")
            return {
                "articles": [],
                "overall_sentiment": "neutral",
                "sentiment_confidence": 0.0,
                "currency_sentiment": {},
                "error": str(e),
                "analysis_method": "failed",
            }

    def _predict_impact(self, news_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Predict market impact of news items.

        Args:
            news_items: List of news items

        Returns:
            Dictionary containing impact predictions
        """
        if not self.has_advanced_components or not news_items:
            return {
                "avg_probability": 0.0,
                "avg_magnitude": 0.0,
                "affected_pairs": [],
                "direction": "neutral",
            }

        try:
            # Predict impact for each news item
            impact_probabilities = []
            impact_magnitudes = []
            affected_pairs_map = {}

            for news_item in news_items:
                # Skip items that already have impact prediction
                if (
                    "analysis" in news_item
                    and "impact_prediction" in news_item["analysis"]
                ):
                    impact = news_item["analysis"]["impact_prediction"]
                else:
                    # Predict impact
                    impact = self.impact_predictor.predict(news_item)

                # Store impact metrics
                impact_probabilities.append(impact.get("impact_probability", 0))
                impact_magnitudes.append(impact.get("impact_magnitude", 0))

                # Track affected pairs
                for pair in impact.get("affected_pairs", []):
                    if pair not in affected_pairs_map:
                        affected_pairs_map[pair] = []

                    affected_pairs_map[pair].append(
                        {
                            "probability": impact.get("impact_probability", 0),
                            "magnitude": impact.get("impact_magnitude", 0),
                            "sentiment": news_item.get("analysis", {})
                            .get("sentiment", {})
                            .get("label", "neutral"),
                        }
                    )

            # Calculate average impact metrics
            avg_probability = (
                sum(impact_probabilities) / len(impact_probabilities)
                if impact_probabilities
                else 0
            )
            avg_magnitude = (
                sum(impact_magnitudes) / len(impact_magnitudes)
                if impact_magnitudes
                else 0
            )

            # Sort pairs by total impact (probability * magnitude)
            pair_impact = {}
            for pair, impacts in affected_pairs_map.items():
                total_impact = sum(
                    imp["probability"] * imp["magnitude"] for imp in impacts
                )
                pair_impact[pair] = total_impact

            # Get top affected pairs (sorted by impact)
            top_pairs = sorted(pair_impact.items(), key=lambda x: x[1], reverse=True)
            affected_pairs = [pair for pair, _ in top_pairs[:5]]  # Top 5 pairs

            # Determine overall direction (bullish/bearish)
            # For simplicity, we'll use the most affected pair's sentiment
            direction = "neutral"
            if affected_pairs:
                top_pair = affected_pairs[0]
                pair_sentiments = [
                    imp.get("sentiment", "neutral")
                    for imp in affected_pairs_map[top_pair]
                ]

                bullish_count = pair_sentiments.count(
                    "positive"
                ) + pair_sentiments.count("bullish")
                bearish_count = pair_sentiments.count(
                    "negative"
                ) + pair_sentiments.count("bearish")

                if bullish_count > bearish_count:
                    direction = "bullish"
                elif bearish_count > bullish_count:
                    direction = "bearish"

            return {
                "avg_probability": avg_probability,
                "avg_magnitude": avg_magnitude,
                "affected_pairs": affected_pairs,
                "direction": direction,
                "pair_impacts": pair_impact,
            }

        except Exception as e:
            logger.error(f"Error predicting impact: {str(e)}")
            return {
                "avg_probability": 0.0,
                "avg_magnitude": 0.0,
                "affected_pairs": [],
                "direction": "neutral",
                "error": str(e),
            }

    def _update_sentiment_history(self, symbol: str, analysis: Dict[str, Any]) -> None:
        """
        Update sentiment history in memory.

        Args:
            symbol: Symbol being analyzed
            analysis: Analysis results
        """
        history = self.memory.get_state("sentiment_history") or {}

        if symbol not in history:
            history[symbol] = []

        # Add new entry
        history[symbol].append(
            {
                "timestamp": datetime.now().isoformat(),
                "sentiment": analysis.get("overall_sentiment", "neutral"),
                "confidence": analysis.get("sentiment_confidence", 0.0),
            }
        )

        # Keep only last 10 entries
        history[symbol] = history[symbol][-10:]

        # Update memory
        self.memory.update_state("sentiment_history", history)

    def _create_analysis_summary(self, result: Dict[str, Any]) -> str:
        """
        Create a human-readable summary of the sentiment analysis.

        Args:
            result: Analysis result

        Returns:
            Summary text
        """
        symbol = result.get("symbol", "unknown")
        sentiment = result.get("sentiment", "neutral")
        confidence = result.get("confidence", 0.0)
        news_count = result.get("news_count", 0)

        # Create summary
        summary = f"Sentiment Analysis Summary for {symbol}:\n\n"

        # Overall sentiment
        summary += (
            f"Overall Sentiment: {sentiment.title()} (Confidence: {confidence:.2f})\n"
        )
        summary += f"Based on {news_count} news articles\n\n"

        # Sentiment by currency
        currency_sentiment = result.get("sentiment_by_currency", {})
        if currency_sentiment:
            summary += "Sentiment by Currency:\n"
            for currency, sent in currency_sentiment.items():
                summary += f"- {currency}: {sent.title()}\n"
            summary += "\n"

        # Impact prediction
        impact = result.get("impact")
        if impact:
            probability = impact.get("probability", 0)
            magnitude = impact.get("magnitude", 0)
            direction = impact.get("direction", "neutral")
            affected_pairs = impact.get("affected_pairs", [])

            summary += f"Market Impact Prediction:\n"
            summary += f"- Probability: {probability:.2f}\n"
            summary += f"- Magnitude: {magnitude:.2f}\n"
            summary += f"- Direction: {direction.title()}\n"

            if affected_pairs:
                summary += f"- Most affected pairs: {', '.join(affected_pairs)}\n"

            summary += "\n"

        # Top news
        top_news = result.get("top_news", [])
        if top_news:
            summary += "Top Recent News:\n"
            for i, news in enumerate(top_news, 1):
                title = news.get("title", "No title")
                source = news.get("source", "Unknown")
                summary += f"{i}. {title} ({source})\n"

        return summary

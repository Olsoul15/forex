"""
Social Media Sentiment Analysis for Forex AI Trading System.

This module provides specialized sentiment analysis for social media content
related to forex trading, with specific handling for platforms like Twitter,
Reddit, and StockTwits.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import numpy as np

from forex_ai.exceptions import AnalysisError, ModelError, ModelLoadingError
from forex_ai.core.exceptions import AnalysisError as CoreAnalysisError
from forex_ai.models.controller import get_model_controller

logger = logging.getLogger(__name__)

# Flag to indicate whether ML libraries are available
ML_LIBRARIES_AVAILABLE = False
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    ML_LIBRARIES_AVAILABLE = True
    logger.info(
        "ML libraries (torch, transformers) are available for social media sentiment analysis."
    )
except ImportError:
    logger.warning(
        "ML libraries (torch, transformers) are not available. Using fallback implementations for social media sentiment analysis."
    )


class SocialMediaSentimentAnalyzer:
    """
    Specialized sentiment analyzer for social media content related to forex trading.
    Handles Twitter, Reddit, StockTwits and other social platforms with specialized
    processing for the unique language patterns found in these sources.
    """

    def __init__(self, model_id: str = "social_sentiment"):
        """
        Initialize the social media sentiment analyzer.

        Args:
            model_id: ID of the model to use (default: "social_sentiment")
        """
        self.model_id = model_id
        self.model_controller = get_model_controller()

        # Register model configuration if not already registered
        if self.model_id not in self.model_controller._model_configs:
            self.model_controller._model_configs[self.model_id] = {
                "name": "cardiffnlp/twitter-roberta-base-sentiment",
                "type": "sentiment",
                "module_path": "forex_ai.analysis.sentiment",
                "class_name": "SentimentModel",
            }

        # Special tokens and patterns for social media
        self.special_patterns = {
            "twitter": {
                "cashtags": re.compile(r"\$[A-Za-z]+"),
                "hashtags": re.compile(r"#[A-Za-z0-9_]+"),
                "mentions": re.compile(r"@[A-Za-z0-9_]+"),
            },
            "reddit": {
                "subreddits": re.compile(r"r/[A-Za-z0-9_]+"),
                "users": re.compile(r"u/[A-Za-z0-9_]+"),
            },
            "stocktwits": {
                "cashtags": re.compile(r"\$[A-Za-z.]+"),
                "bullish": re.compile(
                    r"\b(bullish|long|calls|to the moon|🚀|💎)\b", re.IGNORECASE
                ),
                "bearish": re.compile(
                    r"\b(bearish|short|puts|drilling|💩|🐻)\b", re.IGNORECASE
                ),
            },
        }

        # Emoji sentiment mapping (simplified)
        self.emoji_sentiment = {
            "🚀": 1.0,  # very positive
            "💎": 0.8,  # positive
            "📈": 0.7,  # positive
            "🐂": 0.6,  # positive (bull)
            "👍": 0.5,  # positive
            "😊": 0.4,  # positive
            "🤔": 0.0,  # neutral
            "😐": 0.0,  # neutral
            "👎": -0.5,  # negative
            "😡": -0.6,  # negative
            "📉": -0.7,  # negative
            "🐻": -0.8,  # negative (bear)
            "💩": -1.0,  # very negative
        }

        # Load the model if ML libraries are available
        if ML_LIBRARIES_AVAILABLE:
            self._load_model()
        else:
            logger.warning(
                "ML libraries not available. Social media sentiment analyzer will use rule-based methods."
            )

    def _load_model(self):
        """Load the pre-trained model and tokenizer."""
        try:
            logger.info(f"Loading social media sentiment model: {self.model_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_controller._model_configs[self.model_id]["name"]
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_controller._model_configs[self.model_id]["name"]
            )
            self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            logger.info("Social media sentiment model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load social media sentiment model: {e}")
            raise ModelLoadingError(f"Failed to load social media sentiment model: {e}")

    def preprocess_social_text(self, text: str, source: str = None) -> str:
        """
        Preprocess social media text to improve sentiment analysis accuracy.

        Args:
            text: The raw social media text
            source: The source platform (twitter, reddit, stocktwits)

        Returns:
            Preprocessed text
        """
        if not text:
            return ""

        # Apply source-specific preprocessing if specified
        if source and source.lower() in self.special_patterns:
            patterns = self.special_patterns[source.lower()]

            # Extract special tokens for later analysis
            for pattern_name, pattern in patterns.items():
                # Save matches for later but remove/normalize for the NLP model
                if pattern_name in ["cashtags", "hashtags"]:
                    text = pattern.sub(" ", text)
                elif pattern_name in ["mentions", "users", "subreddits"]:
                    text = pattern.sub(" ", text)

        # General social media preprocessing
        # Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        # Convert HTML entities
        text = re.sub(r"&amp;", "&", text)
        # Handle repeated characters (e.g., "sooooo good" -> "so good")
        text = re.sub(r"(.)\1{2,}", r"\1\1", text)
        # Handle all caps (often indicates shouting/emphasis)
        if text.isupper():
            text = text.lower() + " [EMPHASIS]"

        return text

    def extract_symbols(self, text: str, source: str = None) -> List[str]:
        """
        Extract financial symbols (cashtags) from social media text.

        Args:
            text: The social media text
            source: The source platform

        Returns:
            List of extracted symbols
        """
        symbols = []

        # Use source-specific pattern if available
        if source and source.lower() in self.special_patterns:
            if "cashtags" in self.special_patterns[source.lower()]:
                pattern = self.special_patterns[source.lower()]["cashtags"]
                symbols = [m.group(0)[1:] for m in pattern.finditer(text)]
        else:
            # Generic cashtag pattern as fallback
            symbols = [m.group(0)[1:] for m in re.finditer(r"\$([A-Za-z.]+)", text)]

        # Handle forex pairs specifically - look for major currencies
        currency_patterns = [
            (r"\bEUR/USD\b", "EURUSD"),
            (r"\bGBP/USD\b", "GBPUSD"),
            (r"\bUSD/JPY\b", "USDJPY"),
            (r"\bAUD/USD\b", "AUDUSD"),
            (r"\bUSD/CAD\b", "USDCAD"),
            (r"\bUSD/CHF\b", "USDCHF"),
            (r"\bNZD/USD\b", "NZDUSD"),
        ]

        for pattern, symbol in currency_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                symbols.append(symbol)

        return list(set(symbols))  # Remove duplicates

    def analyze(self, posts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze sentiment of social media posts.

        Args:
            posts: List of social media posts with text and metadata

        Returns:
            Dict with sentiment analysis results

        Raises:
            AnalysisError: If analysis fails
        """
        try:
            # Get model through controller
            model = self.model_controller.get_model(self.model_id)

            # Extract text from posts
            texts = [post.get("text", "") for post in posts]

            # Process in batches
            results = []
            for i in range(0, len(texts), 32):  # Process 32 texts at a time
                batch = texts[i : i + 32]
                batch_results = model.predict(batch)
                results.extend(batch_results)

            # Combine results with post metadata
            analyzed_posts = []
            for post, result in zip(posts, results):
                analyzed_posts.append(
                    {
                        **post,
                        "sentiment": result["sentiment"],
                        "probabilities": result["probabilities"],
                    }
                )

            return {"posts": analyzed_posts, "timestamp": datetime.now().isoformat()}

        except Exception as e:
            logger.error(f"Social media sentiment analysis failed: {str(e)}")
            raise CoreAnalysisError(
                f"Failed to analyze social media sentiment: {str(e)}"
            )

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get model performance metrics.

        Returns:
            Dict with model metrics
        """
        return self.model_controller.get_model_status(self.model_id)["metrics"]


class SocialMediaConnector:
    """
    Connector for retrieving data from various social media platforms.
    """

    def __init__(self, api_keys: Dict[str, str] = None):
        """
        Initialize the social media connector.

        Args:
            api_keys: Dictionary of API keys for different platforms
        """
        self.api_keys = api_keys or {}
        self.available_sources = self._check_available_sources()

    def _check_available_sources(self) -> Dict[str, bool]:
        """Check which social media sources are available based on API keys."""
        return {
            "twitter": "twitter" in self.api_keys,
            "reddit": "reddit" in self.api_keys,
            "stocktwits": "stocktwits" in self.api_keys,
        }

    def get_twitter_data(self, query: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get tweets related to forex trading.

        Args:
            query: Search query (e.g., "EURUSD", "forex")
            limit: Maximum number of tweets to retrieve

        Returns:
            List of tweets with metadata
        """
        if not self.available_sources.get("twitter"):
            logger.warning("Twitter API key not available")
            return []

        # Placeholder implementation - would use tweepy or Twitter API in real implementation
        return [
            {
                "id": f"placeholder_{i}",
                "text": f"Sample tweet about {query} #{i}",
                "created_at": "2023-01-01T00:00:00Z",
                "user": {"screen_name": f"user_{i}", "followers_count": i * 10},
                "retweet_count": i,
                "like_count": i * 2,
                "source": "twitter",
            }
            for i in range(min(10, limit))
        ]

    def get_reddit_data(
        self, subreddit: str = "Forex", limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get posts from forex-related subreddits.

        Args:
            subreddit: Subreddit to query
            limit: Maximum number of posts to retrieve

        Returns:
            List of Reddit posts with metadata
        """
        if not self.available_sources.get("reddit"):
            logger.warning("Reddit API key not available")
            return []

        # Placeholder implementation - would use PRAW in real implementation
        return [
            {
                "id": f"placeholder_{i}",
                "title": f"Sample post in r/{subreddit} #{i}",
                "selftext": f"This is the content of post #{i} discussing forex trading.",
                "created_utc": 1672531200,  # 2023-01-01 00:00:00 UTC
                "author": f"reddit_user_{i}",
                "score": i * 5,
                "num_comments": i * 3,
                "source": "reddit",
            }
            for i in range(min(10, limit))
        ]

    def get_stocktwits_data(
        self, symbol: str = "EURUSD", limit: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Get messages from StockTwits related to forex symbols.

        Args:
            symbol: Symbol to query
            limit: Maximum number of messages to retrieve

        Returns:
            List of StockTwits messages with metadata
        """
        if not self.available_sources.get("stocktwits"):
            logger.warning("StockTwits API key not available")
            return []

        # Placeholder implementation
        return [
            {
                "id": i,
                "body": f"StockTwits message about ${symbol} #{i}",
                "created_at": "2023-01-01T00:00:00Z",
                "user": {"username": f"stocktwits_user_{i}", "followers": i * 8},
                "symbols": [symbol],
                "source": "stocktwits",
            }
            for i in range(min(10, limit))
        ]

    def search_all_platforms(
        self, query: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search across all available platforms for forex-related content.

        Args:
            query: Search query
            limit: Maximum total results

        Returns:
            Combined list of social media content
        """
        results = []
        per_source_limit = limit // max(1, sum(self.available_sources.values()))

        if self.available_sources.get("twitter"):
            results.extend(self.get_twitter_data(query, per_source_limit))

        if self.available_sources.get("reddit"):
            results.extend(self.get_reddit_data("Forex", per_source_limit))

        if self.available_sources.get("stocktwits"):
            # Convert forex pair format if needed
            if "/" in query:
                symbol = query.replace("/", "")
            else:
                symbol = query
            results.extend(self.get_stocktwits_data(symbol, per_source_limit))

        # Sort by most recent first
        # In a real implementation, this would use actual timestamps
        return results[:limit]


class SocialSentimentAggregator:
    """
    Aggregates and analyzes sentiment from social media sources for forex trading.
    """

    def __init__(
        self,
        connector: Optional[SocialMediaConnector] = None,
        analyzer: Optional[SocialMediaSentimentAnalyzer] = None,
    ):
        """
        Initialize the social sentiment aggregator.

        Args:
            connector: SocialMediaConnector instance
            analyzer: SocialMediaSentimentAnalyzer instance
        """
        self.connector = connector or SocialMediaConnector()
        self.analyzer = analyzer or SocialMediaSentimentAnalyzer()

    def get_sentiment_for_currency_pair(
        self, pair: str, timeframe: str = "recent"
    ) -> Dict[str, Any]:
        """
        Get aggregated sentiment for a specific currency pair.

        Args:
            pair: Currency pair (e.g., "EUR/USD")
            timeframe: Timeframe for analysis ("recent", "daily", "weekly")

        Returns:
            Dictionary with sentiment metrics
        """
        # Get social media data
        social_data = self.connector.search_all_platforms(pair)

        if not social_data:
            return {
                "pair": pair,
                "sentiment_score": 0,
                "sentiment": "neutral",
                "confidence": 0,
                "volume": 0,
                "source_breakdown": {},
                "timeframe": timeframe,
            }

        # Analyze sentiment for each post
        texts = [
            item.get("text", item.get("body", item.get("selftext", "")))
            for item in social_data
        ]
        sources = [item.get("source") for item in social_data]

        sentiment_results = self.analyzer.analyze(social_data)

        # Process results
        sentiment_scores = [result["sentiment_score"] for result in sentiment_results]
        weighted_scores = []
        source_counts = {"twitter": 0, "reddit": 0, "stocktwits": 0}
        source_scores = {"twitter": [], "reddit": [], "stocktwits": []}

        # Calculate weighted scores based on engagement
        for i, (result, data) in enumerate(zip(sentiment_results, social_data)):
            weight = 1.0  # Base weight

            # Apply weights based on engagement metrics
            if data.get("source") == "twitter":
                followers = data.get("user", {}).get("followers_count", 0)
                retweets = data.get("retweet_count", 0)
                likes = data.get("like_count", 0)
                weight = 1.0 + min(1.0, (followers / 10000) + (retweets + likes) / 100)
                source_counts["twitter"] += 1
                source_scores["twitter"].append(result["sentiment_score"])

            elif data.get("source") == "reddit":
                score = data.get("score", 0)
                comments = data.get("num_comments", 0)
                weight = 1.0 + min(1.0, (score + comments) / 50)
                source_counts["reddit"] += 1
                source_scores["reddit"].append(result["sentiment_score"])

            elif data.get("source") == "stocktwits":
                followers = data.get("user", {}).get("followers", 0)
                weight = 1.0 + min(1.0, followers / 5000)
                source_counts["stocktwits"] += 1
                source_scores["stocktwits"].append(result["sentiment_score"])

            weighted_scores.append(result["sentiment_score"] * weight)

        # Calculate aggregated metrics
        if weighted_scores:
            overall_score = sum(weighted_scores) / sum(1.0 for _ in weighted_scores)
            confidence = min(0.95, 0.5 + (len(sentiment_results) / 100))

            # Determine overall sentiment
            if overall_score > 0.1:
                sentiment = "positive"
            elif overall_score < -0.1:
                sentiment = "negative"
            else:
                sentiment = "neutral"

            # Calculate source breakdown
            source_breakdown = {}
            for source in source_counts:
                if source_counts[source] > 0:
                    source_breakdown[source] = {
                        "count": source_counts[source],
                        "sentiment": sum(source_scores[source])
                        / max(1, len(source_scores[source])),
                    }
        else:
            overall_score = 0
            sentiment = "neutral"
            confidence = 0
            source_breakdown = {}

        return {
            "pair": pair,
            "sentiment_score": round(overall_score, 2),
            "sentiment": sentiment,
            "confidence": round(confidence, 2),
            "volume": len(sentiment_results),
            "source_breakdown": source_breakdown,
            "timeframe": timeframe,
        }

    def get_market_social_sentiment(self, pairs: List[str] = None) -> Dict[str, Any]:
        """
        Get overall market sentiment from social media for multiple currency pairs.

        Args:
            pairs: List of currency pairs to analyze

        Returns:
            Dictionary with market sentiment metrics
        """
        if pairs is None:
            pairs = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD"]

        pair_sentiments = {}
        for pair in pairs:
            pair_sentiments[pair] = self.get_sentiment_for_currency_pair(pair)

        # Calculate overall market sentiment
        sentiment_scores = [
            data["sentiment_score"] for data in pair_sentiments.values()
        ]

        if sentiment_scores:
            market_score = sum(sentiment_scores) / len(sentiment_scores)

            if market_score > 0.1:
                market_sentiment = "positive"
            elif market_score < -0.1:
                market_sentiment = "negative"
            else:
                market_sentiment = "neutral"
        else:
            market_score = 0
            market_sentiment = "neutral"

        return {
            "market_sentiment": market_sentiment,
            "market_score": round(market_score, 2),
            "pair_data": pair_sentiments,
            "timestamp": datetime.now().isoformat(),
            "total_sources_analyzed": sum(
                data["volume"] for data in pair_sentiments.values()
            ),
        }

"""
Social Media Sentiment Analysis Example for Forex Trading.

This example demonstrates how to use the social media sentiment analysis
components to analyze forex-related sentiment from various social platforms.
"""

import os
import logging
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any

from forex_ai.analysis.social_media_sentiment import (
    SocialMediaSentimentAnalyzer,
    SocialMediaConnector,
    SocialSentimentAggregator,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def basic_sentiment_example():
    """Demonstrate basic social media sentiment analysis."""
    logger.info("Running basic social media sentiment analysis example")

    # Initialize the sentiment analyzer
    analyzer = SocialMediaSentimentAnalyzer(use_gpu=False)

    # Example social media texts
    texts = [
        "EUR/USD is looking bullish today! Breaking resistance at 1.12 🚀",
        "Bearish on GBPUSD, BoE policy decision could push it lower 📉",
        "USDJPY showing weakness, might retest support level soon.",
        "Anyone else thinking EURUSD is going to moon? 💎🙌",
        "$EURUSD shorts are getting squeezed, might be time to go long!",
    ]

    sources = ["twitter", "reddit", "stocktwits", "twitter", "stocktwits"]

    # Analyze individual texts
    logger.info("Analyzing individual texts:")
    for i, (text, source) in enumerate(zip(texts, sources)):
        result = analyzer.analyze(text, source)
        logger.info(
            f"Text {i+1} ({source}): {result['sentiment']} (score: {result['score']:.2f}, confidence: {result['confidence']:.2f})"
        )
        if result["symbols"]:
            logger.info(f"  Detected symbols: {', '.join(result['symbols'])}")

    # Batch analyze
    batch_results = analyzer.batch_analyze(texts, sources)
    logger.info(f"\nBatch analysis complete - {len(batch_results)} texts processed")


def social_media_connector_example():
    """Demonstrate retrieving data from social media platforms."""
    logger.info("\nRunning social media connector example")

    # In a real implementation, you would provide API keys
    api_keys = {
        # Normally you would get these from environment variables
        # 'twitter': os.environ.get('TWITTER_API_KEY'),
        # 'reddit': os.environ.get('REDDIT_API_KEY'),
        # 'stocktwits': os.environ.get('STOCKTWITS_API_KEY')
    }

    # Initialize connector (will use placeholder data since no real API keys provided)
    connector = SocialMediaConnector(api_keys)

    # Get data for EUR/USD
    logger.info("Retrieving social media data for EUR/USD:")

    # Since we don't have real API keys, this will return placeholder data
    results = connector.search_all_platforms("EUR/USD", limit=30)

    logger.info(f"Retrieved {len(results)} social media posts")

    # Show data sources breakdown
    sources = {}
    for item in results:
        source = item.get("source", "unknown")
        sources[source] = sources.get(source, 0) + 1

    for source, count in sources.items():
        logger.info(f"  {source}: {count} posts")


def sentiment_aggregation_example():
    """Demonstrate aggregating sentiment across social media platforms."""
    logger.info("\nRunning sentiment aggregation example")

    # Initialize components
    connector = SocialMediaConnector()
    analyzer = SocialMediaSentimentAnalyzer()
    aggregator = SocialSentimentAggregator(connector, analyzer)

    # Analyze sentiment for major currency pairs
    pairs = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD"]

    logger.info("Analyzing social sentiment for major currency pairs:")

    # Get sentiment for each pair
    pair_results = {}
    for pair in pairs:
        pair_results[pair] = aggregator.get_sentiment_for_currency_pair(pair)
        sentiment = pair_results[pair]["sentiment"]
        score = pair_results[pair]["sentiment_score"]
        volume = pair_results[pair]["volume"]
        logger.info(f"  {pair}: {sentiment} (score: {score}, volume: {volume})")

    # Get overall market sentiment
    market_sentiment = aggregator.get_market_social_sentiment(pairs)
    logger.info(
        f"\nOverall market social sentiment: {market_sentiment['market_sentiment']} (score: {market_sentiment['market_score']})"
    )
    logger.info(f"Total sources analyzed: {market_sentiment['total_sources_analyzed']}")


def visualize_sentiment(pair_results: Dict[str, Any]):
    """Visualize sentiment results."""
    # Create DataFrame for visualization
    data = {"Pair": [], "Sentiment Score": [], "Volume": []}

    for pair, result in pair_results.items():
        data["Pair"].append(pair)
        data["Sentiment Score"].append(result["sentiment_score"])
        data["Volume"].append(result["volume"])

    df = pd.DataFrame(data)

    # Create visualization
    plt.figure(figsize=(12, 6))

    # Sentiment scores plot
    ax1 = plt.subplot(1, 2, 1)
    bars = ax1.bar(
        df["Pair"],
        df["Sentiment Score"],
        color=["green" if s > 0 else "red" for s in df["Sentiment Score"]],
    )
    ax1.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax1.set_title("Social Media Sentiment by Currency Pair")
    ax1.set_ylabel("Sentiment Score (-1 to 1)")
    ax1.set_ylim(-1, 1)

    # Volume plot
    ax2 = plt.subplot(1, 2, 2)
    ax2.bar(df["Pair"], df["Volume"], color="blue", alpha=0.7)
    ax2.set_title("Social Media Volume by Currency Pair")
    ax2.set_ylabel("Number of Posts")

    plt.tight_layout()
    plt.savefig("social_sentiment_analysis.png")
    logger.info("Visualization saved to 'social_sentiment_analysis.png'")


def run_full_example():
    """Run the complete social media sentiment analysis example."""
    logger.info("=== FOREX AI SOCIAL MEDIA SENTIMENT ANALYSIS EXAMPLE ===")

    # Basic sentiment analysis
    basic_sentiment_example()

    # Social media connector
    social_media_connector_example()

    # Full sentiment aggregation
    logger.info("\nRunning full sentiment aggregation and visualization example")

    # Initialize components
    aggregator = SocialSentimentAggregator()

    # Get sentiment for currency pairs
    pairs = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD"]

    # Get results for each pair
    pair_results = {}
    for pair in pairs:
        pair_results[pair] = aggregator.get_sentiment_for_currency_pair(pair)

    # Visualize results
    try:
        visualize_sentiment(pair_results)
    except Exception as e:
        logger.error(f"Visualization failed: {e}")

    # Get market sentiment
    market_sentiment = aggregator.get_market_social_sentiment(pairs)

    logger.info(
        f"\nFinal market social media sentiment: {market_sentiment['market_sentiment']}"
    )
    logger.info(f"Analysis complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    run_full_example()

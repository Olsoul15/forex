"""
News fetcher module for Forex AI Trading System.

This module fetches financial news from Alpha Vantage and stores it in the database.
"""

import os
import logging
import re
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import requests

from forex_ai.data.storage.supabase_client import get_supabase_db_client
from forex_ai.config.settings import get_settings
from forex_ai.exceptions import DataSourceError, DatabaseError
# Import advanced sentiment analysis tools
from forex_ai.analysis.sentiment_analysis import SentimentAnalyzer, ForexEntityExtractor

logger = logging.getLogger(__name__)

# List of forex currency pairs to track
CURRENCY_PAIRS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", 
    "USD/CAD", "AUD/USD", "NZD/USD", "EUR/GBP",
    "EUR/JPY", "GBP/JPY"
]

# Match currency pairs in text
CURRENCY_PAIR_REGEX = re.compile(r'\b(' + '|'.join(CURRENCY_PAIRS) + r')\b')

# Initialize sentiment analyzer (lazy loading)
_sentiment_analyzer = None
_entity_extractor = None

def get_sentiment_analyzer():
    """Get or create the sentiment analyzer instance."""
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        try:
            _sentiment_analyzer = SentimentAnalyzer(use_gpu=False)
            logger.info("Initialized advanced sentiment analyzer")
        except Exception as e:
            logger.error(f"Failed to initialize sentiment analyzer: {str(e)}")
            logger.warning("Falling back to simple sentiment analysis")
            _sentiment_analyzer = "simple"
    return _sentiment_analyzer

def get_entity_extractor():
    """Get or create the entity extractor instance."""
    global _entity_extractor
    if _entity_extractor is None:
        try:
            _entity_extractor = ForexEntityExtractor()
            logger.info("Initialized forex entity extractor")
        except Exception as e:
            logger.error(f"Failed to initialize entity extractor: {str(e)}")
            _entity_extractor = None
    return _entity_extractor

def mask_api_key(api_key: str) -> str:
    """
    Mask an API key for safe display.
    
    Args:
        api_key: API key to mask.
        
    Returns:
        Masked API key.
    """
    if not api_key or api_key == "placeholder":
        return "[NOT SET]"
    
    # Show first 4 and last 4 characters, mask the rest
    if len(api_key) <= 8:
        return "****"
    return api_key[:4] + "****" + api_key[-4:]

def extract_currencies(text: str) -> List[str]:
    """
    Extract currency pairs mentioned in text.
    
    Args:
        text: Text to extract currencies from.
        
    Returns:
        List of currency pairs.
    """
    # Try to use the advanced entity extractor if available
    entity_extractor = get_entity_extractor()
    if entity_extractor:
        try:
            currencies = entity_extractor.extract_currencies(text)
            
            # Convert individual currencies to pairs if needed
            currency_pairs = []
            if len(currencies) >= 2:
                # Create logical pairs from extracted currencies
                for i, base in enumerate(currencies):
                    for quote in currencies[i+1:]:
                        # Only use standard pairs, not all combinations
                        pair = f"{base}/{quote}"
                        if pair in CURRENCY_PAIRS:
                            currency_pairs.append(pair)
                        
                        # Check reverse pair
                        reverse_pair = f"{quote}/{base}"
                        if reverse_pair in CURRENCY_PAIRS:
                            currency_pairs.append(reverse_pair)
            
            # If we found pairs, return them
            if currency_pairs:
                return list(set(currency_pairs))
                
            # If we only have individual currencies, return as is
            return list(set(currencies))
        except Exception as e:
            logger.error(f"Error using entity extractor: {str(e)}")
    
    # Fallback to simple regex matching
    if not text:
        return []
        
    # Find all currency pairs in the text
    matches = CURRENCY_PAIR_REGEX.findall(text)
    
    # Remove duplicates and return
    return list(set(matches))

def calculate_importance(news_item: Dict[str, Any]) -> int:
    """
    Calculate importance score (1-10) for a news item.
    
    Args:
        news_item: News item to analyze.
        
    Returns:
        Importance score from 1 to 10.
    """
    score = 5  # Default middle importance
    
    # Factors that increase importance
    title = news_item.get("title", "").lower()
    description = news_item.get("summary", "").lower() 
    combined_text = title + " " + description
    
    # Keywords that indicate high importance
    high_impact_keywords = [
        "federal reserve", "fed ", "rate decision", "rate hike", "interest rate",
        "central bank", "ecb", "boe", "bank of japan", "rba", "fomc",
        "nonfarm payroll", "gdp", "inflation", "cpi", "unemployment",
        "breaking", "breaking news", "urgent", "flash"
    ]
    
    # Count high impact keywords
    keyword_count = sum(1 for keyword in high_impact_keywords if keyword in combined_text)
    
    # Adjust score based on keyword count
    score += min(keyword_count, 4)  # Max +4 for keywords
    
    # Adjust for currency pair mentions
    currency_count = len(extract_currencies(combined_text))
    if currency_count > 0:
        score += min(currency_count, 2)  # Max +2 for currency mentions
    
    # Ensure score is within bounds
    return max(1, min(score, 10))

def analyze_sentiment(text: str) -> float:
    """
    Analyze sentiment of text.
    
    This function uses the advanced SentimentAnalyzer when available, with fallback
    to a simple rule-based implementation.
    
    Args:
        text: Text to analyze.
        
    Returns:
        Sentiment score from -1 (negative) to 1 (positive).
    """
    # Try to use the advanced sentiment analyzer
    analyzer = get_sentiment_analyzer()
    if analyzer and analyzer != "simple":
        try:
            result = analyzer.analyze(text)
            return result["score"]
        except Exception as e:
            logger.error(f"Advanced sentiment analysis failed: {str(e)}")
            logger.warning("Falling back to simple sentiment analysis")
    
    # Simple rule-based fallback implementation
    positive_words = ["bullish", "rise", "gain", "positive", "up", "growth", "strong", 
                     "rally", "improve", "optimistic", "recovery", "surge"]
    negative_words = ["bearish", "fall", "drop", "decline", "negative", "down", "weak", 
                     "collapse", "worsen", "pessimistic", "recession", "plunge"]
    
    text = text.lower()
    
    # Count occurrences of positive and negative words
    pos_count = sum(1 for word in positive_words if word in text)
    neg_count = sum(1 for word in negative_words if word in text)
    
    # Calculate sentiment
    if pos_count == 0 and neg_count == 0:
        return 0.0  # Neutral
    
    return (pos_count - neg_count) / (pos_count + neg_count)

def fetch_alpha_vantage_news() -> List[Dict[str, Any]]:
    """
    Fetch news from Alpha Vantage.
    
    Returns:
        List of news items.
        
    Raises:
        DataSourceError: If news fetching fails.
    """
    settings = get_settings()
    api_key = settings.ALPHA_VANTAGE_API_KEY
    
    # Check for placeholder or empty API key
    if not api_key or api_key == "placeholder":
        logger.warning("Alpha Vantage API key is not properly set. Using placeholder data.")
        return get_placeholder_news()
    
    logger.info(f"Using Alpha Vantage API key: {mask_api_key(api_key)}")
    
    # Build the Alpha Vantage News API URL with forex tickers
    tickers = "FOREX:EUR/USD,FOREX:GBP/USD,FOREX:USD/JPY,FOREX:USD/CHF,FOREX:AUD/USD"
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={tickers}&apikey={api_key}"
    
    try:
        logger.info(f"Fetching news from Alpha Vantage")
        response = requests.get(url, timeout=10)
        
        # Handle HTTP errors
        if response.status_code != 200:
            logger.error(f"HTTP error {response.status_code}: {response.text}")
            if response.status_code == 429:
                logger.error("API rate limit exceeded")
            raise DataSourceError(f"Alpha Vantage API returned HTTP error {response.status_code}")
            
        # Parse response
        try:
            data = response.json()
        except json.JSONDecodeError:
            logger.error(f"Failed to parse response as JSON: {response.text[:100]}")
            raise DataSourceError("Invalid response format from Alpha Vantage")
        
        # Check for error responses
        if "Information" in data:
            error_msg = data.get("Information", "Unknown error")
            logger.error(f"Alpha Vantage API error: {error_msg}")
            raise DataSourceError(f"Alpha Vantage API error: {error_msg}")
            
        if "feed" not in data:
            logger.error(f"Alpha Vantage API returned unexpected response structure: {list(data.keys())}")
            if "Note" in data:
                note = data["Note"]
                logger.error(f"API message: {note}")
                if "API call frequency" in note:
                    logger.error("API call frequency exceeded")
            raise DataSourceError("Alpha Vantage API returned unexpected data format")
            
        # Process news items
        news_items = []
        for item in data["feed"]:
            title = item.get("title", "")
            summary = item.get("summary", "")
            combined_text = title + " " + summary
            
            # Use Alpha Vantage's sentiment if available, otherwise calculate our own
            if "overall_sentiment_score" in item:
                sentiment = float(item.get("overall_sentiment_score", 0))
            else:
                sentiment = analyze_sentiment(combined_text)
                
            # Extract currencies mentioned
            currencies = extract_currencies(combined_text)
            
            # If no currencies found in text but the item is tagged with ticker symbols,
            # extract currencies from the ticker symbols
            if not currencies and "ticker_sentiment" in item:
                for ticker_data in item["ticker_sentiment"]:
                    ticker = ticker_data.get("ticker", "")
                    if ticker.startswith("FOREX:"):
                        pair = ticker.replace("FOREX:", "")
                        if pair in CURRENCY_PAIRS:
                            currencies.append(pair)
            
            news_item = {
                "title": title,
                "content": summary,
                "source": item.get("source", "Alpha Vantage"),
                "url": item.get("url", ""),
                "published_at": item.get("time_published", datetime.now().isoformat()),
                "currencies": currencies,
                "sentiment": sentiment,
                "importance": calculate_importance(item),
            }
            news_items.append(news_item)
            
        if not news_items:
            logger.warning("No news items found in the response")
            
        logger.info(f"Fetched {len(news_items)} news items from Alpha Vantage")
        return news_items
        
    except requests.exceptions.Timeout:
        logger.error("Request to Alpha Vantage timed out")
        raise DataSourceError("Request to Alpha Vantage timed out")
    except requests.exceptions.ConnectionError:
        logger.error("Connection error when connecting to Alpha Vantage")
        raise DataSourceError("Could not connect to Alpha Vantage")
    except Exception as e:
        logger.error(f"Failed to fetch news from Alpha Vantage: {str(e)}")
        if isinstance(e, DataSourceError):
            raise
        raise DataSourceError(f"Failed to fetch news from Alpha Vantage: {str(e)}")

def store_news_in_database(news_items: List[Dict[str, Any]]) -> int:
    """
    Store news items in the database.
    
    Args:
        news_items: List of news items to store.
        
    Returns:
        Number of items stored.
        
    Raises:
        DatabaseError: If storing news fails.
    """
    if not news_items:
        logger.info("No news items to store")
        return 0
        
    try:
        client = get_supabase_db_client()
        
        # Check for existing URLs to avoid duplicates
        stored_count = 0
        
        for item in news_items:
            # Check if this news item already exists
            if item["url"]:
                existing = client.fetch_one(
                    "news_data",
                    where={"url": item["url"]}
                )
                
                if existing:
                    logger.debug(f"News item already exists: {item['title']}")
                    continue
            
            # Insert the news item
            client.insert_one("news_data", item)
            stored_count += 1
            
        logger.info(f"Stored {stored_count} news items in database")
        return stored_count
        
    except Exception as e:
        logger.error(f"Failed to store news in database: {str(e)}")
        raise DatabaseError(f"Failed to store news in database: {str(e)}")

def fetch_and_store_news() -> int:
    """
    Fetch news from Alpha Vantage and store it in the database.
    
    Returns:
        Number of news items stored.
        
    Raises:
        DataSourceError: If news fetching fails.
        DatabaseError: If storing news fails.
    """
    try:
        # Always try to fetch real data first
        try:
            alpha_news = fetch_alpha_vantage_news()
            logger.info(f"Fetched {len(alpha_news)} news items from Alpha Vantage")
        except DataSourceError as e:
            # If fetching fails, use placeholder data
            logger.warning(f"Failed to fetch news from Alpha Vantage: {str(e)}. Using placeholder data.")
            alpha_news = get_placeholder_news()
            
        return store_news_in_database(alpha_news)
    except Exception as e:
        logger.error(f"Failed to fetch and store news: {str(e)}")
        raise

def get_latest_news(limit: int = 10, currency_pair: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get latest news from the database.
    
    Args:
        limit: Maximum number of news items to return.
        currency_pair: Filter news by currency pair.
        
    Returns:
        List of news items.
        
    Raises:
        DatabaseError: If fetching news fails.
    """
    try:
        client = get_supabase_db_client()
        
        # Build the query
        if currency_pair:
            # Filter by currency pair
            # Note: This is a bit complex since we need to check if the currency pair is in the array
            # Using PostgreSQL's array contains operator (@>) via a RPC function for this query would be better
            # This is a simplistic approach that works but isn't optimal
            news_items = client.fetch_all(
                "news_data",
                order_by="-published_at",
                limit=limit * 5  # Fetch more than needed to filter
            )
            
            # Filter client-side
            filtered_news = [item for item in news_items if currency_pair in (item.get("currencies") or [])]
            return filtered_news[:limit]
        else:
            # No filtering needed
            return client.fetch_all(
                "news_data",
                order_by="-published_at",
                limit=limit
            )
            
    except Exception as e:
        logger.error(f"Failed to get latest news: {str(e)}")
        raise DatabaseError(f"Failed to get latest news: {str(e)}")

if __name__ == "__main__":
    # When run directly, fetch and store news
    logging.basicConfig(level=logging.INFO)
    
    print("Forex AI News Fetcher (Alpha Vantage)")
    print("===================================")
    
    try:
        count = fetch_and_store_news()
        print(f"Successfully stored {count} news items")
        
        # Show the latest news
        latest = get_latest_news(5)
        print("\nLatest News:")
        for item in latest:
            print(f"- {item['title']} ({item['source']}) - {item['published_at']}")
            
    except Exception as e:
        print(f"Error: {str(e)}") 
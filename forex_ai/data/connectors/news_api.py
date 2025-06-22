"""
News API connector for the Forex AI Trading System.

This module provides functionality to fetch financial news from The News API.
"""

import logging
import requests
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import json

from pydantic import BaseModel

from forex_ai.config.settings import get_settings
from forex_ai.exceptions import DataSourceError, ApiConnectionError, ApiRateLimitError, ApiResponseError
from forex_ai.data.connectors.base import BaseConnector
from forex_ai.data.storage.supabase_client import get_supabase_db_client

logger = logging.getLogger(__name__)

class NewsItem(BaseModel):
    """News article data model."""
    
    title: str
    description: Optional[str] = None
    content: Optional[str] = None
    url: str
    image_url: Optional[str] = None
    source_name: str
    published_at: datetime
    categories: List[str] = []
    sentiment: Optional[float] = None
    relevance_score: Optional[float] = None

class NewsApiConnector(BaseConnector):
    """
    Connector for The News API.
    
    This connector provides methods to fetch financial news from The News API.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the News API connector.
        
        Args:
            api_key: The News API key. If not provided, it will be read from settings.
        """
        settings = get_settings()
        self.api_key = api_key or settings.THE_NEWS_API_KEY
        self.base_url = "https://api.thenewsapi.com/v1"
        self.db_client = get_supabase_db_client()
        
        if not self.api_key or self.api_key == "placeholder":
            logger.critical("The News API key not provided. API calls will fail.")
            raise ValueError("The News API key is required. Please set THE_NEWS_API_KEY in environment variables.")
    
    async def connect(self) -> bool:
        """
        Connect to the data source.
        
        Returns:
            bool: True if connection is successful, False otherwise.
        """
        try:
            # Test connection by fetching a single news item
            await self.get_news(limit=1)
            return True
        except Exception as e:
            logger.error(f"Error connecting to The News API: {str(e)}")
            return False
    
    async def disconnect(self) -> bool:
        """
        Disconnect from the data source.
        
        Returns:
            bool: True if disconnection is successful, False otherwise.
        """
        # No persistent connection to close
        return True
    
    async def is_connected(self) -> bool:
        """
        Check if connected to the data source.
        
        Returns:
            bool: True if connected, False otherwise.
        """
        try:
            # Test connection by fetching a single news item
            await self.get_news(limit=1)
            return True
        except Exception:
            return False
    
    async def get_data(self, **kwargs) -> Any:
        """
        Get data from the data source.
        
        Args:
            **kwargs: Keyword arguments for the data request.
            
        Returns:
            Any: Data from the data source.
        """
        return await self.get_news(**kwargs)
    
    async def get_news(
        self,
        keywords: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        days: int = 1,
        limit: int = 10
    ) -> List[NewsItem]:
        """
        Get news articles.
        
        Args:
            keywords: List of keywords to search for.
            categories: List of categories to filter by.
            days: Number of days to look back.
            limit: Maximum number of results to return.
            
        Returns:
            List of news articles.
            
        Raises:
            DataSourceError: If fetching the news fails.
            ApiConnectionError: If connection to The News API fails.
            ApiRateLimitError: If API rate limit is exceeded.
            ApiResponseError: If response from The News API is invalid.
        """
        try:
            # Calculate date range
            published_after = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            
            # Prepare request parameters
            params = {
                "api_token": self.api_key,
                "language": "en",
                "published_after": published_after,
                "limit": min(limit, 100),  # API limit is 100 per request
                "sort": "published_at"
            }
            
            # Add keywords if provided
            if keywords:
                params["search"] = " OR ".join(keywords)
            
            # Add categories if provided
            if categories:
                params["categories"] = ",".join(categories)
            
            # Make request to The News API
            response = requests.get(f"{self.base_url}/news/all", params=params)
            
            # Check for HTTP errors
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            
            # Check for API errors
            if "error" in data:
                if "rate limit" in data["error"].lower():
                    raise ApiRateLimitError(f"The News API rate limit exceeded: {data['error']}")
                raise ApiResponseError(f"The News API error: {data['error']}")
            
            if "data" not in data:
                raise ApiResponseError("Invalid response format from The News API")
            
            # Extract and format news articles
            articles = []
            for article in data["data"]:
                try:
                    # Convert to NewsItem model
                    news_item = NewsItem(
                        title=article.get("title", ""),
                        description=article.get("description", ""),
                        content=article.get("snippet", ""),
                        url=article.get("url", ""),
                        image_url=article.get("image_url"),
                        source_name=article.get("source", "Unknown"),
                        published_at=datetime.fromisoformat(article.get("published_at").replace("Z", "+00:00")),
                        categories=article.get("categories", []),
                        sentiment=None,  # Sentiment not provided by The News API
                        relevance_score=None  # Relevance score not provided by The News API
                    )
                    articles.append(news_item)
                except Exception as e:
                    logger.warning(f"Error parsing news article: {str(e)}")
                    continue
            
            return articles
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to The News API: {str(e)}")
            raise ApiConnectionError(f"Failed to connect to The News API: {str(e)}")
        except ApiRateLimitError:
            logger.error("The News API rate limit exceeded")
            raise
        except ApiResponseError as e:
            logger.error(f"Invalid response from The News API: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error fetching news from The News API: {str(e)}")
            raise DataSourceError(f"Failed to fetch news: {str(e)}")
    
    def get_top_headlines(
        self,
        category: str = "business",
        country: str = "us",
        query: Optional[str] = None,
        page_size: int = 20,
        page: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Get top headlines.
        
        Args:
            category: News category (e.g., "business", "technology").
            country: Country code (e.g., "us", "gb").
            query: Search query.
            page_size: Number of results per page.
            page: Page number.
            
        Returns:
            List of news articles.
            
        Raises:
            DataSourceError: If fetching the headlines fails.
            ApiConnectionError: If connection to The News API fails.
            ApiRateLimitError: If API rate limit is exceeded.
            ApiResponseError: If response from The News API is invalid.
        """
        try:
            # Prepare request parameters
            params = {
                "api_token": self.api_key,
                "language": "en",
                "limit": min(page_size, 100),  # API limit is 100 per request
                "page": page,
                "categories": category
            }
            
            # Add search query if provided
            if query:
                params["search"] = query
            
            # Make request to The News API
            response = requests.get(f"{self.base_url}/news/top", params=params)
            
            # Check for HTTP errors
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            
            # Check for API errors
            if "error" in data:
                if "rate limit" in data["error"].lower():
                    raise ApiRateLimitError(f"The News API rate limit exceeded: {data['error']}")
                raise ApiResponseError(f"The News API error: {data['error']}")
            
            if "data" not in data:
                raise ApiResponseError("Invalid response format from The News API")
            
            # Extract articles
            articles = data["data"]
            
            return articles
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to The News API: {str(e)}")
            raise ApiConnectionError(f"Failed to connect to The News API: {str(e)}")
        except ApiRateLimitError:
            logger.error("The News API rate limit exceeded")
            raise
        except ApiResponseError as e:
            logger.error(f"Invalid response from The News API: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error fetching headlines from The News API: {str(e)}")
            raise DataSourceError(f"Failed to fetch headlines: {str(e)}")
    
    def get_everything(
        self,
        query: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        language: str = "en",
        sort_by: str = "publishedAt",
        page_size: int = 20,
        page: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Search for news articles.
        
        Args:
            query: Search query.
            from_date: Start date for search.
            to_date: End date for search.
            language: Language code (e.g., "en", "fr").
            sort_by: Sort order (e.g., "publishedAt", "relevancy").
            page_size: Number of results per page.
            page: Page number.
            
        Returns:
            List of news articles.
            
        Raises:
            DataSourceError: If searching for articles fails.
            ApiConnectionError: If connection to The News API fails.
            ApiRateLimitError: If API rate limit is exceeded.
            ApiResponseError: If response from The News API is invalid.
        """
        try:
            # Prepare request parameters
            params = {
                "api_token": self.api_key,
                "language": language,
                "search": query,
                "limit": min(page_size, 100),  # API limit is 100 per request
                "page": page,
                "sort": "published_at" if sort_by == "publishedAt" else "relevance"
            }
            
            # Add date range if provided
            if from_date:
                params["published_after"] = from_date.strftime("%Y-%m-%d")
            
            if to_date:
                params["published_before"] = to_date.strftime("%Y-%m-%d")
            
            # Make request to The News API
            response = requests.get(f"{self.base_url}/news/all", params=params)
            
            # Check for HTTP errors
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            
            # Check for API errors
            if "error" in data:
                if "rate limit" in data["error"].lower():
                    raise ApiRateLimitError(f"The News API rate limit exceeded: {data['error']}")
                raise ApiResponseError(f"The News API error: {data['error']}")
            
            if "data" not in data:
                raise ApiResponseError("Invalid response format from The News API")
            
            # Extract articles
            articles = data["data"]
            
            return articles
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to The News API: {str(e)}")
            raise ApiConnectionError(f"Failed to connect to The News API: {str(e)}")
        except ApiRateLimitError:
            logger.error("The News API rate limit exceeded")
            raise
        except ApiResponseError as e:
            logger.error(f"Invalid response from The News API: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error searching articles from The News API: {str(e)}")
            raise DataSourceError(f"Failed to search articles: {str(e)}")
    
    def get_sources(
        self,
        category: Optional[str] = None,
        language: str = "en",
        country: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get news sources.
        
        Args:
            category: News category (e.g., "business", "technology").
            language: Language code (e.g., "en", "fr").
            country: Country code (e.g., "us", "gb").
            
        Returns:
            List of news sources.
            
        Raises:
            DataSourceError: If fetching the sources fails.
            ApiConnectionError: If connection to The News API fails.
            ApiRateLimitError: If API rate limit is exceeded.
            ApiResponseError: If response from The News API is invalid.
        """
        try:
            # Prepare request parameters
            params = {
                "api_token": self.api_key,
                "language": language,
                "limit": 100  # Maximum allowed by API
            }
            
            # Add category if provided
            if category:
                params["categories"] = category
            
            # Make request to The News API
            response = requests.get(f"{self.base_url}/news/sources", params=params)
            
            # Check for HTTP errors
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            
            # Check for API errors
            if "error" in data:
                if "rate limit" in data["error"].lower():
                    raise ApiRateLimitError(f"The News API rate limit exceeded: {data['error']}")
                raise ApiResponseError(f"The News API error: {data['error']}")
            
            if "data" not in data:
                raise ApiResponseError("Invalid response format from The News API")
            
            # Extract sources
            sources = data["data"]
            
            return sources
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to The News API: {str(e)}")
            raise ApiConnectionError(f"Failed to connect to The News API: {str(e)}")
        except ApiRateLimitError:
            logger.error("The News API rate limit exceeded")
            raise
        except ApiResponseError as e:
            logger.error(f"Invalid response from The News API: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error fetching sources from The News API: {str(e)}")
            raise DataSourceError(f"Failed to fetch sources: {str(e)}")
    
    def save_to_database(self, articles: List[Dict[str, Any]]) -> int:
        """
        Save news articles to the database.
        
        Args:
            articles: List of news articles.
            
        Returns:
            Number of articles saved.
            
        Raises:
            DataSourceError: If saving to the database fails.
        """
        try:
            # Insert articles into the database
            result = self.db_client.insert_many("news_articles", articles)
            
            # Return the number of articles saved
            return len(articles)
        except Exception as e:
            logger.error(f"Error saving articles to database: {str(e)}")
            raise DataSourceError(f"Failed to save articles to database: {str(e)}")
    
    def get_forex_related_news(
        self,
        currency_pair: Optional[str] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Get forex-related news.
        
        Args:
            currency_pair: Currency pair (e.g., "EUR/USD").
            from_date: Start date for search.
            to_date: End date for search.
            limit: Maximum number of results.
            
        Returns:
            List of forex-related news articles.
            
        Raises:
            DataSourceError: If fetching forex news fails.
            NotImplementedError: This method is not yet implemented.
        """
        try:
            # Prepare query conditions
            conditions = {"category": "forex"}
            
            if currency_pair:
                # Use a more flexible search approach with Supabase
                # This assumes there's a text search capability or a keywords column
                conditions["keywords"] = f"%{currency_pair}%"
            
            # Prepare date range conditions if provided
            if from_date or to_date:
                date_conditions = {}
                if from_date:
                    date_conditions["gte"] = from_date.isoformat()
                if to_date:
                    date_conditions["lte"] = to_date.isoformat()
                
                conditions["published_at"] = date_conditions
            
            # Query the database
            articles = self.db_client.fetch_all(
                "news_articles",
                where=conditions,
                order_by="-published_at",
                limit=limit
            )
            
            return articles
        except Exception as e:
            logger.error(f"Error fetching forex news: {str(e)}")
            raise DataSourceError(f"Failed to fetch forex news: {str(e)}") 
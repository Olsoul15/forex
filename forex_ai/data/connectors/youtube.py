"""
YouTube connector for the Forex AI Trading System.

This module provides functionality to fetch and analyze financial content from YouTube.
"""

import logging
import requests
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import isodate

from forex_ai.config.settings import get_settings
from forex_ai.exceptions import DataSourceError, ApiConnectionError, ApiRateLimitError, ApiResponseError
from forex_ai.data.storage.supabase_client import get_supabase_db_client

logger = logging.getLogger(__name__)

class YouTubeConnector:
    """
    Connector for YouTube API.
    
    This connector provides methods to fetch and analyze financial content from YouTube.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the YouTube connector.
        
        Args:
            api_key: YouTube API key. If not provided, it will be read from settings.
        """
        settings = get_settings()
        self.api_key = api_key or settings.YOUTUBE_API_KEY
        self.base_url = "https://www.googleapis.com/youtube/v3"
        self.db_client = get_supabase_db_client()
        
        if not self.api_key or self.api_key == "placeholder":
            logger.critical("YouTube API key not provided. API calls will fail.")
            raise ValueError("YouTube API key is required. Please set YOUTUBE_API_KEY in environment variables.")
    
    def search_videos(
        self,
        query: str,
        max_results: int = 10,
        published_after: Optional[datetime] = None,
        published_before: Optional[datetime] = None,
        order: str = "relevance",
    ) -> List[Dict[str, Any]]:
        """
        Search for YouTube videos.
        
        Args:
            query: Search query.
            max_results: Maximum number of results.
            published_after: Only include videos published after this date.
            published_before: Only include videos published before this date.
            order: Order of results (relevance, date, rating, viewCount).
            
        Returns:
            List of video data.
            
        Raises:
            DataSourceError: If searching for videos fails.
            ApiConnectionError: If connection to YouTube API fails.
            ApiRateLimitError: If API rate limit is exceeded.
            ApiResponseError: If response from YouTube API is invalid.
        """
        try:
            # Prepare request parameters
            params = {
                "part": "snippet",
                "q": query,
                "type": "video",
                "maxResults": min(max_results, 50),  # YouTube API limit is 50 per request
                "order": order,
                "key": self.api_key,
                "relevanceLanguage": "en"
            }
            
            # Add date filters if provided
            if published_after:
                params["publishedAfter"] = published_after.isoformat() + "Z"
            
            if published_before:
                params["publishedBefore"] = published_before.isoformat() + "Z"
            
            # Make request to YouTube API
            response = requests.get(f"{self.base_url}/search", params=params)
            
            # Check for HTTP errors
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            
            # Check for API errors
            if "error" in data:
                if "quota" in data["error"].get("message", "").lower():
                    raise ApiRateLimitError(f"YouTube API quota exceeded: {data['error']['message']}")
                raise ApiResponseError(f"YouTube API error: {data['error']['message']}")
            
            # Extract video IDs
            video_ids = [item["id"]["videoId"] for item in data.get("items", []) if "videoId" in item["id"]]
            
            # If no videos found, return empty list
            if not video_ids:
                return []
            
            # Get detailed video information
            videos = self.get_videos_details(video_ids)
            
            return videos
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to YouTube API: {str(e)}")
            raise ApiConnectionError(f"Failed to connect to YouTube API: {str(e)}")
        except ApiRateLimitError:
            logger.error("YouTube API quota exceeded")
            raise
        except ApiResponseError as e:
            logger.error(f"Invalid response from YouTube API: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error searching YouTube videos: {str(e)}")
            raise DataSourceError(f"Failed to search YouTube videos: {str(e)}")
    
    def get_videos_details(self, video_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get details for multiple YouTube videos.
        
        Args:
            video_ids: List of YouTube video IDs.
            
        Returns:
            List of video details.
            
        Raises:
            DataSourceError: If fetching video details fails.
            ApiConnectionError: If connection to YouTube API fails.
            ApiRateLimitError: If API rate limit is exceeded.
            ApiResponseError: If response from YouTube API is invalid.
        """
        try:
            # YouTube API has a limit of 50 video IDs per request
            max_ids_per_request = 50
            all_videos = []
            
            # Process video IDs in batches
            for i in range(0, len(video_ids), max_ids_per_request):
                batch_ids = video_ids[i:i + max_ids_per_request]
                
                # Prepare request parameters
                params = {
                    "part": "snippet,contentDetails,statistics",
                    "id": ",".join(batch_ids),
                    "key": self.api_key
                }
                
                # Make request to YouTube API
                response = requests.get(f"{self.base_url}/videos", params=params)
                
                # Check for HTTP errors
                response.raise_for_status()
                
                # Parse response
                data = response.json()
                
                # Check for API errors
                if "error" in data:
                    if "quota" in data["error"].get("message", "").lower():
                        raise ApiRateLimitError(f"YouTube API quota exceeded: {data['error']['message']}")
                    raise ApiResponseError(f"YouTube API error: {data['error']['message']}")
                
                # Process video details
                for item in data.get("items", []):
                    video = {
                        "id": item["id"],
                        "title": item["snippet"]["title"],
                        "description": item["snippet"]["description"],
                        "published_at": item["snippet"]["publishedAt"],
                        "channel_id": item["snippet"]["channelId"],
                        "channel_title": item["snippet"]["channelTitle"],
                        "thumbnails": item["snippet"]["thumbnails"],
                        "tags": item["snippet"].get("tags", []),
                        "category_id": item["snippet"].get("categoryId"),
                        "duration": isodate.parse_duration(item["contentDetails"]["duration"]).total_seconds(),
                        "view_count": int(item["statistics"].get("viewCount", 0)),
                        "like_count": int(item["statistics"].get("likeCount", 0)),
                        "comment_count": int(item["statistics"].get("commentCount", 0)),
                        "url": f"https://www.youtube.com/watch?v={item['id']}"
                    }
                    all_videos.append(video)
            
            return all_videos
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to YouTube API: {str(e)}")
            raise ApiConnectionError(f"Failed to connect to YouTube API: {str(e)}")
        except ApiRateLimitError:
            logger.error("YouTube API quota exceeded")
            raise
        except ApiResponseError as e:
            logger.error(f"Invalid response from YouTube API: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error fetching YouTube video details: {str(e)}")
            raise DataSourceError(f"Failed to fetch YouTube video details: {str(e)}")
    
    def get_video_details(self, video_id: str) -> Dict[str, Any]:
        """
        Get details for a YouTube video.
        
        Args:
            video_id: YouTube video ID.
            
        Returns:
            Video details.
            
        Raises:
            DataSourceError: If fetching video details fails.
            ApiConnectionError: If connection to YouTube API fails.
            ApiRateLimitError: If API rate limit is exceeded.
            ApiResponseError: If response from YouTube API is invalid.
        """
        videos = self.get_videos_details([video_id])
        if not videos:
            raise DataSourceError(f"Video with ID {video_id} not found")
        return videos[0]
    
    def get_video_transcript(self, video_id: str) -> str:
        """
        Get transcript for a YouTube video.
        
        Args:
            video_id: YouTube video ID.
            
        Returns:
            Video transcript.
            
        Raises:
            DataSourceError: If fetching transcript fails.
            ApiConnectionError: If connection to YouTube API fails.
            ApiRateLimitError: If API rate limit is exceeded.
            ApiResponseError: If response from YouTube API is invalid.
        """
        try:
            # Note: The YouTube Data API doesn't directly provide transcripts
            # This requires the YouTube Transcript API which is a third-party library
            # For now, we'll raise a more informative error
            raise NotImplementedError(
                "YouTube transcript fetching requires the YouTube Transcript API. "
                "Please install the 'youtube_transcript_api' package and update this method."
            )
            
        except Exception as e:
            logger.error(f"Error fetching YouTube transcript: {str(e)}")
            raise DataSourceError(f"Failed to fetch YouTube transcript: {str(e)}")
    
    def get_channel_videos(
        self,
        channel_id: str,
        max_results: int = 50,
        published_after: Optional[datetime] = None,
        order: str = "date"
    ) -> List[Dict[str, Any]]:
        """
        Get videos from a YouTube channel.
        
        Args:
            channel_id: YouTube channel ID.
            max_results: Maximum number of results.
            published_after: Only include videos published after this date.
            order: Order of results (date, rating, viewCount, title).
            
        Returns:
            List of video data.
            
        Raises:
            DataSourceError: If fetching channel videos fails.
            ApiConnectionError: If connection to YouTube API fails.
            ApiRateLimitError: If API rate limit is exceeded.
            ApiResponseError: If response from YouTube API is invalid.
        """
        try:
            # Prepare request parameters
            params = {
                "part": "snippet",
                "channelId": channel_id,
                "type": "video",
                "maxResults": min(max_results, 50),  # YouTube API limit is 50 per request
                "order": order,
                "key": self.api_key
            }
            
            # Add date filter if provided
            if published_after:
                params["publishedAfter"] = published_after.isoformat() + "Z"
            
            # Make request to YouTube API
            response = requests.get(f"{self.base_url}/search", params=params)
            
            # Check for HTTP errors
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            
            # Check for API errors
            if "error" in data:
                if "quota" in data["error"].get("message", "").lower():
                    raise ApiRateLimitError(f"YouTube API quota exceeded: {data['error']['message']}")
                raise ApiResponseError(f"YouTube API error: {data['error']['message']}")
            
            # Extract video IDs
            video_ids = [item["id"]["videoId"] for item in data.get("items", []) if "videoId" in item["id"]]
            
            # If no videos found, return empty list
            if not video_ids:
                return []
            
            # Get detailed video information
            videos = self.get_videos_details(video_ids)
            
            return videos
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to YouTube API: {str(e)}")
            raise ApiConnectionError(f"Failed to connect to YouTube API: {str(e)}")
        except ApiRateLimitError:
            logger.error("YouTube API quota exceeded")
            raise
        except ApiResponseError as e:
            logger.error(f"Invalid response from YouTube API: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error fetching YouTube channel videos: {str(e)}")
            raise DataSourceError(f"Failed to fetch YouTube channel videos: {str(e)}")
    
    def search_channels(
        self,
        query: str,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for YouTube channels.
        
        Args:
            query: Search query.
            max_results: Maximum number of results.
            
        Returns:
            List of channel data.
            
        Raises:
            DataSourceError: If searching for channels fails.
            ApiConnectionError: If connection to YouTube API fails.
            ApiRateLimitError: If API rate limit is exceeded.
            ApiResponseError: If response from YouTube API is invalid.
        """
        try:
            # Prepare request parameters
            params = {
                "part": "snippet",
                "q": query,
                "type": "channel",
                "maxResults": min(max_results, 50),  # YouTube API limit is 50 per request
                "key": self.api_key,
                "relevanceLanguage": "en"
            }
            
            # Make request to YouTube API
            response = requests.get(f"{self.base_url}/search", params=params)
            
            # Check for HTTP errors
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            
            # Check for API errors
            if "error" in data:
                if "quota" in data["error"].get("message", "").lower():
                    raise ApiRateLimitError(f"YouTube API quota exceeded: {data['error']['message']}")
                raise ApiResponseError(f"YouTube API error: {data['error']['message']}")
            
            # Process channel data
            channels = []
            for item in data.get("items", []):
                channel = {
                    "id": item["id"]["channelId"],
                    "title": item["snippet"]["title"],
                    "description": item["snippet"]["description"],
                    "published_at": item["snippet"]["publishedAt"],
                    "thumbnails": item["snippet"]["thumbnails"],
                    "url": f"https://www.youtube.com/channel/{item['id']['channelId']}"
                }
                channels.append(channel)
            
            return channels
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to YouTube API: {str(e)}")
            raise ApiConnectionError(f"Failed to connect to YouTube API: {str(e)}")
        except ApiRateLimitError:
            logger.error("YouTube API quota exceeded")
            raise
        except ApiResponseError as e:
            logger.error(f"Invalid response from YouTube API: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error searching YouTube channels: {str(e)}")
            raise DataSourceError(f"Failed to search YouTube channels: {str(e)}")
    
    def save_to_database(self, videos: List[Dict[str, Any]]) -> int:
        """
        Save videos to the database.
        
        Args:
            videos: List of video data.
            
        Returns:
            Number of videos saved.
            
        Raises:
            DataSourceError: If saving to the database fails.
        """
        try:
            # Insert videos into the database
            result = self.db_client.insert_many("youtube_videos", videos)
            
            # Return the number of videos saved
            return len(videos)
        except Exception as e:
            logger.error(f"Error saving videos to database: {str(e)}")
            raise DataSourceError(f"Failed to save videos to database: {str(e)}")
    
    def get_forex_related_videos(
        self,
        currency_pair: Optional[str] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get forex-related YouTube videos.
        
        Args:
            currency_pair: Currency pair (e.g., "EUR/USD").
            from_date: Only include videos published after this date.
            to_date: Only include videos published before this date.
            limit: Maximum number of results.
            
        Returns:
            List of forex-related videos.
            
        Raises:
            DataSourceError: If fetching videos fails.
        """
        try:
            # Prepare query conditions
            conditions = {"category": "forex"}
            
            if currency_pair:
                # Use a more flexible search approach with Supabase
                conditions["tags"] = f"%{currency_pair}%"
            
            # Prepare date range conditions if provided
            if from_date or to_date:
                date_conditions = {}
                if from_date:
                    date_conditions["gte"] = from_date.isoformat()
                if to_date:
                    date_conditions["lte"] = to_date.isoformat()
                
                conditions["published_at"] = date_conditions
            
            # Query the database
            videos = self.db_client.fetch_all(
                "youtube_videos",
                where=conditions,
                order_by="-published_at",
                limit=limit
            )
            
            return videos
        except Exception as e:
            logger.error(f"Error fetching forex-related videos: {str(e)}")
            raise DataSourceError(f"Failed to fetch forex-related videos: {str(e)}") 
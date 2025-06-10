"""
YouTube connector for the Forex AI Trading System.

This module provides functionality to extract and analyze
financial content from YouTube videos.
"""

import os
import logging
import re
import json
import tempfile
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

import whisper
from pytubefix import YouTube
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from forex_ai.config.settings import get_settings
from forex_ai.exceptions import DataSourceError, ApiConnectionError
from forex_ai.data.storage.postgres_client import get_postgres_client

logger = logging.getLogger(__name__)

class YouTubeConnector:
    """
    Connector for YouTube.
    
    This connector provides methods to extract and analyze
    financial content from YouTube videos. It downloads audio,
    transcribes it using Whisper, and analyzes the content
    using AI models.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_size: str = "base"):
        """
        Initialize the YouTube connector.
        
        Args:
            api_key: YouTube API key. If not provided, it will be read from settings.
            model_size: Whisper model size (tiny, base, small, medium, large).
        """
        settings = get_settings()
        self.api_key = api_key or settings.YOUTUBE_API_KEY
        self.model_size = model_size
        self.postgres_client = get_postgres_client()
        self._whisper_model = None
        
        if not self.api_key:
            logger.warning("YouTube API key not provided. Some functionality may be limited.")
    
    def _get_whisper_model(self):
        """
        Load the Whisper model.
        
        Returns:
            The Whisper model.
        """
        if self._whisper_model is None:
            logger.info(f"Loading Whisper model ({self.model_size})...")
            self._whisper_model = whisper.load_model(self.model_size)
            logger.info("Whisper model loaded successfully.")
            
        return self._whisper_model
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def download_audio(self, video_url: str, output_dir: Optional[str] = None) -> str:
        """
        Download the audio from a YouTube video.
        
        Args:
            video_url: URL of the YouTube video.
            output_dir: Output directory. If None, a temporary directory will be used.
            
        Returns:
            Path to the downloaded audio file.
            
        Raises:
            DataSourceError: If downloading the audio fails.
        """
        try:
            logger.info(f"Downloading audio from {video_url}")
            
            # Create YouTube object
            yt = YouTube(video_url)
            
            # Get video title and ID
            video_title = yt.title
            video_id = yt.video_id
            logger.info(f"Video title: {video_title}, ID: {video_id}")
            
            # Extract audio stream
            audio_stream = yt.streams.filter(only_audio=True).first()
            
            # Determine output directory
            if output_dir is None:
                output_dir = tempfile.gettempdir()
                
            # Download the file
            out_file = audio_stream.download(output_path=output_dir)
            
            # Rename to MP3
            base, ext = os.path.splitext(out_file)
            mp3_file = base + '.mp3'
            os.rename(out_file, mp3_file)
            
            logger.info(f"Downloaded audio to {mp3_file}")
            return mp3_file
            
        except Exception as e:
            error_msg = f"Error downloading audio from {video_url}: {str(e)}"
            logger.error(error_msg)
            raise DataSourceError(error_msg) from e
    
    def transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribe an audio file using Whisper.
        
        Args:
            audio_path: Path to the audio file.
            
        Returns:
            The transcription text.
            
        Raises:
            DataSourceError: If transcription fails.
        """
        try:
            logger.info(f"Transcribing audio file: {audio_path}")
            
            # Load Whisper model if not already loaded
            model = self._get_whisper_model()
            
            # Transcribe audio
            result = model.transcribe(audio_path)
            transcript = result["text"]
            
            logger.info(f"Transcription complete ({len(transcript)} characters)")
            return transcript
            
        except Exception as e:
            error_msg = f"Error transcribing audio: {str(e)}"
            logger.error(error_msg)
            raise DataSourceError(error_msg) from e
    
    def summarize_transcript(self, transcript: str, api_type: str = "azure_openai") -> str:
        """
        Summarize a transcript using an AI model.
        
        Args:
            transcript: The transcript to summarize.
            api_type: The AI API to use (azure_openai, openai, anthropic).
            
        Returns:
            The summary text.
            
        Raises:
            DataSourceError: If summarization fails.
        """
        try:
            logger.info(f"Summarizing transcript ({len(transcript)} chars) using {api_type}")
            
            settings = get_settings()
            
            if api_type == "azure_openai":
                from azure.ai.openai import OpenAI
                
                # Initialize Azure OpenAI client
                client = OpenAI(
                    api_key=settings.AZURE_OPENAI_KEY,
                    api_version=settings.AZURE_OPENAI_API_VERSION,
                    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT
                )
                
                # System message for summarization
                system_message = """
                You are a helpful assistant that summarises YouTube transcripts about finance and forex trading.

                Think step-by-step:

                1. Read the transcript carefully, focusing on key financial information.
                2. Identify the main forex-related topics, strategies, or market analysis presented.
                3. Extract specific currency pairs, trading indicators, or economic events mentioned.
                4. Note any trade setups, entry/exit points, or risk management advice.
                5. Summarize the core message in clear, concise language.
                6. Organize the information logically, maintaining the original meaning.
                7. Include any important quotes or specific numerical data.
                8. Keep the length reasonable (about 15-20% of the original).

                Provide a clear, informative summary that would be valuable for a forex trader.
                """
                
                # Create messages
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": transcript}
                ]
                
                # Get response
                response = client.chat.completions.create(
                    model=settings.REASONING_MODEL,
                    messages=messages,
                    temperature=0.0
                )
                
                # Extract summary
                summary = response.choices[0].message.content
                
            elif api_type == "openai":
                from openai import OpenAI
                
                # Initialize OpenAI client
                client = OpenAI(api_key=settings.OPENAI_API_KEY)
                
                # System message for summarization
                system_message = """
                You are a helpful assistant that summarises YouTube transcripts about finance and forex trading.

                Think step-by-step:

                1. Read the transcript carefully, focusing on key financial information.
                2. Identify the main forex-related topics, strategies, or market analysis presented.
                3. Extract specific currency pairs, trading indicators, or economic events mentioned.
                4. Note any trade setups, entry/exit points, or risk management advice.
                5. Summarize the core message in clear, concise language.
                6. Organize the information logically, maintaining the original meaning.
                7. Include any important quotes or specific numerical data.
                8. Keep the length reasonable (about 15-20% of the original).

                Provide a clear, informative summary that would be valuable for a forex trader.
                """
                
                # Create messages
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": transcript}
                ]
                
                # Get response
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=messages,
                    temperature=0.0
                )
                
                # Extract summary
                summary = response.choices[0].message.content
                
            elif api_type == "anthropic":
                from anthropic import Anthropic
                
                # Initialize Anthropic client
                client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
                
                # System message for summarization
                system_message = """
                You are a helpful assistant that summarises YouTube transcripts about finance and forex trading.

                Think step-by-step:

                1. Read the transcript carefully, focusing on key financial information.
                2. Identify the main forex-related topics, strategies, or market analysis presented.
                3. Extract specific currency pairs, trading indicators, or economic events mentioned.
                4. Note any trade setups, entry/exit points, or risk management advice.
                5. Summarize the core message in clear, concise language.
                6. Organize the information logically, maintaining the original meaning.
                7. Include any important quotes or specific numerical data.
                8. Keep the length reasonable (about 15-20% of the original).

                Provide a clear, informative summary that would be valuable for a forex trader.
                """
                
                # Create message
                message = f"{system_message}\n\nTranscript:\n{transcript}\n\nSummary:"
                
                # Get response
                response = client.messages.create(
                    model=settings.REASONING_MODEL,
                    max_tokens=4000,
                    messages=[{"role": "user", "content": message}]
                )
                
                # Extract summary
                summary = response.content[0].text
                
            else:
                raise ValueError(f"Unsupported API type: {api_type}")
                
            logger.info(f"Summary generated ({len(summary)} characters)")
            return summary
            
        except Exception as e:
            error_msg = f"Error summarizing transcript: {str(e)}"
            logger.error(error_msg)
            raise DataSourceError(error_msg) from e
    
    def extract_currency_pairs(self, text: str) -> List[str]:
        """
        Extract currency pairs from text.
        
        Args:
            text: The text to extract from.
            
        Returns:
            A list of currency pairs.
        """
        # Pattern for currency pairs (e.g., EUR/USD, GBP/JPY)
        pattern = r'\b[A-Z]{3}/[A-Z]{3}\b'
        pairs = re.findall(pattern, text)
        return list(set(pairs))  # Remove duplicates
    
    def extract_sentiment(self, summary: str, api_type: str = "azure_openai") -> Dict[str, Any]:
        """
        Extract sentiment and key information from a summary.
        
        Args:
            summary: The summary to analyze.
            api_type: The AI API to use (azure_openai, openai, anthropic).
            
        Returns:
            A dictionary with sentiment analysis and extracted information.
            
        Raises:
            DataSourceError: If sentiment extraction fails.
        """
        try:
            logger.info(f"Extracting sentiment from summary using {api_type}")
            
            settings = get_settings()
            
            if api_type == "azure_openai":
                from azure.ai.openai import OpenAI
                
                # Initialize Azure OpenAI client
                client = OpenAI(
                    api_key=settings.AZURE_OPENAI_KEY,
                    api_version=settings.AZURE_OPENAI_API_VERSION,
                    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT
                )
                
                # System message for sentiment analysis
                system_message = """
                You are a financial sentiment analyzer specializing in forex trading content.
                
                Analyze the provided summary of a YouTube video about forex trading and extract the following information in JSON format:
                
                1. Overall sentiment (numerical score from -1.0 to 1.0 where -1.0 is extremely bearish, 0 is neutral, and 1.0 is extremely bullish)
                2. Currency pairs mentioned (list)
                3. Key topics discussed (list)
                4. Trade setups or strategies mentioned (list)
                5. Economic events or indicators mentioned (list)
                6. Market outlook (short text description)
                
                Return only valid JSON without any additional text or explanations.
                """
                
                # Create messages
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": summary}
                ]
                
                # Get response
                response = client.chat.completions.create(
                    model=settings.REASONING_MODEL,
                    messages=messages,
                    temperature=0.0
                )
                
                # Parse JSON response
                content = response.choices[0].message.content
                sentiment_data = json.loads(content)
                
            elif api_type == "openai":
                from openai import OpenAI
                
                # Initialize OpenAI client
                client = OpenAI(api_key=settings.OPENAI_API_KEY)
                
                # System message for sentiment analysis
                system_message = """
                You are a financial sentiment analyzer specializing in forex trading content.
                
                Analyze the provided summary of a YouTube video about forex trading and extract the following information in JSON format:
                
                1. Overall sentiment (numerical score from -1.0 to 1.0 where -1.0 is extremely bearish, 0 is neutral, and 1.0 is extremely bullish)
                2. Currency pairs mentioned (list)
                3. Key topics discussed (list)
                4. Trade setups or strategies mentioned (list)
                5. Economic events or indicators mentioned (list)
                6. Market outlook (short text description)
                
                Return only valid JSON without any additional text or explanations.
                """
                
                # Create messages
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": summary}
                ]
                
                # Get response
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=messages,
                    temperature=0.0
                )
                
                # Parse JSON response
                content = response.choices[0].message.content
                sentiment_data = json.loads(content)
                
            elif api_type == "anthropic":
                from anthropic import Anthropic
                
                # Initialize Anthropic client
                client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
                
                # System message for sentiment analysis
                system_message = """
                You are a financial sentiment analyzer specializing in forex trading content.
                
                Analyze the provided summary of a YouTube video about forex trading and extract the following information in JSON format:
                
                1. Overall sentiment (numerical score from -1.0 to 1.0 where -1.0 is extremely bearish, 0 is neutral, and 1.0 is extremely bullish)
                2. Currency pairs mentioned (list)
                3. Key topics discussed (list)
                4. Trade setups or strategies mentioned (list)
                5. Economic events or indicators mentioned (list)
                6. Market outlook (short text description)
                
                Return only valid JSON without any additional text or explanations.
                """
                
                # Create message
                message = f"{system_message}\n\nSummary:\n{summary}"
                
                # Get response
                response = client.messages.create(
                    model=settings.REASONING_MODEL,
                    max_tokens=2000,
                    messages=[{"role": "user", "content": message}]
                )
                
                # Parse JSON response
                content = response.content[0].text
                sentiment_data = json.loads(content)
                
            else:
                raise ValueError(f"Unsupported API type: {api_type}")
                
            logger.info(f"Sentiment extracted: {sentiment_data['overall_sentiment']}")
            return sentiment_data
            
        except Exception as e:
            error_msg = f"Error extracting sentiment: {str(e)}"
            logger.error(error_msg)
            raise DataSourceError(error_msg) from e
    
    def get_video_metadata(self, video_url: str) -> Dict[str, Any]:
        """
        Get metadata for a YouTube video.
        
        Args:
            video_url: URL of the YouTube video.
            
        Returns:
            Video metadata.
            
        Raises:
            DataSourceError: If getting metadata fails.
        """
        try:
            logger.info(f"Getting metadata for {video_url}")
            
            # Create YouTube object
            yt = YouTube(video_url)
            
            # Extract metadata
            metadata = {
                "video_id": yt.video_id,
                "title": yt.title,
                "description": yt.description,
                "channel": yt.author,
                "published_at": datetime.fromtimestamp(yt.publish_date.timestamp()) if yt.publish_date else None,
                "length": yt.length,
                "views": yt.views,
                "keywords": yt.keywords,
                "thumbnail_url": yt.thumbnail_url,
            }
            
            logger.info(f"Metadata retrieved for {yt.title}")
            return metadata
            
        except Exception as e:
            error_msg = f"Error getting video metadata: {str(e)}"
            logger.error(error_msg)
            raise DataSourceError(error_msg) from e
    
    def process_video(
        self, 
        video_url: str,
        api_type: str = "azure_openai",
        save_to_db: bool = True,
        cleanup_files: bool = True
    ) -> Dict[str, Any]:
        """
        Process a YouTube video: download, transcribe, summarize, and analyze.
        
        Args:
            video_url: URL of the YouTube video.
            api_type: The AI API to use (azure_openai, openai, anthropic).
            save_to_db: Whether to save the results to the database.
            cleanup_files: Whether to delete temporary files after processing.
            
        Returns:
            A dictionary with the processing results.
            
        Raises:
            DataSourceError: If processing fails.
        """
        try:
            logger.info(f"Processing YouTube video: {video_url}")
            
            # Get video metadata
            metadata = self.get_video_metadata(video_url)
            
            # Check if video already exists in database
            if save_to_db:
                existing_video = self.postgres_client.find_one(
                    "youtube_videos",
                    {"video_id": metadata["video_id"]}
                )
                
                if existing_video:
                    logger.info(f"Video already exists in database: {metadata['video_id']}")
                    return {
                        "status": "already_exists",
                        "video_id": metadata["video_id"],
                        "title": metadata["title"],
                        "channel": metadata["channel"]
                    }
            
            # Download audio
            temp_dir = tempfile.mkdtemp()
            audio_path = self.download_audio(video_url, temp_dir)
            
            # Transcribe audio
            transcript = self.transcribe_audio(audio_path)
            
            # Summarize transcript
            summary = self.summarize_transcript(transcript, api_type)
            
            # Extract sentiment and information
            sentiment_data = self.extract_sentiment(summary, api_type)
            
            # Extract currency pairs
            currency_pairs = self.extract_currency_pairs(summary)
            if not currency_pairs and "currency_pairs_mentioned" in sentiment_data:
                currency_pairs = sentiment_data["currency_pairs_mentioned"]
            
            # Prepare result
            result = {
                "video_id": metadata["video_id"],
                "title": metadata["title"],
                "channel": metadata["channel"],
                "published_at": metadata["published_at"],
                "transcript": transcript,
                "summary": summary,
                "sentiment": sentiment_data.get("overall_sentiment", 0),
                "currency_pairs": currency_pairs,
                "metadata": metadata,
                "processed_at": datetime.now()
            }
            
            # Save to database
            if save_to_db:
                # Prepare data for insertion
                db_record = {
                    "video_id": metadata["video_id"],
                    "title": metadata["title"],
                    "channel": metadata["channel"],
                    "published_at": metadata["published_at"],
                    "transcript": transcript,
                    "summary": summary,
                    "sentiment": sentiment_data.get("overall_sentiment", 0),
                    "currencies": currency_pairs,
                    "tags": sentiment_data.get("key_topics_discussed", []),
                    "metadata": metadata,
                    "processed_at": datetime.now()
                }
                
                # Insert into database
                self.postgres_client.insert_one("youtube_videos", db_record)
                logger.info(f"Saved video to database: {metadata['video_id']}")
            
            # Clean up temporary files
            if cleanup_files:
                try:
                    os.remove(audio_path)
                    os.rmdir(temp_dir)
                    logger.info(f"Cleaned up temporary files: {audio_path}")
                except Exception as e:
                    logger.warning(f"Error cleaning up temporary files: {str(e)}")
            
            return result
            
        except Exception as e:
            error_msg = f"Error processing YouTube video: {str(e)}"
            logger.error(error_msg)
            raise DataSourceError(error_msg) from e
    
    def batch_process_videos(self, video_urls: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        Process multiple YouTube videos.
        
        Args:
            video_urls: List of YouTube video URLs.
            **kwargs: Additional arguments to pass to process_video.
            
        Returns:
            A list of dictionaries with the processing results.
        """
        results = []
        
        for url in video_urls:
            try:
                result = self.process_video(url, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing video {url}: {str(e)}")
                results.append({
                    "status": "error",
                    "url": url,
                    "error": str(e)
                })
        
        return results
    
    def search_videos_by_sentiment(
        self,
        currency_pair: Optional[str] = None,
        min_sentiment: float = -1.0,
        max_sentiment: float = 1.0,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for videos by sentiment.
        
        Args:
            currency_pair: Filter by currency pair.
            min_sentiment: Minimum sentiment score.
            max_sentiment: Maximum sentiment score.
            limit: Maximum number of results.
            
        Returns:
            A list of videos matching the criteria.
        """
        try:
            # Build query
            query = """
                SELECT video_id, title, channel, published_at, 
                       summary, sentiment, currencies, processed_at
                FROM youtube_videos
                WHERE sentiment BETWEEN %s AND %s
            """
            params = [min_sentiment, max_sentiment]
            
            if currency_pair:
                query += " AND %s = ANY(currencies)"
                params.append(currency_pair)
                
            query += " ORDER BY published_at DESC LIMIT %s"
            params.append(limit)
            
            # Execute query
            results = self.postgres_client.execute(query, tuple(params))
            
            return results or []
            
        except Exception as e:
            error_msg = f"Error searching videos by sentiment: {str(e)}"
            logger.error(error_msg)
            raise DataSourceError(error_msg) from e 
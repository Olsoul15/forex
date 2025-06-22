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
from typing import Dict, List, Optional, Any
import whisper
from pytubefix import YouTube
from tenacity import retry, stop_after_attempt, wait_exponential

from forex_ai.config.settings import get_settings
from forex_ai.exceptions import DataSourceError
from forex_ai.models.llm_controller import LLMController
from forex_ai.models.mcp import get_mcp_agent

logger = logging.getLogger(__name__)

class YouTubeConnector:
    """
    Connector for YouTube.

    This connector provides methods to extract and analyze
    financial content from YouTube videos. It downloads audio,
    transcribes it using Whisper, and analyzes the content
    using the MCP agent.
    """

    def __init__(self, model_size: str = "base"):
        """
        Initialize the YouTube connector.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large).
        """
        self.settings = get_settings()
        self.model_size = model_size
        self._whisper_model = None
        self.llm_controller = LLMController()
        self.mcp_agent = get_mcp_agent()

    def _get_whisper_model(self):
        """Load the Whisper model."""
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
        """Download the audio from a YouTube video."""
        try:
            logger.info(f"Downloading audio from {video_url}")
            yt = YouTube(video_url)
            audio_stream = yt.streams.filter(only_audio=True).first()
            output_dir = output_dir or tempfile.gettempdir()
            out_file = audio_stream.download(output_path=output_dir)
            base, _ = os.path.splitext(out_file)
            mp3_file = base + '.mp3'
            os.rename(out_file, mp3_file)
            logger.info(f"Downloaded audio to {mp3_file}")
            return mp3_file
        except Exception as e:
            error_msg = f"Error downloading audio from {video_url}: {e}"
            logger.error(error_msg)
            raise DataSourceError(error_msg) from e

    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe an audio file using Whisper."""
        try:
            logger.info(f"Transcribing audio file: {audio_path}")
            model = self._get_whisper_model()
            result = model.transcribe(audio_path)
            transcript = result.get("text", "")
            logger.info(f"Transcription complete ({len(transcript)} characters)")
            return transcript
        except Exception as e:
            error_msg = f"Error transcribing audio: {e}"
            logger.error(error_msg)
            raise DataSourceError(error_msg) from e

    async def summarize_transcript(self, transcript: str) -> str:
        """Summarizes the given transcript using the MCP agent."""
        try:
            logger.info("Summarizing transcript using MCP agent")
            
            from forex_ai.models.mcp import Message, MessageRole
            
            messages = [
                Message(
                    role=MessageRole.SYSTEM,
                    content="You are a financial analyst specializing in forex trading. Summarize the following transcript from a financial news video. Focus on the key market insights, sentiment, and any mention of currency pairs or economic events. Keep the summary concise and to the point."
                ),
                Message(
                    role=MessageRole.USER,
                    content=f"Transcript:\n{transcript}"
                )
            ]
            
            response = await self.mcp_agent.ask(messages=messages)
            
            if not response.success:
                logger.error(f"MCP agent failed to summarize transcript: {response.error_message}")
                return f"Error: Could not summarize transcript. Details: {response.error_message}"
                
            return response.content or "No summary generated."
        except Exception as e:
            logger.error(f"Failed to summarize transcript: {e}")
            return f"Error: Could not summarize transcript. Details: {e}"

    def extract_currency_pairs(self, text: str) -> List[str]:
        """Extract currency pairs from text."""
        pattern = r'\b[A-Z]{3}/[A-Z]{3}\b'
        pairs = re.findall(pattern, text)
        return list(set(pairs))

    async def extract_sentiment(self, summary: str) -> Dict[str, Any]:
        """Extract sentiment and key information from a summary using the MCP agent."""
        try:
            logger.info("Extracting sentiment from summary using MCP agent")
            
            from forex_ai.models.mcp import Message, MessageRole
            
            messages = [
                Message(
                    role=MessageRole.SYSTEM,
                    content="""
                    You are a financial sentiment analyzer specializing in forex trading content.
                    Analyze the provided summary of a YouTube video and return a JSON object with:
                    1. 'sentiment_score': a float from -1.0 (bearish) to 1.0 (bullish).
                    2. 'currency_pairs': a list of any currency pairs mentioned.
                    3. 'key_topics': a list of key topics discussed.
                    4. 'trade_setups': a list of trade setups or strategies mentioned.
                    5. 'economic_events': a list of economic events or indicators.
                    6. 'market_outlook': a short text description of the market outlook.
                    Return only valid JSON.
                    """
                ),
                Message(
                    role=MessageRole.USER,
                    content=f"Summary:\n{summary}"
                )
            ]
            
            response = await self.mcp_agent.ask(messages=messages)
            
            if not response.success:
                logger.error(f"MCP agent failed to extract sentiment: {response.error_message}")
                return {"error": f"Failed to extract sentiment: {response.error_message}"}
            
            # Clean the response to ensure it's valid JSON
            content = response.content or "{}"
            clean_response = content.strip().replace("```json", "").replace("```", "")
            
            try:
                return json.loads(clean_response)
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON response: {e}")
                return {"error": f"Failed to parse sentiment data: {e}"}
                
        except Exception as e:
            logger.error(f"Error extracting sentiment: {e}")
            return {"error": f"Failed to extract sentiment: {e}"}

    def get_video_metadata(self, video_url: str) -> Dict[str, Any]:
        """Get metadata for a YouTube video."""
        try:
            yt = YouTube(video_url)
            return {
                "video_id": yt.video_id,
                "title": yt.title,
                "author": yt.author,
                "length": yt.length,
                "publish_date": yt.publish_date,
                "views": yt.views,
                "rating": yt.rating,
            }
        except Exception as e:
            logger.error(f"Error getting video metadata for {video_url}: {e}")
            return {"error": str(e)}

    async def process_video(self, video_url: str, cleanup_files: bool = True) -> Dict[str, Any]:
        """
        Fully processes a YouTube video: download, transcribe, summarize, and analyze.
        """
        audio_path = None
        try:
            metadata = self.get_video_metadata(video_url)
            if "error" in metadata:
                return metadata

            audio_path = self.download_audio(video_url)
            transcript = self.transcribe_audio(audio_path)
            
            if not transcript:
                return {"error": "Transcription failed or produced no content.", **metadata}
                
            summary = await self.summarize_transcript(transcript)
            if summary.startswith("Error:"):
                return {"error": "Summarization failed.", **metadata}

            sentiment_data = await self.extract_sentiment(summary)
            if "error" in sentiment_data:
                 return {"error": "Sentiment analysis failed.", **metadata}

            result = {
                **metadata,
                "transcript_length": len(transcript),
                "summary": summary,
                "sentiment_score": sentiment_data.get("sentiment_score"),
                "currency_pairs": sentiment_data.get("currency_pairs"),
                "key_topics": sentiment_data.get("key_topics"),
            }
            return result
        except Exception as e:
            logger.error(f"An error occurred during video processing for {video_url}: {e}")
            return {"error": f"An unexpected error occurred: {e}"}
        finally:
            if cleanup_files and audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
                logger.info(f"Cleaned up audio file: {audio_path}")
                
    async def batch_process_videos(self, video_urls: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        Process a batch of YouTube videos.
        
        Args:
            video_urls: A list of YouTube video URLs.
            **kwargs: Additional arguments to pass to process_video.
            
        Returns:
            A list of results, one for each video.
        """
        results = []
        for url in video_urls:
            try:
                result = await self.process_video(url, **kwargs)
                results.append(result)
            except Exception as e:
                results.append({"video_url": url, "error": str(e)})
        return results

    def search_videos_by_sentiment(
        self,
        currency_pair: Optional[str] = None,
        min_sentiment: float = -1.0,
        max_sentiment: float = 1.0,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for processed videos based on sentiment score and currency pair.
        
        Note: This is a placeholder for a more robust database query.
        This implementation is for demonstration and will be inefficient on large datasets.
        """
        # This is where you would typically query your database (e.g., Supabase)
        # For now, we are not implementing this part as it depends on the DB schema.
        logger.warning("search_videos_by_sentiment is a placeholder and not implemented.")
        return [] 
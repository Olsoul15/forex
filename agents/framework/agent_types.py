"""
Agent Types and Message Definitions for the Forex AI system.

This module defines the message types and agent interfaces used for communication
between different AI agents in the Forex AI system.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field

# Setup logging
logger = logging.getLogger(__name__)

# -------------------- Message Types for Agent Communication -------------------- #


class Message(BaseModel):
    """Base class for all message types."""

    timestamp: datetime = Field(default_factory=datetime.now)
    message_id: str = Field(default_factory=lambda: f"msg_{datetime.now().timestamp()}")


class UserQuery(Message):
    """A query from the user to the system."""

    query_text: str
    query_type: Optional[str] = None  # "strategy", "analysis", "general", etc.
    context: Dict[str, Any] = Field(default_factory=dict)


class SystemResponse(Message):
    """A response from the system to the user."""

    response_text: str
    confidence: float = 1.0
    source_agent: str = "system"


class MarketData(Message):
    """Market data used as input to analysis agents."""

    pairs: List[str]
    timeframes: List[str]
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    indicators: Dict[str, Any] = Field(default_factory=dict)
    price_data: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    news_data: List[Dict[str, Any]] = Field(default_factory=list)
    economic_data: Dict[str, Any] = Field(default_factory=dict)


class TechnicalAnalysisResult(Message):
    """Results from technical analysis."""

    pair: str
    timeframe: str
    trend: str  # "bullish", "bearish", "neutral", "ranging"
    trend_strength: float  # 0.0 to 1.0
    support_levels: List[float] = Field(default_factory=list)
    resistance_levels: List[float] = Field(default_factory=list)
    indicator_values: Dict[str, float] = Field(default_factory=dict)
    patterns: List[Dict[str, Any]] = Field(default_factory=list)
    key_levels: Dict[str, float] = Field(default_factory=dict)


class FundamentalAnalysisResult(Message):
    """Results from fundamental analysis."""

    pair: str
    economic_indicators: Dict[str, Any] = Field(default_factory=dict)
    news_impact: Dict[str, float] = Field(default_factory=dict)
    central_bank_outlook: Dict[str, str] = Field(default_factory=dict)
    economic_events: List[Dict[str, Any]] = Field(default_factory=list)
    fundamental_rating: str = "neutral"  # "bullish", "bearish", "neutral", "mixed"
    rating_confidence: float = 0.5  # 0.0 to 1.0


class SentimentAnalysisResult(Message):
    """Results from sentiment analysis."""

    pair: str
    market_sentiment: str  # "bullish", "bearish", "neutral", "mixed"
    sentiment_score: float  # -1.0 to 1.0
    key_topics: List[str] = Field(default_factory=list)
    source_distribution: Dict[str, float] = Field(default_factory=dict)
    social_media_trends: Dict[str, Any] = Field(default_factory=dict)
    news_sentiment: Dict[str, float] = Field(default_factory=dict)
    confidence: float = 0.5  # 0.0 to 1.0


class StrategyRecommendation(Message):
    """A strategy recommendation from the Strategy Agent."""

    strategy_type: str
    pair: str
    timeframe: str
    entry_conditions: List[Dict[str, Any]]
    exit_conditions: List[Dict[str, Any]]
    risk_management: Dict[str, Any]
    rationale: str
    confidence: float = 0.5  # 0.0 to 1.0
    supporting_analysis: Dict[str, Any] = Field(default_factory=dict)


class StrategyExecutionRequest(Message):
    """A request to execute a trading strategy."""

    strategy_id: str
    pair: str
    timeframe: str
    direction: str  # "buy" or "sell"
    size: float
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class StrategyExecutionResult(Message):
    """Result of a strategy execution."""

    execution_id: str
    strategy_id: str
    pair: str
    status: str  # "pending", "executed", "failed", "cancelled"
    order_details: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None


# -------------------- Agent Base Classes -------------------- #


class Agent(BaseModel):
    """Base class for all agents in the system."""

    name: str
    description: str = ""
    config: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    async def process(self, message: Message) -> Message:
        """
        Process a message and return a response.

        This method should be overridden by subclasses.

        Args:
            message: The input message

        Returns:
            Message: The response message
        """
        raise NotImplementedError("Agent subclasses must implement process method")


class Chain(BaseModel):
    """A chain of agents that process messages in sequence."""

    agents: List[Agent]
    name: str = "default_chain"

    class Config:
        arbitrary_types_allowed = True

    async def run(self, message: Message) -> Message:
        """
        Run the message through the chain of agents.

        Args:
            message: The input message

        Returns:
            Message: The final response message
        """
        current_message = message

        for agent in self.agents:
            logger.info(f"Running agent {agent.name} in chain {self.name}")
            current_message = await agent.process(current_message)

        return current_message


class AnalysisResult:
    """
    Result of an analysis operation.

    This class represents the result of an analysis operation
    performed by an agent.
    """

    def __init__(
        self,
        success: bool = True,
        pair: str = "",
        timeframe: str = "",
        data: Optional[Dict[str, Any]] = None,
        message: str = "",
    ):
        """
        Initialize the analysis result.

        Args:
            success: Whether the analysis was successful
            pair: Currency pair analyzed
            timeframe: Timeframe of the analysis
            data: Analysis data
            message: Message describing the result
        """
        self.success = success
        self.pair = pair
        self.timeframe = timeframe
        self.data = data or {}
        self.message = message

    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "pair": self.pair,
            "timeframe": self.timeframe,
            "data": self.data,
            "message": self.message,
        }

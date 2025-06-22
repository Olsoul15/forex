"""
Strategy Models for Forex AI Trading System.

This module provides data models for strategy management.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class StrategyType(str, Enum):
    """Strategy type enumeration."""

    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    PATTERN_RECOGNITION = "pattern_recognition"
    MOMENTUM = "momentum"
    MACHINE_LEARNING = "machine_learning"
    SENTIMENT_BASED = "sentiment_based"
    ARBITRAGE = "arbitrage"
    CUSTOM = "custom"


class TimeFrame(str, Enum):
    """Timeframe enumeration."""

    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    M30 = "M30"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"
    W1 = "W1"
    MN1 = "MN1"


class RiskProfile(str, Enum):
    """Risk profile enumeration."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ImportExportFormat(str, Enum):
    """Import/export format enumeration."""

    JSON = "json"
    YAML = "yaml"
    PYTHON = "python"
    PINESCRIPT = "pinescript"


class ParameterType(str, Enum):
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    ENUM = "enum"


class StrategyBase(BaseModel):
    """Base model for strategy data."""

    name: str
    description: str
    strategy_type: StrategyType
    timeframes: List[TimeFrame]
    instruments: List[str]
    risk_profile: RiskProfile
    is_active: bool = True
    parameters: Dict[str, Any] = Field(default_factory=dict)
    parameter_definitions: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    source_code: Optional[str] = None


class StrategyCreate(StrategyBase):
    """Model for strategy creation."""

    pass


class StrategyUpdate(BaseModel):
    """Model for strategy update."""

    name: Optional[str] = None
    description: Optional[str] = None
    strategy_type: Optional[StrategyType] = None
    timeframes: Optional[List[TimeFrame]] = None
    instruments: Optional[List[str]] = None
    risk_profile: Optional[RiskProfile] = None
    is_active: Optional[bool] = None
    parameters: Optional[Dict[str, Any]] = None
    parameter_definitions: Optional[Dict[str, Dict[str, Any]]] = None
    source_code: Optional[str] = None


class Strategy(StrategyBase):
    """Complete strategy model."""

    id: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    backtest_results: Dict[str, Any] = Field(default_factory=dict)
    performance_metrics: Dict[str, Any] = Field(default_factory=dict)


class StrategyListItem(BaseModel):
    """Simplified strategy model for list responses."""

    id: str
    name: str
    description: str
    strategy_type: StrategyType
    timeframes: List[TimeFrame]
    instruments: List[str]
    risk_profile: RiskProfile
    is_active: bool
    created_at: datetime
    performance_summary: Optional[Dict[str, Any]] = None


class StrategyDetail(Strategy):
    """Detailed strategy model with additional information."""

    execution_history: List[Dict[str, Any]] = Field(default_factory=list)
    similar_strategies: List[Dict[str, Any]] = Field(default_factory=list)


class StrategyListResponse(BaseModel):
    """Response model for strategy list."""

    strategies: List[StrategyListItem]
    count: int
    timestamp: datetime


class StrategyDetailResponse(BaseModel):
    """Response model for strategy detail."""

    strategy: StrategyDetail
    timestamp: datetime


class StrategyRecommendation(BaseModel):
    """Recommendation model for strategies."""

    strategy_id: str
    name: str
    score: float
    reason: str
    performance_summary: Dict[str, Any]


class StrategyRecommendationResponse(BaseModel):
    """Response model for strategy recommendations."""

    recommendations: List[StrategyRecommendation]
    instrument: str
    timeframe: TimeFrame
    timestamp: datetime


class StrategyEvaluation(BaseModel):
    """Evaluation model for strategies."""

    strategy_id: str
    name: str
    strengths: List[str]
    weaknesses: List[str]
    opportunities: List[str]
    threats: List[str]
    score: float
    recommendations: List[str]


class StrategyEvaluationResponse(BaseModel):
    """Response model for strategy evaluation."""

    evaluation: StrategyEvaluation
    timestamp: datetime


class StrategyImportRequest(BaseModel):
    """Request model for strategy import."""

    content: str
    format: ImportExportFormat = ImportExportFormat.JSON
    overwrite_existing: bool = False


class StrategyExportRequest(BaseModel):
    """Request model for strategy export."""

    format: ImportExportFormat = ImportExportFormat.JSON
    include_history: bool = False
    include_backtests: bool = True


class StrategyImportResponse(BaseModel):
    """Response model for strategy import."""

    success: bool
    strategy_id: Optional[str] = None
    message: str
    timestamp: datetime


class StrategyExportResponse(BaseModel):
    """Response model for strategy export."""

    success: bool
    content: Optional[str] = None
    format: ImportExportFormat
    strategy_id: str
    timestamp: datetime 
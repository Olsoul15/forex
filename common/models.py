"""
Common data models for the AI Forex system.

This module provides reusable data models that are used across
different components of the system.
"""

from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


# Error models
class ErrorResponse(BaseModel):
    """Error response model for API endpoints."""

    success: bool = False
    detail: str
    code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Analysis models
class AnalysisResponse(BaseModel):
    """Response model for analysis endpoints."""

    success: bool = True
    message: str
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AnalysisRequest(BaseModel):
    """Request model for analysis endpoints."""

    pair: str
    timeframe: str
    analysis_type: Optional[str] = "technical"
    include_context: Optional[bool] = True
    additional_params: Optional[Dict[str, Any]] = Field(default_factory=dict)


# Pattern models
class PatternResult(BaseModel):
    """Model for pattern detection results."""

    name: str
    type: str
    direction: str
    reliability: str
    confidence: float
    completion: Optional[float] = None
    description: Optional[str] = None
    requirements: Optional[List[Dict[str, Any]]] = None


# Risk metrics models
class RiskMetric(BaseModel):
    """Model for risk metrics."""

    name: str
    value: float
    category: str
    description: Optional[str] = None
    status: Optional[str] = None


# Elliott Wave models
class ElliottWaveAnalysis(BaseModel):
    """Model for Elliott Wave analysis results."""

    currentWave: str
    position: str
    confidence: float
    nextMove: Dict[str, str]

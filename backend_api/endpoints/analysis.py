"""
Analysis API endpoints.

This module provides FastAPI endpoints for AI analysis services,
including the context-aware analysis capabilities.
"""

from typing import Dict, Any, Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query
import logging

from forex_ai.agents.agent_manager import get_agent_manager
from forex_ai.agents.context_aware_analyzer import ContextAwareAnalyzer
from forex_ai.common.models import AnalysisResponse, ErrorResponse
from forex_ai.common.validators import validate_pair, validate_timeframe
from forex_ai.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai", tags=["AI Analysis"])


@router.get(
    "/analysis",
    response_model=AnalysisResponse,
    responses={
        200: {"description": "Successful response with analysis data"},
        400: {"description": "Bad request", "model": ErrorResponse},
        500: {"description": "Server error", "model": ErrorResponse},
    },
)
async def get_analysis(
    pair: str = Query(..., description="Currency pair to analyze, e.g. EURUSD"),
    timeframe: str = Query(..., description="Timeframe for analysis, e.g. 1h, 4h, 1d"),
    analysis_type: str = Query("technical", description="Type of analysis to perform"),
    include_context: bool = Query(
        True, description="Whether to include previous context in analysis"
    ),
):
    """
    Get AI analysis for a currency pair on a specific timeframe.

    This endpoint provides comprehensive technical analysis including:
    - Multi-timeframe analysis
    - Pattern recognition
    - Risk metrics evaluation
    - Elliott Wave analysis (when applicable)

    With context-aware capabilities, the analysis builds upon previous findings,
    identifying changes and trends over time for more nuanced insights.

    Parameters:
        pair: Currency pair to analyze
        timeframe: Timeframe for analysis
        analysis_type: Type of analysis (technical, fundamental, sentiment)
        include_context: Whether to include context from previous analyses

    Returns:
        Analysis results including indicators, patterns, and contextual insights
    """
    try:
        # Validate inputs
        validate_pair(pair)
        validate_timeframe(timeframe)

        # Get the context-aware analyzer from the agent manager
        agent_manager = get_agent_manager()
        analyzer = agent_manager.get_agent("ContextAnalyzer")

        if not analyzer or not isinstance(analyzer, ContextAwareAnalyzer):
            # Create a new analyzer if it doesn't exist yet
            analyzer = ContextAwareAnalyzer(name="ContextAnalyzer")
            agent_manager.register_agent(analyzer)

        # Perform analysis with context awareness
        result = await analyzer.process(
            {
                "pair": pair,
                "timeframe": timeframe,
                "analysis_type": analysis_type,
                "include_context": include_context,
            }
        )

        # Check if analysis was successful
        if not result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=f"Analysis failed: {result.get('message', 'Unknown error')}",
            )

        # Return the analysis data
        return AnalysisResponse(
            success=True,
            data=result.get("data", {}),
            message="Analysis completed successfully",
        )

    except ValueError as e:
        # Input validation error
        logger.warning(f"Validation error in analysis request: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        # Server error
        logger.error(f"Error performing analysis: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

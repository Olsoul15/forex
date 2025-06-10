"""
Strategy API Endpoints for Forex AI Trading System.

This module provides FastAPI endpoints for strategy management.
"""

import logging
import json
import yaml
from typing import Dict, List, Any, Optional
from datetime import datetime
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
    Query,
    Path,
    Body,
    File,
    UploadFile,
)
from fastapi.responses import JSONResponse

from app.models.strategy_models import (
    Strategy,
    StrategyBase,
    StrategyCreate,
    StrategyUpdate,
    StrategyType,
    TimeFrame,
    RiskProfile,
    StrategyListResponse,
    StrategyDetailResponse,
    StrategyListItem,
    StrategyDetail,
    StrategyRecommendation,
    StrategyRecommendationResponse,
    StrategyEvaluation,
    StrategyEvaluationResponse,
    ImportExportFormat,
    StrategyImportRequest,
    StrategyExportRequest,
    StrategyImportResponse,
    StrategyExportResponse,
)
from app.db.strategy_db import (
    get_all_strategies,
    get_strategy_by_id,
    create_strategy,
    update_strategy,
    delete_strategy,
    get_strategies_by_type,
    get_strategies_for_instrument,
    get_strategies_for_timeframe,
)

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/strategies", tags=["strategies"])


# Helper functions
def convert_to_list_item(strategy: Strategy) -> StrategyListItem:
    """Convert a Strategy to a StrategyListItem."""
    performance_summary = None
    if strategy.performance_metrics:
        performance_summary = {
            "win_rate": strategy.performance_metrics.get("win_rate"),
            "profit_factor": strategy.performance_metrics.get("profit_factor"),
            "max_drawdown": strategy.performance_metrics.get("max_drawdown"),
        }

    return StrategyListItem(
        id=strategy.id,
        name=strategy.name,
        description=strategy.description,
        strategy_type=strategy.strategy_type,
        timeframes=strategy.timeframes,
        instruments=strategy.instruments,
        risk_profile=strategy.risk_profile,
        is_active=strategy.is_active,
        created_at=strategy.created_at,
        performance_summary=performance_summary,
    )


def strategy_to_detail(strategy: Strategy) -> StrategyDetail:
    """Convert a Strategy to a StrategyDetail."""
    return StrategyDetail(
        id=strategy.id,
        name=strategy.name,
        description=strategy.description,
        strategy_type=strategy.strategy_type,
        timeframes=strategy.timeframes,
        instruments=strategy.instruments,
        risk_profile=strategy.risk_profile,
        is_active=strategy.is_active,
        parameters=strategy.parameters,
        parameter_definitions=strategy.parameter_definitions,
        created_at=strategy.created_at,
        updated_at=strategy.updated_at,
        source_code=strategy.source_code,
        backtest_results=strategy.backtest_results,
        performance_metrics=strategy.performance_metrics,
        execution_history=[],  # This would come from a real execution history database
        similar_strategies=[],  # This would be populated by a recommendation engine
    )


# Endpoints
@router.get("/", response_model=StrategyListResponse)
async def get_strategies(
    strategy_type: Optional[StrategyType] = Query(
        None, description="Filter by strategy type"
    ),
    instrument: Optional[str] = Query(None, description="Filter by instrument"),
    timeframe: Optional[TimeFrame] = Query(None, description="Filter by timeframe"),
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
):
    """
    Get all available strategies.

    Returns a list of all strategies in the system with optional filtering.
    """
    logger.info("Processing get strategies request")

    # Get all strategies
    all_strategies = get_all_strategies()

    # Apply filters
    filtered_strategies = all_strategies

    if strategy_type:
        filtered_strategies = [
            s for s in filtered_strategies if s.strategy_type == strategy_type
        ]

    if instrument:
        filtered_strategies = [
            s for s in filtered_strategies if instrument in s.instruments
        ]

    if timeframe:
        filtered_strategies = [
            s for s in filtered_strategies if timeframe in s.timeframes
        ]

    if is_active is not None:
        filtered_strategies = [
            s for s in filtered_strategies if s.is_active == is_active
        ]

    # Convert to list items
    strategy_items = [convert_to_list_item(s) for s in filtered_strategies]

    return StrategyListResponse(
        strategies=strategy_items, count=len(strategy_items), timestamp=datetime.now()
    )


@router.get("/{strategy_id}", response_model=StrategyDetailResponse)
async def get_strategy(
    strategy_id: str = Path(..., description="The ID of the strategy to retrieve")
):
    """
    Get detailed information about a specific strategy.

    Returns comprehensive details about the strategy, including parameters,
    performance metrics, and execution history.
    """
    logger.info(f"Processing get strategy request for ID: {strategy_id}")

    # Get strategy
    strategy = get_strategy_by_id(strategy_id)
    if not strategy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy with ID {strategy_id} not found",
        )

    # Convert to detail
    strategy_detail = strategy_to_detail(strategy)

    # In a real implementation, we would also load:
    # - Recent execution history
    # - Similar strategies (recommendation engine)
    # - Latest backtest results

    return StrategyDetailResponse(strategy=strategy_detail, timestamp=datetime.now())


@router.post(
    "/", response_model=StrategyDetailResponse, status_code=status.HTTP_201_CREATED
)
async def create_new_strategy(
    strategy_data: StrategyCreate = Body(..., description="Strategy creation data")
):
    """
    Create a new trading strategy.

    Creates a new strategy with the provided details and returns the created strategy.
    """
    logger.info(f"Processing create strategy request: {strategy_data.name}")

    # Create strategy
    try:
        strategy = create_strategy(strategy_data)
    except Exception as e:
        logger.error(f"Error creating strategy: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error creating strategy: {str(e)}",
        )

    # Convert to detail
    strategy_detail = strategy_to_detail(strategy)

    return StrategyDetailResponse(strategy=strategy_detail, timestamp=datetime.now())


@router.put("/{strategy_id}", response_model=StrategyDetailResponse)
async def update_existing_strategy(
    strategy_id: str = Path(..., description="The ID of the strategy to update"),
    strategy_data: StrategyUpdate = Body(..., description="Strategy update data"),
):
    """
    Update an existing strategy.

    Updates the specified strategy with the provided details and returns the updated strategy.
    """
    logger.info(f"Processing update strategy request for ID: {strategy_id}")

    # Update strategy
    updated_strategy = update_strategy(strategy_id, strategy_data)
    if not updated_strategy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy with ID {strategy_id} not found",
        )

    # Convert to detail
    strategy_detail = strategy_to_detail(updated_strategy)

    return StrategyDetailResponse(strategy=strategy_detail, timestamp=datetime.now())


@router.delete("/{strategy_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_existing_strategy(
    strategy_id: str = Path(..., description="The ID of the strategy to delete")
):
    """
    Delete a strategy.

    Permanently removes the specified strategy from the system.
    """
    logger.info(f"Processing delete strategy request for ID: {strategy_id}")

    # Delete strategy
    success = delete_strategy(strategy_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy with ID {strategy_id} not found",
        )

    return None


@router.get(
    "/recommendations/{instrument}/{timeframe}",
    response_model=StrategyRecommendationResponse,
)
async def get_strategy_recommendations(
    instrument: str = Path(
        ..., description="The instrument to get recommendations for"
    ),
    timeframe: TimeFrame = Path(
        ..., description="The timeframe to get recommendations for"
    ),
    limit: int = Query(
        5, description="Maximum number of recommendations to return", ge=1, le=20
    ),
):
    """
    Get strategy recommendations for a specific instrument and timeframe.

    Returns a list of recommended strategies based on current market conditions
    and historical performance.
    """
    logger.info(f"Processing strategy recommendations for {instrument} on {timeframe}")

    # In a real implementation, this would use an AI/ML model to make recommendations
    # For now, we'll return mock data

    # Get strategies for this instrument and timeframe
    all_strategies = get_all_strategies()
    suitable_strategies = [
        s
        for s in all_strategies
        if instrument in s.instruments and timeframe in s.timeframes
    ]

    # If we don't have enough, also include strategies just for this instrument
    if len(suitable_strategies) < limit:
        for s in get_strategies_for_instrument(instrument):
            if s not in suitable_strategies:
                suitable_strategies.append(s)
                if len(suitable_strategies) >= limit:
                    break

    # If we still don't have enough, include any strategies for this timeframe
    if len(suitable_strategies) < limit:
        for s in get_strategies_for_timeframe(timeframe):
            if s not in suitable_strategies:
                suitable_strategies.append(s)
                if len(suitable_strategies) >= limit:
                    break

    # Create recommendations
    recommendations = []
    for i, strategy in enumerate(suitable_strategies[:limit]):
        # Calculate a mock score - in reality this would be from a model
        score = 0.9 - (i * 0.1)
        if score < 0.5:
            score = 0.5

        # Generate a reason
        if i == 0:
            reason = f"Best historical performance for {instrument} on {timeframe}"
        elif i == 1:
            reason = f"Good risk-adjusted returns for {instrument}"
        elif i == 2:
            reason = f"Performs well in current market conditions"
        else:
            reason = f"Alternative strategy for {instrument}"

        recommendations.append(
            StrategyRecommendation(
                strategy_id=strategy.id,
                name=strategy.name,
                score=score,
                reason=reason,
                suitable_timeframes=[tf for tf in strategy.timeframes],
                suitable_instruments=[ins for ins in strategy.instruments],
            )
        )

    return StrategyRecommendationResponse(
        recommendations=recommendations,
        pair=instrument,
        timeframe=timeframe,
        timestamp=datetime.now(),
    )


@router.post("/{strategy_id}/evaluate", response_model=StrategyEvaluationResponse)
async def evaluate_strategy(
    strategy_id: str = Path(..., description="The ID of the strategy to evaluate")
):
    """
    Evaluate a strategy for quality and improvement opportunities.

    Analyzes the strategy and provides insights on its strengths, weaknesses,
    and potential optimizations.
    """
    logger.info(f"Processing strategy evaluation for ID: {strategy_id}")

    # Get strategy
    strategy = get_strategy_by_id(strategy_id)
    if not strategy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy with ID {strategy_id} not found",
        )

    # In a real implementation, this would use advanced analysis
    # For now, we'll return mock data

    # Calculate mock score based on performance metrics
    score = 0.0
    if strategy.performance_metrics:
        win_rate = strategy.performance_metrics.get("win_rate", 0.5)
        profit_factor = strategy.performance_metrics.get("profit_factor", 1.0)

        # Simple score calculation
        score = (win_rate * 0.4) + (min(profit_factor / 3, 1.0) * 0.6)
        score = min(round(score * 100) / 100, 1.0)
    else:
        score = 0.6  # Default score

    # Generate generic evaluation
    strengths = []
    weaknesses = []
    suggestions = []

    # Add strategy-specific evaluations
    if strategy.strategy_type == StrategyType.TREND_FOLLOWING:
        strengths.append("Good for stable trending markets")
        weaknesses.append("May underperform in sideways markets")
        suggestions.append("Consider adding a trend filter to avoid false signals")
    elif strategy.strategy_type == StrategyType.BREAKOUT:
        strengths.append("Good for capturing large moves")
        weaknesses.append("Subject to false breakouts")
        suggestions.append("Add volume confirmation to reduce false breakouts")
    elif strategy.strategy_type == StrategyType.SCALPING:
        strengths.append("Generates frequent trading opportunities")
        weaknesses.append("Transaction costs can significantly impact profitability")
        suggestions.append("Optimize for minimal spreads and commission impact")

    # Add score-based evaluations
    if score > 0.8:
        strengths.append("Excellent historical performance")
    elif score > 0.6:
        strengths.append("Good overall performance")
    else:
        weaknesses.append("Suboptimal performance metrics")
        suggestions.append("Consider parameter optimization or strategy revision")

    # Add parameter-based suggestions
    if strategy.parameters:
        if (
            "fast_ma_period" in strategy.parameters
            and "slow_ma_period" in strategy.parameters
        ):
            fast_period = strategy.parameters["fast_ma_period"]
            slow_period = strategy.parameters["slow_ma_period"]

            if slow_period - fast_period < 10:
                weaknesses.append("Moving average periods are too close together")
                suggestions.append(
                    "Increase separation between fast and slow moving averages"
                )

    # Create mock optimized parameters
    optimized_parameters = dict(strategy.parameters) if strategy.parameters else {}

    evaluation = StrategyEvaluation(
        score=score,
        strengths=strengths,
        weaknesses=weaknesses,
        suggestions=suggestions,
        optimized_parameters=optimized_parameters,
    )

    return StrategyEvaluationResponse(
        evaluation=evaluation, strategy_id=strategy_id, timestamp=datetime.now()
    )


@router.post("/import", response_model=StrategyImportResponse)
async def import_strategy(
    import_request: StrategyImportRequest = Body(
        ..., description="Strategy import request"
    )
):
    """
    Import a strategy from an external format.

    Imports a strategy from various formats like JSON, YAML, Pine Script, or Python.
    """
    logger.info(f"Processing strategy import in {import_request.format} format")

    try:
        # Parse the content based on format
        if import_request.format == ImportExportFormat.JSON:
            data = json.loads(import_request.content)
        elif import_request.format == ImportExportFormat.YAML:
            data = yaml.safe_load(import_request.content)
        else:
            # For Pine Script and Python, we'd need a more sophisticated parser
            # This is just a placeholder implementation
            raise NotImplementedError(
                f"Import from {import_request.format} not yet implemented"
            )

        # Convert to StrategyCreate
        strategy_data = StrategyCreate(
            name=import_request.name_override or data.get("name", "Imported Strategy"),
            description=data.get("description", "Imported strategy"),
            strategy_type=data.get("strategy_type", StrategyType.CUSTOM),
            timeframes=[TimeFrame(tf) for tf in data.get("timeframes", ["H1"])],
            instruments=data.get("instruments", ["EUR_USD"]),
            risk_profile=RiskProfile(data.get("risk_profile", "moderate")),
            parameters=data.get("parameters", {}),
            source_code=import_request.content,
        )

        # Create strategy
        strategy = create_strategy(strategy_data)

        return StrategyImportResponse(
            strategy_id=strategy.id,
            name=strategy.name,
            success=True,
            message=f"Successfully imported strategy in {import_request.format} format",
            timestamp=datetime.now(),
        )

    except Exception as e:
        logger.error(f"Error importing strategy: {str(e)}", exc_info=True)
        # Return error response without raising exception
        return StrategyImportResponse(
            strategy_id="",
            name="",
            success=False,
            message=f"Error importing strategy: {str(e)}",
            timestamp=datetime.now(),
        )


@router.post("/{strategy_id}/export", response_model=StrategyExportResponse)
async def export_strategy(
    strategy_id: str = Path(..., description="The ID of the strategy to export"),
    export_request: StrategyExportRequest = Body(
        ..., description="Strategy export request"
    ),
):
    """
    Export a strategy to an external format.

    Exports a strategy to various formats like JSON, YAML, Pine Script, or Python.
    """
    logger.info(
        f"Processing strategy export for ID: {strategy_id} in {export_request.format} format"
    )

    # Get strategy
    strategy = get_strategy_by_id(strategy_id)
    if not strategy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy with ID {strategy_id} not found",
        )

    try:
        # Prepare data for export
        export_data = {
            "name": strategy.name,
            "description": strategy.description,
            "strategy_type": strategy.strategy_type,
            "timeframes": [str(tf) for tf in strategy.timeframes],
            "instruments": strategy.instruments,
            "risk_profile": strategy.risk_profile,
            "parameters": strategy.parameters,
        }

        # Convert to requested format
        content = ""
        if export_request.format == ImportExportFormat.JSON:
            content = json.dumps(export_data, indent=2)
        elif export_request.format == ImportExportFormat.YAML:
            content = yaml.dump(export_data)
        else:
            # For Pine Script and Python, we'd need a more sophisticated converter
            # This is just a placeholder implementation
            raise NotImplementedError(
                f"Export to {export_request.format} not yet implemented"
            )

        return StrategyExportResponse(
            strategy_id=strategy_id,
            name=strategy.name,
            format=export_request.format,
            content=content,
            timestamp=datetime.now(),
        )

    except Exception as e:
        logger.error(f"Error exporting strategy: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error exporting strategy: {str(e)}",
        )


@router.get("/types", response_model=List[Dict[str, str]])
async def get_strategy_types():
    """
    Get available strategy types.

    Returns a list of available strategy types and their descriptions.
    """
    logger.info("Processing get strategy types request")

    # Create a list of strategy types with descriptions
    strategy_types = [
        {
            "id": StrategyType.TEMPLATE,
            "name": "Template",
            "description": "Basic template strategies",
        },
        {
            "id": StrategyType.TREND_FOLLOWING,
            "name": "Trend Following",
            "description": "Strategies that follow market trends",
        },
        {
            "id": StrategyType.MEAN_REVERSION,
            "name": "Mean Reversion",
            "description": "Strategies that trade mean reversion",
        },
        {
            "id": StrategyType.BREAKOUT,
            "name": "Breakout",
            "description": "Strategies that trade breakouts from ranges",
        },
        {
            "id": StrategyType.CANDLESTICK,
            "name": "Candlestick",
            "description": "Strategies based on candlestick patterns",
        },
        {
            "id": StrategyType.MARKET_STATE,
            "name": "Market State",
            "description": "Strategies that adapt to market conditions",
        },
        {
            "id": StrategyType.SCALPING,
            "name": "Scalping",
            "description": "High-frequency strategies for small profits",
        },
        {
            "id": StrategyType.MOMENTUM_INTRADAY,
            "name": "Momentum (Intraday)",
            "description": "Day trading strategies based on momentum",
        },
        {
            "id": StrategyType.SUPPORT_RESISTANCE_INTRADAY,
            "name": "Support/Resistance (Intraday)",
            "description": "Day trading strategies based on key levels",
        },
        {
            "id": StrategyType.VOLATILITY_BREAKOUT,
            "name": "Volatility Breakout",
            "description": "Strategies that trade volatility expansions",
        },
        {
            "id": StrategyType.SESSION_TRANSITION,
            "name": "Session Transition",
            "description": "Strategies that trade market session transitions",
        },
        {
            "id": StrategyType.AUTO_GENERATED,
            "name": "Auto-Generated",
            "description": "AI-generated strategies",
        },
        {
            "id": StrategyType.CUSTOM,
            "name": "Custom",
            "description": "User-defined custom strategies",
        },
    ]

    return strategy_types


@router.get("/{strategy_id}/parameters", response_model=Dict[str, Any])
async def get_strategy_parameters(
    strategy_id: str = Path(
        ..., description="The ID of the strategy to get parameters for"
    )
):
    """
    Get parameters for a specific strategy.

    Returns all parameters and their definitions for the specified strategy.
    """
    logger.info(f"Processing get strategy parameters for ID: {strategy_id}")

    # Get strategy
    strategy = get_strategy_by_id(strategy_id)
    if not strategy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy with ID {strategy_id} not found",
        )

    # Create response with parameters and their definitions
    response = {
        "current_values": strategy.parameters or {},
        "definitions": (
            [pd.dict() for pd in strategy.parameter_definitions]
            if strategy.parameter_definitions
            else []
        ),
    }

    return response


@router.put("/{strategy_id}/parameters", response_model=Dict[str, Any])
async def update_strategy_parameters(
    strategy_id: str = Path(
        ..., description="The ID of the strategy to update parameters for"
    ),
    parameters: Dict[str, Any] = Body(..., description="Updated parameter values"),
):
    """
    Update parameters for a specific strategy.

    Updates the parameter values for the specified strategy.
    """
    logger.info(f"Processing update strategy parameters for ID: {strategy_id}")

    # Get strategy
    strategy = get_strategy_by_id(strategy_id)
    if not strategy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy with ID {strategy_id} not found",
        )

    # Validate parameters against definitions
    if strategy.parameter_definitions:
        for param_def in strategy.parameter_definitions:
            param_name = param_def.name

            # Skip parameters that aren't being updated
            if param_name not in parameters:
                continue

            param_value = parameters[param_name]

            # Type validation
            if param_def.type == "integer" and not isinstance(param_value, int):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Parameter {param_name} must be an integer",
                )
            elif param_def.type == "float" and not isinstance(
                param_value, (int, float)
            ):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Parameter {param_name} must be a number",
                )
            elif param_def.type == "boolean" and not isinstance(param_value, bool):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Parameter {param_name} must be a boolean",
                )
            elif param_def.type == "string" and not isinstance(param_value, str):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Parameter {param_name} must be a string",
                )
            elif param_def.type == "enum" and param_value not in param_def.options:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Parameter {param_name} must be one of: {', '.join(param_def.options)}",
                )

            # Range validation
            if param_def.type in ["integer", "float"]:
                if (
                    param_def.min_value is not None
                    and param_value < param_def.min_value
                ):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Parameter {param_name} must be >= {param_def.min_value}",
                    )
                if (
                    param_def.max_value is not None
                    and param_value > param_def.max_value
                ):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Parameter {param_name} must be <= {param_def.max_value}",
                    )

    # Update parameters
    strategy_update = StrategyUpdate(parameters=parameters)
    updated_strategy = update_strategy(strategy_id, strategy_update)

    return {"parameters": updated_strategy.parameters or {}}


@router.post("/by-type", response_model=StrategyListResponse)
async def get_strategies_by_types(
    strategy_types: List[StrategyType] = Body(
        ..., description="List of strategy types to filter by"
    ),
    timeframe: Optional[TimeFrame] = Query(None, description="Filter by timeframe"),
    instrument: Optional[str] = Query(None, description="Filter by instrument"),
):
    """
    Get strategies by multiple types.

    Returns strategies that match any of the specified strategy types,
    with optional filtering by timeframe and instrument.
    """
    logger.info(f"Processing get strategies by types: {strategy_types}")

    # Get all strategies
    all_strategies = get_all_strategies()

    # Filter by strategy types
    filtered_strategies = [
        s for s in all_strategies if s.strategy_type in strategy_types
    ]

    # Apply additional filters
    if timeframe:
        filtered_strategies = [
            s for s in filtered_strategies if timeframe in s.timeframes
        ]

    if instrument:
        filtered_strategies = [
            s for s in filtered_strategies if instrument in s.instruments
        ]

    # Convert to list items
    strategy_items = [convert_to_list_item(s) for s in filtered_strategies]

    return StrategyListResponse(
        strategies=strategy_items, count=len(strategy_items), timestamp=datetime.now()
    )

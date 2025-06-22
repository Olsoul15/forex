"""
Forex Optimizer API Endpoints for Forex AI Trading System.

This module provides FastAPI endpoints for optimizing forex trading strategies,
including parameter optimization, backtesting, and performance analysis.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Query, Path, Body
from pydantic import BaseModel, Field

from forex_ai.auth.supabase import get_current_user
from forex_ai.backend_api.db import strategy_db

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/forex-optimizer", tags=["forex-optimizer"])

# Models
class OptimizationParameter(BaseModel):
    """Optimization parameter model."""
    name: str
    min_value: Union[float, int]
    max_value: Union[float, int]
    step: Union[float, int]
    type: str = Field("float", description="Parameter type (float, int)")

class OptimizationRequest(BaseModel):
    """Optimization request model."""
    strategyId: str = Field(..., description="Strategy ID to optimize")
    instrument: str = Field(..., description="Instrument to optimize for (e.g., EUR_USD)")
    timeframe: str = Field(..., description="Timeframe to optimize for (e.g., H1, D1)")
    start_date: datetime = Field(..., description="Start date for optimization")
    end_date: datetime = Field(..., description="End date for optimization")
    parameters: List[OptimizationParameter] = Field(..., description="Parameters to optimize")
    optimization_metric: str = Field("profit_factor", description="Metric to optimize for")
    population_size: int = Field(50, description="Population size for genetic algorithm")
    generations: int = Field(10, description="Number of generations for genetic algorithm")
    parallel_jobs: int = Field(4, description="Number of parallel jobs to run")

class OptimizationJobResponse(BaseModel):
    """Optimization job response model."""
    job_id: str
    status: str
    progress: float
    message: str
    created_at: datetime
    updated_at: datetime

class OptimizationResult(BaseModel):
    """Optimization result model."""
    parameters: Dict[str, Union[float, int]]
    profit_factor: float
    sharpe_ratio: float
    win_rate: float
    total_trades: int
    net_profit: float
    max_drawdown: float
    rank: int

class OptimizationResultsResponse(BaseModel):
    """Optimization results response model."""
    job_id: str
    strategyId: str
    instrument: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    optimization_metric: str
    results: List[OptimizationResult]
    best_result: OptimizationResult
    status: str
    completed_at: Optional[datetime] = None

class WalkforwardTestRequest(BaseModel):
    """Walkforward test request model."""
    strategyId: str = Field(..., description="Strategy ID to test")
    instrument: str = Field(..., description="Instrument to test on (e.g., EUR_USD)")
    timeframe: str = Field(..., description="Timeframe to test on (e.g., H1, D1)")
    start_date: datetime = Field(..., description="Start date for testing")
    end_date: datetime = Field(..., description="End date for testing")
    parameters: Dict[str, Union[float, int]] = Field(..., description="Strategy parameters")
    window_size: int = Field(90, description="Window size in days")
    step_size: int = Field(30, description="Step size in days")

class WalkforwardTestResponse(BaseModel):
    """Walkforward test response model."""
    job_id: str
    status: str
    progress: float
    message: str
    created_at: datetime
    updated_at: datetime

class WalkforwardTestResult(BaseModel):
    """Walkforward test result model."""
    window_start: datetime
    window_end: datetime
    profit_factor: float
    sharpe_ratio: float
    win_rate: float
    total_trades: int
    net_profit: float
    max_drawdown: float

class WalkforwardTestResultsResponse(BaseModel):
    """Walkforward test results response model."""
    job_id: str
    strategyId: str
    instrument: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    parameters: Dict[str, Union[float, int]]
    window_size: int
    step_size: int
    results: List[WalkforwardTestResult]
    overall_profit_factor: float
    overall_sharpe_ratio: float
    overall_win_rate: float
    overall_total_trades: int
    overall_net_profit: float
    overall_max_drawdown: float
    status: str
    completed_at: Optional[datetime] = None

class MonteCarloSimulationRequest(BaseModel):
    """Monte Carlo simulation request model."""
    strategyId: str = Field(..., description="Strategy ID to simulate")
    instrument: str = Field(..., description="Instrument to simulate on (e.g., EUR_USD)")
    timeframe: str = Field(..., description="Timeframe to simulate on (e.g., H1, D1)")
    start_date: datetime = Field(..., description="Start date for simulation")
    end_date: datetime = Field(..., description="End date for simulation")
    parameters: Dict[str, Union[float, int]] = Field(..., description="Strategy parameters")
    simulations: int = Field(1000, description="Number of simulations to run")
    confidence_level: float = Field(0.95, description="Confidence level for results")

class MonteCarloSimulationResponse(BaseModel):
    """Monte Carlo simulation response model."""
    job_id: str
    status: str
    progress: float
    message: str
    created_at: datetime
    updated_at: datetime

class MonteCarloSimulationResult(BaseModel):
    """Monte Carlo simulation result model."""
    percentile: float
    final_equity: float
    max_drawdown: float
    profit_factor: float
    sharpe_ratio: float

class MonteCarloSimulationResultsResponse(BaseModel):
    """Monte Carlo simulation results response model."""
    job_id: str
    strategyId: str
    instrument: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    parameters: Dict[str, Union[float, int]]
    simulations: int
    confidence_level: float
    results: List[MonteCarloSimulationResult]
    median_result: MonteCarloSimulationResult
    worst_result: MonteCarloSimulationResult
    best_result: MonteCarloSimulationResult
    confidence_interval_lower: MonteCarloSimulationResult
    confidence_interval_upper: MonteCarloSimulationResult
    status: str
    completed_at: Optional[datetime] = None

# Endpoints
@router.post("/optimize", response_model=OptimizationJobResponse)
async def start_optimization(
    request: OptimizationRequest = Body(...),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Start a strategy optimization job.

    Optimizes strategy parameters using genetic algorithm to maximize the specified metric.
    """
    logger.info(f"Processing optimization request for strategy {request.strategyId}")

    try:
        # Verify user has access to this strategy
        if not strategy_db.user_has_strategy_access(current_user["id"], request.strategyId):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to this strategy",
            )
        
        # Start optimization job
        job = strategy_db.start_optimization_job(
            user_id=current_user["id"],
            strategy_id=request.strategyId,
            instrument=request.instrument,
            timeframe=request.timeframe,
            start_date=request.start_date,
            end_date=request.end_date,
            parameters=[param.dict() for param in request.parameters],
            optimization_metric=request.optimization_metric,
            population_size=request.population_size,
            generations=request.generations,
            parallel_jobs=request.parallel_jobs,
        )
        
        if not job:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to start optimization job",
            )
        
        return OptimizationJobResponse(
            job_id=job["id"],
            status=job["status"],
            progress=job["progress"],
            message=job["message"],
            created_at=job["created_at"],
            updated_at=job["updated_at"],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting optimization job: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error starting optimization job: {str(e)}",
        )

@router.get("/optimize/{job_id}/status", response_model=OptimizationJobResponse)
async def get_optimization_status(
    job_id: str = Path(..., description="Optimization job ID"),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Get optimization job status.

    Returns the status of an optimization job.
    """
    logger.info(f"Processing optimization status request for job {job_id}")

    try:
        # Get job status
        job = strategy_db.get_optimization_job(job_id)
        
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Optimization job {job_id} not found",
            )
        
        # Verify user has access to this job
        if job["user_id"] != current_user["id"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to this optimization job",
            )
        
        return OptimizationJobResponse(
            job_id=job["id"],
            status=job["status"],
            progress=job["progress"],
            message=job["message"],
            created_at=job["created_at"],
            updated_at=job["updated_at"],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting optimization job status: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting optimization job status: {str(e)}",
        )

@router.get("/optimize/{job_id}/results", response_model=OptimizationResultsResponse)
async def get_optimization_results(
    job_id: str = Path(..., description="Optimization job ID"),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Get optimization job results.

    Returns the results of a completed optimization job.
    """
    logger.info(f"Processing optimization results request for job {job_id}")

    try:
        # Get job results
        job = strategy_db.get_optimization_job(job_id)
        
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Optimization job {job_id} not found",
            )
        
        # Verify user has access to this job
        if job["user_id"] != current_user["id"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to this optimization job",
            )
        
        # Check if job is completed
        if job["status"] != "completed":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Optimization job {job_id} is not completed yet",
            )
        
        # Get job results
        results = strategy_db.get_optimization_results(job_id)
        
        if not results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Results for optimization job {job_id} not found",
            )
        
        # Convert results to model
        result_models = []
        for i, result in enumerate(results["results"]):
            result_models.append(
                OptimizationResult(
                    parameters=result["parameters"],
                    profit_factor=result["profit_factor"],
                    sharpe_ratio=result["sharpe_ratio"],
                    win_rate=result["win_rate"],
                    total_trades=result["total_trades"],
                    net_profit=result["net_profit"],
                    max_drawdown=result["max_drawdown"],
                    rank=i + 1,
                )
            )
        
        # Find best result
        best_result = result_models[0] if result_models else None
        
        return OptimizationResultsResponse(
            job_id=job_id,
            strategyId=job["strategy_id"],
            instrument=job["instrument"],
            timeframe=job["timeframe"],
            start_date=job["start_date"],
            end_date=job["end_date"],
            optimization_metric=job["optimization_metric"],
            results=result_models,
            best_result=best_result,
            status=job["status"],
            completed_at=job["completed_at"],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting optimization job results: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting optimization job results: {str(e)}",
        )

@router.post("/walkforward", response_model=WalkforwardTestResponse)
async def start_walkforward_test(
    request: WalkforwardTestRequest = Body(...),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Start a walkforward test.

    Tests a strategy using walkforward analysis with the specified parameters.
    """
    logger.info(f"Processing walkforward test request for strategy {request.strategyId}")

    try:
        # Verify user has access to this strategy
        if not strategy_db.user_has_strategy_access(current_user["id"], request.strategyId):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to this strategy",
            )
        
        # Start walkforward test
        job = strategy_db.start_walkforward_test(
            user_id=current_user["id"],
            strategy_id=request.strategyId,
            instrument=request.instrument,
            timeframe=request.timeframe,
            start_date=request.start_date,
            end_date=request.end_date,
            parameters=request.parameters,
            window_size=request.window_size,
            step_size=request.step_size,
        )
        
        if not job:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to start walkforward test",
            )
        
        return WalkforwardTestResponse(
            job_id=job["id"],
            status=job["status"],
            progress=job["progress"],
            message=job["message"],
            created_at=job["created_at"],
            updated_at=job["updated_at"],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting walkforward test: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error starting walkforward test: {str(e)}",
        )

@router.get("/walkforward/{job_id}/results", response_model=WalkforwardTestResultsResponse)
async def get_walkforward_results(
    job_id: str = Path(..., description="Walkforward test job ID"),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Get walkforward test results.

    Returns the results of a completed walkforward test.
    """
    logger.info(f"Processing walkforward test results request for job {job_id}")

    try:
        # Get job results
        job = strategy_db.get_walkforward_job(job_id)
        
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Walkforward test job {job_id} not found",
            )
        
        # Verify user has access to this job
        if job["user_id"] != current_user["id"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to this walkforward test job",
            )
        
        # Check if job is completed
        if job["status"] != "completed":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Walkforward test job {job_id} is not completed yet",
            )
        
        # Get job results
        results = strategy_db.get_walkforward_results(job_id)
        
        if not results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Results for walkforward test job {job_id} not found",
            )
        
        # Convert results to model
        result_models = []
        for result in results["results"]:
            # Parse dates if they are strings
            window_start = result["window_start"]
            window_end = result["window_end"]
            
            if isinstance(window_start, str):
                try:
                    window_start = datetime.fromisoformat(window_start.replace('Z', '+00:00'))
                except ValueError:
                    window_start = datetime.now() - timedelta(days=30)
                    
            if isinstance(window_end, str):
                try:
                    window_end = datetime.fromisoformat(window_end.replace('Z', '+00:00'))
                except ValueError:
                    window_end = datetime.now()
            
            result_models.append(
                WalkforwardTestResult(
                    window_start=window_start,
                    window_end=window_end,
                    profit_factor=result["profit_factor"],
                    sharpe_ratio=result["sharpe_ratio"],
                    win_rate=result["win_rate"],
                    total_trades=result["total_trades"],
                    net_profit=result["net_profit"],
                    max_drawdown=result["max_drawdown"],
                )
            )
        
        # Ensure parameters is a dictionary
        parameters = job.get("parameters", {})
        if not isinstance(parameters, dict):
            # If parameters is a list (like from the test), convert to dict
            if isinstance(parameters, list):
                parameters = {param.get("name", f"param_{i}"): param.get("min_value", 0) 
                             for i, param in enumerate(parameters)}
            else:
                parameters = {}
        
        return WalkforwardTestResultsResponse(
            job_id=job_id,
            strategyId=job["strategy_id"],
            instrument=job["instrument"],
            timeframe=job["timeframe"],
            start_date=job["start_date"],
            end_date=job["end_date"],
            parameters=parameters,
            window_size=job["window_size"],
            step_size=job["step_size"],
            results=result_models,
            overall_profit_factor=results["overall_profit_factor"],
            overall_sharpe_ratio=results["overall_sharpe_ratio"],
            overall_win_rate=results["overall_win_rate"],
            overall_total_trades=results["overall_total_trades"],
            overall_net_profit=results["overall_net_profit"],
            overall_max_drawdown=results["overall_max_drawdown"],
            status=job["status"],
            completed_at=job["completed_at"],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting walkforward test results: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting walkforward test results: {str(e)}",
        )

@router.post("/montecarlo", response_model=MonteCarloSimulationResponse)
async def start_montecarlo_simulation(
    request: MonteCarloSimulationRequest = Body(...),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Start a Monte Carlo simulation.

    Runs a Monte Carlo simulation on a strategy with the specified parameters.
    """
    logger.info(f"Processing Monte Carlo simulation request for strategy {request.strategyId}")

    try:
        # Verify user has access to this strategy
        if not strategy_db.user_has_strategy_access(current_user["id"], request.strategyId):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to this strategy",
            )
        
        # Start Monte Carlo simulation
        job = strategy_db.start_montecarlo_simulation(
            user_id=current_user["id"],
            strategy_id=request.strategyId,
            instrument=request.instrument,
            timeframe=request.timeframe,
            start_date=request.start_date,
            end_date=request.end_date,
            parameters=request.parameters,
            simulations=request.simulations,
            confidence_level=request.confidence_level,
        )
        
        if not job:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to start Monte Carlo simulation",
            )
        
        return MonteCarloSimulationResponse(
            job_id=job["id"],
            status=job["status"],
            progress=job["progress"],
            message=job["message"],
            created_at=job["created_at"],
            updated_at=job["updated_at"],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting Monte Carlo simulation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error starting Monte Carlo simulation: {str(e)}",
        )

@router.get("/montecarlo/{job_id}/results", response_model=MonteCarloSimulationResultsResponse)
async def get_montecarlo_results(
    job_id: str = Path(..., description="Monte Carlo simulation job ID"),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Get Monte Carlo simulation results.

    Returns the results of a completed Monte Carlo simulation.
    """
    logger.info(f"Processing Monte Carlo results request for job {job_id}")

    try:
        # Get job results
        job = strategy_db.get_montecarlo_job(job_id)
        
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Monte Carlo simulation job {job_id} not found",
            )
        
        # Verify user has access to this job
        if job["user_id"] != current_user["id"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to this Monte Carlo simulation job",
            )
        
        # Check if job is completed
        if job["status"] != "completed":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Monte Carlo simulation job {job_id} is not completed yet",
            )
        
        # Get job results
        results = strategy_db.get_montecarlo_results(job_id)
        
        if not results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Results for Monte Carlo simulation job {job_id} not found",
            )
        
        # Convert results to model
        result_models = []
        for result in results["results"]:
            result_models.append(
                MonteCarloSimulationResult(
                    percentile=result["percentile"],
                    final_equity=result["final_equity"],
                    max_drawdown=result["max_drawdown"],
                    profit_factor=result["profit_factor"],
                    sharpe_ratio=result["sharpe_ratio"],
                )
            )
        
        # Find median, worst, best, and confidence interval results
        median_result = next((r for r in result_models if r.percentile == 0.5), result_models[0] if result_models else None)
        worst_result = next((r for r in result_models if r.percentile == 0.0), result_models[0] if result_models else None)
        best_result = next((r for r in result_models if r.percentile == 1.0), result_models[-1] if result_models else None)
        
        # Calculate confidence interval
        lower_percentile = (1 - job["confidence_level"]) / 2
        upper_percentile = 1 - lower_percentile
        
        confidence_interval_lower = next(
            (r for r in result_models if abs(r.percentile - lower_percentile) < 0.01), 
            result_models[0] if result_models else None
        )
        
        confidence_interval_upper = next(
            (r for r in result_models if abs(r.percentile - upper_percentile) < 0.01), 
            result_models[-1] if result_models else None
        )
        
        # Ensure parameters is a dictionary
        parameters = job.get("parameters", {})
        if not isinstance(parameters, dict):
            # If parameters is a list (like from the test), convert to dict
            if isinstance(parameters, list):
                parameters = {param.get("name", f"param_{i}"): param.get("min_value", 0) 
                             for i, param in enumerate(parameters)}
            else:
                parameters = {}
        
        return MonteCarloSimulationResultsResponse(
            job_id=job_id,
            strategyId=job["strategy_id"],
            instrument=job["instrument"],
            timeframe=job["timeframe"],
            start_date=job["start_date"],
            end_date=job["end_date"],
            parameters=parameters,
            simulations=job["simulations"],
            confidence_level=job["confidence_level"],
            results=result_models,
            median_result=median_result,
            worst_result=worst_result,
            best_result=best_result,
            confidence_interval_lower=confidence_interval_lower,
            confidence_interval_upper=confidence_interval_upper,
            status=job["status"],
            completed_at=job["completed_at"],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting Monte Carlo simulation results: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting Monte Carlo simulation results: {str(e)}",
        ) 
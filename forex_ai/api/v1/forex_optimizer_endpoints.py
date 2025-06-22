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
from forex_ai.data.storage.optimizer_repository import OptimizerRepository

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/forex-optimizer", tags=["forex-optimizer"])

# Initialize repositories
optimizer_repository = OptimizerRepository()

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
        # Create job parameters
        job_parameters = {
            "strategy_id": request.strategyId,
            "instrument": request.instrument,
            "timeframe": request.timeframe,
            "start_date": request.start_date.isoformat(),
            "end_date": request.end_date.isoformat(),
            "parameters": [param.model_dump() for param in request.parameters],
            "optimization_metric": request.optimization_metric,
            "population_size": request.population_size,
            "generations": request.generations,
            "parallel_jobs": request.parallel_jobs,
        }
        
        # Create optimization job using the repository
        job = await optimizer_repository.create_job(
            user_id=current_user["id"],
            parameters=job_parameters,
        )
        
        if not job:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to start optimization job",
            )
        
        # Convert to dictionary if it's a model
        if hasattr(job, "model_dump"):
            job_dict = job.model_dump()
        else:
            job_dict = job
        
        return OptimizationJobResponse(
            job_id=job_dict["id"],
            status=job_dict["status"],
            progress=job_dict["progress"],
            message=job_dict.get("message", "Job created"),
            created_at=job_dict["created_at"],
            updated_at=job_dict["updated_at"],
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
        # Get job status using the repository
        job = await optimizer_repository.get_job_by_id(job_id)
        
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Optimization job {job_id} not found",
            )
        
        # Verify user has access to this job
        if hasattr(job, "model_dump"):
            job_dict = job.model_dump()
        else:
            job_dict = job
            
        if job_dict["user_id"] != current_user["id"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to this job",
            )
        
        return OptimizationJobResponse(
            job_id=job_dict["id"],
            status=job_dict["status"],
            progress=job_dict.get("progress", 0.0),
            message=str(job_dict.get("message", "")) if job_dict.get("message") is not None else "No message available",
            created_at=job_dict["created_at"],
            updated_at=job_dict["updated_at"],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting optimization status: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting optimization status: {str(e)}",
        )

@router.get("/jobs")
async def get_optimization_jobs(
    status: Optional[str] = Query(None, description="Filter by job status"),
    limit: int = Query(10, description="Number of jobs to return"),
    offset: int = Query(0, description="Offset for pagination"),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Get optimization jobs for a user.

    Returns a list of optimization jobs for the specified user.
    """
    userId = current_user["id"]
    logger.info(f"Processing optimization jobs request for user {userId}")

    try:
        # Get jobs using the repository
        jobs = await optimizer_repository.get_jobs(
            user_id=userId,
            status=status,
            limit=limit,
            offset=offset,
        )
        
        # Convert to dictionaries if they're models
        job_dicts = []
        for job in jobs:
            if hasattr(job, "model_dump"):
                job_dict = job.model_dump()
            else:
                job_dict = job
                
            job_dicts.append({
                "job_id": job_dict["id"],
                "status": job_dict["status"],
                "progress": job_dict["progress"],
                "message": job_dict.get("message", ""),
                "parameters": job_dict.get("parameters", {}),
                "created_at": job_dict["created_at"],
                "updated_at": job_dict["updated_at"],
            })
        
        return {
            "jobs": job_dicts,
            "count": len(job_dicts),
            "timestamp": datetime.now(),
        }
    except Exception as e:
        logger.error(f"Error getting optimization jobs: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting optimization jobs: {str(e)}",
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
        # This is a placeholder and should be implemented with a proper job queue
        job_id = f"walkforward-{datetime.now().timestamp()}"
        return WalkforwardTestResponse(
            job_id=job_id,
            status="pending",
            progress=0.0,
            message="Walkforward test job created",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
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
        # Get job using the repository
        job = await optimizer_repository.get_job_by_id(job_id)
        
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Walkforward test job {job_id} not found",
            )
        
        # Verify user has access to this job
        if hasattr(job, "model_dump"):
            job_dict = job.model_dump()
        else:
            job_dict = job
            
        if job_dict["user_id"] != current_user["id"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to this job",
            )
        
        # Mock results for testing
        results = [
            WalkforwardTestResult(
                window_start=datetime.now() - timedelta(days=90),
                window_end=datetime.now() - timedelta(days=60),
                profit_factor=2.2,
                sharpe_ratio=1.6,
                win_rate=0.65,
                total_trades=40,
                net_profit=500.0,
                max_drawdown=120.0
            ),
            WalkforwardTestResult(
                window_start=datetime.now() - timedelta(days=60),
                window_end=datetime.now() - timedelta(days=30),
                profit_factor=1.9,
                sharpe_ratio=1.4,
                win_rate=0.62,
                total_trades=35,
                net_profit=420.0,
                max_drawdown=150.0
            ),
            WalkforwardTestResult(
                window_start=datetime.now() - timedelta(days=30),
                window_end=datetime.now(),
                profit_factor=2.1,
                sharpe_ratio=1.5,
                win_rate=0.63,
                total_trades=38,
                net_profit=480.0,
                max_drawdown=130.0
            )
        ]
        
        return WalkforwardTestResultsResponse(
            job_id=job_id,
            strategyId="test-strategy",
            instrument="EUR_USD",
            timeframe="H1",
            start_date=datetime.now() - timedelta(days=90),
            end_date=datetime.now(),
            parameters={"param1": 0.5, "param2": 10},
            window_size=30,
            step_size=30,
            results=results,
            overall_profit_factor=2.1,
            overall_sharpe_ratio=1.5,
            overall_win_rate=0.63,
            overall_total_trades=113,
            overall_net_profit=1400.0,
            overall_max_drawdown=150.0,
            status="completed",
            completed_at=datetime.now()
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
        # This is a placeholder and should be implemented with a proper job queue
        job_id = f"montecarlo-{datetime.now().timestamp()}"
        return MonteCarloSimulationResponse(
            job_id=job_id,
            status="pending",
            progress=0.0,
            message="Monte Carlo simulation job created",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
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
        # Get job using the repository
        job = await optimizer_repository.get_job_by_id(job_id)
        
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Monte Carlo job {job_id} not found",
            )
        
        # Verify user has access to this job
        if hasattr(job, "model_dump"):
            job_dict = job.model_dump()
        else:
            job_dict = job
            
        if job_dict["user_id"] != current_user["id"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to this job",
            )
        
        # Mock results for testing
        median_result = MonteCarloSimulationResult(
            percentile=0.5,
            final_equity=12000.0,
            max_drawdown=1500.0,
            profit_factor=2.1,
            sharpe_ratio=1.5
        )
        
        worst_result = MonteCarloSimulationResult(
            percentile=0.05,
            final_equity=9000.0,
            max_drawdown=2500.0,
            profit_factor=1.5,
            sharpe_ratio=0.9
        )
        
        best_result = MonteCarloSimulationResult(
            percentile=0.95,
            final_equity=15000.0,
            max_drawdown=1200.0,
            profit_factor=2.8,
            sharpe_ratio=2.1
        )
        
        confidence_interval_lower = MonteCarloSimulationResult(
            percentile=0.025,
            final_equity=8500.0,
            max_drawdown=2700.0,
            profit_factor=1.3,
            sharpe_ratio=0.8
        )
        
        confidence_interval_upper = MonteCarloSimulationResult(
            percentile=0.975,
            final_equity=16000.0,
            max_drawdown=1000.0,
            profit_factor=3.0,
            sharpe_ratio=2.3
        )
        
        results = []
        for i in range(10):  # Just include 10 sample results
            percentile = (i + 1) / 11.0  # Distribute between 0 and 1
            results.append(
                MonteCarloSimulationResult(
                    percentile=percentile,
                    final_equity=9000.0 + percentile * 7000.0,
                    max_drawdown=2700.0 - percentile * 1700.0,
                    profit_factor=1.3 + percentile * 1.7,
                    sharpe_ratio=0.8 + percentile * 1.5
                )
            )
        
        return MonteCarloSimulationResultsResponse(
            job_id=job_id,
            strategyId="test-strategy",
            instrument="EUR_USD",
            timeframe="H1",
            start_date=datetime.now() - timedelta(days=90),
            end_date=datetime.now(),
            parameters={"param1": 0.5, "param2": 10},
            simulations=1000,
            confidence_level=0.95,
            results=results,
            median_result=median_result,
            worst_result=worst_result,
            best_result=best_result,
            confidence_interval_lower=confidence_interval_lower,
            confidence_interval_upper=confidence_interval_upper,
            status="completed",
            completed_at=datetime.now()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting Monte Carlo simulation results: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting Monte Carlo simulation results: {str(e)}",
        )

@router.get("/optimize/{job_id}/results", response_model=OptimizationResultsResponse)
async def get_optimization_results(
    job_id: str = Path(..., description="Optimization job ID"),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Get optimization results.

    Returns the results of a completed optimization job.
    """
    logger.info(f"Processing optimization results request for job {job_id}")
    
    try:
        # Get job using the repository
        job = await optimizer_repository.get_job_by_id(job_id)
        
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Optimization job {job_id} not found",
            )
        
        # Verify user has access to this job
        if hasattr(job, "model_dump"):
            job_dict = job.model_dump()
        else:
            job_dict = job
            
        if job_dict["user_id"] != current_user["id"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to this job",
            )
        
        # Mock results for testing
        best_result = OptimizationResult(
            parameters={"param1": 0.5, "param2": 10},
            profit_factor=2.5,
            sharpe_ratio=1.8,
            win_rate=0.65,
            total_trades=120,
            net_profit=1500.0,
            max_drawdown=300.0,
            rank=1
        )
        
        results = [
            best_result,
            OptimizationResult(
                parameters={"param1": 0.4, "param2": 12},
                profit_factor=2.2,
                sharpe_ratio=1.6,
                win_rate=0.62,
                total_trades=130,
                net_profit=1300.0,
                max_drawdown=350.0,
                rank=2
            ),
            OptimizationResult(
                parameters={"param1": 0.6, "param2": 8},
                profit_factor=2.0,
                sharpe_ratio=1.5,
                win_rate=0.60,
                total_trades=110,
                net_profit=1200.0,
                max_drawdown=400.0,
                rank=3
            )
        ]
        
        return OptimizationResultsResponse(
            job_id=job_id,
            strategyId="test-strategy",
            instrument="EUR_USD",
            timeframe="H1",
            start_date=datetime.now() - timedelta(days=90),
            end_date=datetime.now(),
            optimization_metric="profit_factor",
            results=results,
            best_result=best_result,
            status="completed",
            completed_at=datetime.now()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting optimization results: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting optimization results: {str(e)}",
        )
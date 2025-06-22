"""
Strategy database operations for Forex AI Trading System.

This module provides functionality to interact with strategy data storage.
"""

import logging
import json
import uuid
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta

from forex_ai.models.strategy_models import (
    Strategy,
    StrategyBase,
    StrategyCreate,
    StrategyUpdate,
    StrategyType,
    TimeFrame,
    RiskProfile,
    ParameterType,
)
from forex_ai.data.storage.supabase_client import SupabaseClient
from forex_ai.exceptions import DatabaseError

# Setup logging
logger = logging.getLogger(__name__)

# In-memory database for strategies (replace with real database in production)
strategies_db = {}

# Initialize Supabase client
supabase_client = SupabaseClient()


# Sample strategies for testing
def initialize_sample_strategies():
    """Initialize sample strategies for testing."""
    sample_strategies = [
        {
            "id": "strat-001",
            "name": "Moving Average Crossover",
            "description": "Simple strategy based on crossing of two moving averages",
            "strategy_type": StrategyType.TREND_FOLLOWING,
            "timeframes": [TimeFrame.H1, TimeFrame.H4],
            "instruments": ["EUR/USD", "GBP/USD"],
            "risk_profile": RiskProfile.MODERATE,
            "is_active": True,
            "parameters": {
                "fast_ma_period": 10,
                "slow_ma_period": 30,
                "ma_type": "SMA",
            },
            "parameter_definitions": [
                {
                    "name": "fast_ma_period",
                    "type": ParameterType.INTEGER,
                    "description": "Period for the fast moving average",
                    "default_value": 10,
                    "min_value": 5,
                    "max_value": 50,
                },
                {
                    "name": "slow_ma_period",
                    "type": ParameterType.INTEGER,
                    "description": "Period for the slow moving average",
                    "default_value": 30,
                    "min_value": 20,
                    "max_value": 200,
                },
                {
                    "name": "ma_type",
                    "type": ParameterType.STRING,
                    "description": "Type of moving average",
                    "default_value": "SMA",
                    "options": ["SMA", "EMA", "WMA"],
                },
            ],
            "source_code": "def on_bar(data, parameters):\n    # Crossover strategy implementation\n    fast_ma = data['close'].rolling(parameters['fast_ma_period']).mean()\n    slow_ma = data['close'].rolling(parameters['slow_ma_period']).mean()\n    \n    if fast_ma[-2] < slow_ma[-2] and fast_ma[-1] >= slow_ma[-1]:\n        return 'BUY'\n    elif fast_ma[-2] > slow_ma[-2] and fast_ma[-1] <= slow_ma[-1]:\n        return 'SELL'\n    \n    return None",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "performance_metrics": {
                "win_rate": 0.62,
                "profit_factor": 1.75,
                "max_drawdown": 0.15,
                "avg_trade_duration": "4h 30m",
                "total_trades": 145,
            },
            "backtest_results": {
                "net_profit": 2450.75,
                "win_count": 90,
                "loss_count": 55,
                "trade_details": [],
            },
        },
        {
            "id": "strat-002",
            "name": "RSI Reversals",
            "description": "Strategy based on RSI indicator's overbought and oversold levels",
            "strategy_type": StrategyType.MEAN_REVERSION,
            "timeframes": [TimeFrame.M15, TimeFrame.H1],
            "instruments": ["EUR/USD", "USD/JPY", "GBP/USD"],
            "risk_profile": RiskProfile.AGGRESSIVE,
            "is_active": True,
            "parameters": {
                "rsi_period": 14,
                "overbought_level": 70,
                "oversold_level": 30,
            },
            "parameter_definitions": [
                {
                    "name": "rsi_period",
                    "type": ParameterType.INTEGER,
                    "description": "Period for RSI calculation",
                    "default_value": 14,
                    "min_value": 5,
                    "max_value": 30,
                },
                {
                    "name": "overbought_level",
                    "type": ParameterType.INTEGER,
                    "description": "Level to consider market overbought",
                    "default_value": 70,
                    "min_value": 60,
                    "max_value": 90,
                },
                {
                    "name": "oversold_level",
                    "type": ParameterType.INTEGER,
                    "description": "Level to consider market oversold",
                    "default_value": 30,
                    "min_value": 10,
                    "max_value": 40,
                },
            ],
            "source_code": "def on_bar(data, parameters):\n    # RSI strategy implementation\n    rsi = calculate_rsi(data['close'], parameters['rsi_period'])\n    \n    if rsi[-1] < parameters['oversold_level']:\n        return 'BUY'\n    elif rsi[-1] > parameters['overbought_level']:\n        return 'SELL'\n    \n    return None",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "performance_metrics": {
                "win_rate": 0.58,
                "profit_factor": 1.45,
                "max_drawdown": 0.22,
                "avg_trade_duration": "2h 15m",
                "total_trades": 210,
            },
            "backtest_results": {
                "net_profit": 1850.25,
                "win_count": 122,
                "loss_count": 88,
                "trade_details": [],
            },
        },
    ]

    # Add sample strategies to in-memory DB
    for strategy_data in sample_strategies:
        strategy = Strategy(**strategy_data)
        strategies_db[strategy.id] = strategy

    logger.info(f"Initialized {len(sample_strategies)} sample strategies")


def initialize_db():
    """Initialize the strategy database."""
    logger.info("Initializing strategy database")
    # In a real app, this would connect to a real database
    # For the mock version, we just initialize sample data
    initialize_sample_strategies()


# Don't automatically initialize at import time since main.py will call the function
# Comment out the automatic initialization
# initialize_sample_strategies()


def get_all_strategies() -> List[Strategy]:
    """Get all strategies from the database."""
    logger.info("Retrieving all strategies")
    return list(strategies_db.values())


def get_strategy_by_id(strategy_id: str) -> Optional[Strategy]:
    """Get a strategy by its ID."""
    logger.info(f"Retrieving strategy with ID: {strategy_id}")
    return strategies_db.get(strategy_id)


def create_strategy(strategy_data: StrategyCreate) -> Strategy:
    """Create a new strategy in the database."""
    logger.info(f"Creating new strategy: {strategy_data.name}")

    # Generate ID
    strategy_id = f"strat-{str(uuid.uuid4())[:8]}"

    # Create strategy object
    strategy = Strategy(
        id=strategy_id,
        name=strategy_data.name,
        description=strategy_data.description,
        strategy_type=strategy_data.strategy_type,
        timeframes=strategy_data.timeframes,
        instruments=strategy_data.instruments,
        risk_profile=strategy_data.risk_profile,
        is_active=strategy_data.is_active,
        parameters=strategy_data.parameters,
        parameter_definitions=strategy_data.parameter_definitions,
        source_code=strategy_data.source_code,
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
        performance_metrics={},
        backtest_results={},
    )

    # Add to database
    strategies_db[strategy_id] = strategy

    return strategy


def update_strategy(
    strategy_id: str, strategy_data: StrategyUpdate
) -> Optional[Strategy]:
    """Update an existing strategy in the database."""
    logger.info(f"Updating strategy with ID: {strategy_id}")

    # Check if strategy exists
    if strategy_id not in strategies_db:
        logger.warning(f"Strategy with ID {strategy_id} not found")
        return None

    # Get current strategy
    current_strategy = strategies_db[strategy_id]

    # Update fields
    update_data = strategy_data.dict(exclude_unset=True)

    for field, value in update_data.items():
        setattr(current_strategy, field, value)

    # Update timestamp
    current_strategy.updated_at = datetime.now().isoformat()

    # Save to database
    strategies_db[strategy_id] = current_strategy

    return current_strategy


def delete_strategy(strategy_id: str) -> bool:
    """Delete a strategy from the database."""
    logger.info(f"Deleting strategy with ID: {strategy_id}")

    if strategy_id in strategies_db:
        del strategies_db[strategy_id]
        return True

    return False


def get_strategies_by_type(strategy_type: StrategyType) -> List[Strategy]:
    """Get strategies of a specific type."""
    logger.info(f"Retrieving strategies of type: {strategy_type}")

    return [s for s in strategies_db.values() if s.strategy_type == strategy_type]


def get_strategies_for_instrument(instrument: str) -> List[Strategy]:
    """Get strategies compatible with a specific instrument."""
    logger.info(f"Retrieving strategies for instrument: {instrument}")

    return [s for s in strategies_db.values() if instrument in s.instruments]


def get_strategies_for_timeframe(timeframe: TimeFrame) -> List[Strategy]:
    """Get strategies compatible with a specific timeframe."""
    logger.info(f"Retrieving strategies for timeframe: {timeframe}")

    return [s for s in strategies_db.values() if timeframe in s.timeframes]


def user_has_strategy_access(user_id: str, strategy_id: str) -> bool:
    """
    Check if a user has access to a strategy.

    Args:
        user_id: User ID
        strategy_id: Strategy ID

    Returns:
        True if user has access, False otherwise
    """
    try:
        logger.info(f"Checking if user {user_id} has access to strategy {strategy_id}")
        
        # Query the database
        result = supabase_client.client.table("strategies") \
            .select("id") \
            .eq("id", strategy_id) \
            .eq("user_id", user_id) \
            .execute()
        
        return bool(result.data)
    except Exception as e:
        logger.error(f"Error checking strategy access: {str(e)}", exc_info=True)
        raise DatabaseError(f"Error checking strategy access: {str(e)}")


def start_optimization_job(
    user_id: str,
    strategy_id: str,
    instrument: str,
    timeframe: str,
    start_date: datetime,
    end_date: datetime,
    parameters: List[Dict[str, Any]],
    optimization_metric: str = "profit_factor",
    population_size: int = 50,
    generations: int = 10,
    parallel_jobs: int = 4,
) -> Dict[str, Any]:
    """
    Start a strategy optimization job.

    Args:
        user_id: User ID
        strategy_id: Strategy ID to optimize
        instrument: Instrument to optimize for
        timeframe: Timeframe to optimize for
        start_date: Start date for optimization
        end_date: End date for optimization
        parameters: Parameters to optimize
        optimization_metric: Metric to optimize for
        population_size: Population size for genetic algorithm
        generations: Number of generations for genetic algorithm
        parallel_jobs: Number of parallel jobs to run

    Returns:
        Dictionary containing job information
    """
    try:
        logger.info(f"Starting optimization job for strategy {strategy_id}")
        
        # Create job ID
        job_id = str(uuid.uuid4())
        
        # Prepare job data
        job_data = {
            "id": job_id,
            "user_id": user_id,
            "strategy_id": strategy_id,
            "instrument": instrument,
            "timeframe": timeframe,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "parameters": parameters,
            "optimization_metric": optimization_metric,
            "population_size": population_size,
            "generations": generations,
            "parallel_jobs": parallel_jobs,
            "status": "pending",
            "progress": 0.0,
            "message": "Job created",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        
        # Insert job into database
        result = supabase_client.client.table("optimization_jobs") \
            .insert(job_data) \
            .execute()
        
        if not result.data:
            logger.error(f"Failed to create optimization job for strategy {strategy_id}")
            raise DatabaseError(f"Failed to create optimization job for strategy {strategy_id}")
        
        # TODO: Start actual optimization process (e.g., using a background task)
        # For now, simulate job start
        
        return result.data[0]
    except Exception as e:
        logger.error(f"Error starting optimization job: {str(e)}", exc_info=True)
        raise DatabaseError(f"Error starting optimization job: {str(e)}")


def get_optimization_job(job_id: str) -> Dict[str, Any]:
    """
    Get optimization job information.

    Args:
        job_id: Job ID

    Returns:
        Dictionary containing job information
    """
    try:
        logger.info(f"Getting optimization job {job_id}")
        
        # Query the database
        result = supabase_client.client.table("optimization_jobs") \
            .select("*") \
            .eq("id", job_id) \
            .execute()
        
        if not result.data:
            logger.info(f"Optimization job {job_id} not found")
            return {}
        
        return result.data[0]
    except Exception as e:
        logger.error(f"Error getting optimization job: {str(e)}", exc_info=True)
        raise DatabaseError(f"Error getting optimization job: {str(e)}")


def get_optimization_results(job_id: str) -> Dict[str, Any]:
    """
    Get optimization job results.

    Args:
        job_id: Job ID

    Returns:
        Dictionary containing job results
    """
    try:
        logger.info(f"Getting optimization results for job {job_id}")
        
        # Query the database
        result = supabase_client.client.table("optimization_results") \
            .select("*") \
            .eq("job_id", job_id) \
            .execute()
        
        if not result.data:
            logger.info(f"No optimization results found for job {job_id}")
            return {}
        
        # Get job information
        job = get_optimization_job(job_id)
        
        # Combine job info with results
        return {
            "job_id": job_id,
            "strategy_id": job.get("strategy_id"),
            "instrument": job.get("instrument"),
            "timeframe": job.get("timeframe"),
            "start_date": job.get("start_date"),
            "end_date": job.get("end_date"),
            "optimization_metric": job.get("optimization_metric"),
            "results": result.data,
        }
    except Exception as e:
        logger.error(f"Error getting optimization results: {str(e)}", exc_info=True)
        raise DatabaseError(f"Error getting optimization results: {str(e)}")


def start_walkforward_test(
    user_id: str,
    strategy_id: str,
    instrument: str,
    timeframe: str,
    start_date: datetime,
    end_date: datetime,
    parameters: Dict[str, Union[float, int]],
    window_size: int = 90,
    step_size: int = 30,
) -> Dict[str, Any]:
    """
    Start a walkforward test.

    Args:
        user_id: User ID
        strategy_id: Strategy ID to test
        instrument: Instrument to test on
        timeframe: Timeframe to test on
        start_date: Start date for testing
        end_date: End date for testing
        parameters: Strategy parameters
        window_size: Window size in days
        step_size: Step size in days

    Returns:
        Dictionary containing job information
    """
    try:
        logger.info(f"Starting walkforward test for strategy {strategy_id}")
        
        # Create job ID
        job_id = str(uuid.uuid4())
        
        # Prepare job data
        job_data = {
            "id": job_id,
            "user_id": user_id,
            "strategy_id": strategy_id,
            "instrument": instrument,
            "timeframe": timeframe,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "parameters": parameters,
            "window_size": window_size,
            "step_size": step_size,
            "status": "pending",
            "progress": 0.0,
            "message": "Job created",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        
        # Insert job into database
        result = supabase_client.client.table("walkforward_jobs") \
            .insert(job_data) \
            .execute()
        
        if not result.data:
            logger.error(f"Failed to create walkforward test job for strategy {strategy_id}")
            raise DatabaseError(f"Failed to create walkforward test job for strategy {strategy_id}")
        
        # TODO: Start actual walkforward test process (e.g., using a background task)
        # For now, simulate job start
        
        return result.data[0]
    except Exception as e:
        logger.error(f"Error starting walkforward test: {str(e)}", exc_info=True)
        raise DatabaseError(f"Error starting walkforward test: {str(e)}")


def get_walkforward_job(job_id: str) -> Dict[str, Any]:
    """
    Get walkforward test job information.

    Args:
        job_id: Job ID

    Returns:
        Dictionary containing job information
    """
    try:
        logger.info(f"Getting walkforward test job {job_id}")
        
        # Query the database
        result = supabase_client.client.table("walkforward_jobs") \
            .select("*") \
            .eq("id", job_id) \
            .execute()
        
        if not result.data:
            logger.info(f"Walkforward test job {job_id} not found")
            return {}
        
        return result.data[0]
    except Exception as e:
        logger.error(f"Error getting walkforward test job: {str(e)}", exc_info=True)
        raise DatabaseError(f"Error getting walkforward test job: {str(e)}")


def get_walkforward_results(job_id: str) -> Dict[str, Any]:
    """
    Get walkforward test job results.

    Args:
        job_id: Job ID

    Returns:
        Dictionary containing job results
    """
    try:
        logger.info(f"Getting walkforward test results for job {job_id}")
        
        # Query the database
        result = supabase_client.client.table("walkforward_results") \
            .select("*") \
            .eq("job_id", job_id) \
            .execute()
        
        if not result.data:
            logger.info(f"No walkforward test results found for job {job_id}")
            return {}
        
        # Calculate overall metrics
        results = result.data
        overall_profit_factor = sum(r.get("profit_factor", 0) for r in results) / len(results)
        overall_sharpe_ratio = sum(r.get("sharpe_ratio", 0) for r in results) / len(results)
        overall_win_rate = sum(r.get("win_rate", 0) for r in results) / len(results)
        overall_total_trades = sum(r.get("total_trades", 0) for r in results)
        overall_net_profit = sum(r.get("net_profit", 0) for r in results)
        overall_max_drawdown = max(r.get("max_drawdown", 0) for r in results)
        
        return {
            "results": results,
            "overall_profit_factor": overall_profit_factor,
            "overall_sharpe_ratio": overall_sharpe_ratio,
            "overall_win_rate": overall_win_rate,
            "overall_total_trades": overall_total_trades,
            "overall_net_profit": overall_net_profit,
            "overall_max_drawdown": overall_max_drawdown,
        }
    except Exception as e:
        logger.error(f"Error getting walkforward test results: {str(e)}", exc_info=True)
        raise DatabaseError(f"Error getting walkforward test results: {str(e)}")


def start_montecarlo_simulation(
    user_id: str,
    strategy_id: str,
    instrument: str,
    timeframe: str,
    start_date: datetime,
    end_date: datetime,
    parameters: Dict[str, Union[float, int]],
    simulations: int = 1000,
    confidence_level: float = 0.95,
) -> Dict[str, Any]:
    """
    Start a Monte Carlo simulation.

    Args:
        user_id: User ID
        strategy_id: Strategy ID to simulate
        instrument: Instrument to simulate on
        timeframe: Timeframe to simulate on
        start_date: Start date for simulation
        end_date: End date for simulation
        parameters: Strategy parameters
        simulations: Number of simulations to run
        confidence_level: Confidence level for results

    Returns:
        Dictionary containing job information
    """
    try:
        logger.info(f"Starting Monte Carlo simulation for strategy {strategy_id}")
        
        # Create job ID
        job_id = str(uuid.uuid4())
        
        # Prepare job data
        job_data = {
            "id": job_id,
            "user_id": user_id,
            "strategy_id": strategy_id,
            "instrument": instrument,
            "timeframe": timeframe,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "parameters": parameters,
            "simulations": simulations,
            "confidence_level": confidence_level,
            "status": "pending",
            "progress": 0.0,
            "message": "Job created",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        
        # Insert job into database
        result = supabase_client.client.table("montecarlo_jobs") \
            .insert(job_data) \
            .execute()
        
        if not result.data:
            logger.error(f"Failed to create Monte Carlo simulation job for strategy {strategy_id}")
            raise DatabaseError(f"Failed to create Monte Carlo simulation job for strategy {strategy_id}")
        
        # TODO: Start actual Monte Carlo simulation process (e.g., using a background task)
        # For now, simulate job start
        
        return result.data[0]
    except Exception as e:
        logger.error(f"Error starting Monte Carlo simulation: {str(e)}", exc_info=True)
        raise DatabaseError(f"Error starting Monte Carlo simulation: {str(e)}")


def get_montecarlo_job(job_id: str) -> Dict[str, Any]:
    """
    Get Monte Carlo simulation job information.

    Args:
        job_id: Job ID

    Returns:
        Dictionary containing job information
    """
    try:
        logger.info(f"Getting Monte Carlo simulation job {job_id}")
        
        # Query the database
        result = supabase_client.client.table("montecarlo_jobs") \
            .select("*") \
            .eq("id", job_id) \
            .execute()
        
        if not result.data:
            logger.info(f"Monte Carlo simulation job {job_id} not found")
            return {}
        
        return result.data[0]
    except Exception as e:
        logger.error(f"Error getting Monte Carlo simulation job: {str(e)}", exc_info=True)
        raise DatabaseError(f"Error getting Monte Carlo simulation job: {str(e)}")


def get_montecarlo_results(job_id: str) -> Dict[str, Any]:
    """
    Get Monte Carlo simulation job results.

    Args:
        job_id: Job ID

    Returns:
        Dictionary containing job results
    """
    try:
        logger.info(f"Getting Monte Carlo simulation results for job {job_id}")
        
        # Query the database
        result = supabase_client.client.table("montecarlo_results") \
            .select("*") \
            .eq("job_id", job_id) \
            .execute()
        
        if not result.data:
            logger.info(f"No Monte Carlo simulation results found for job {job_id}")
            return {}
        
        return {
            "results": result.data,
        }
    except Exception as e:
        logger.error(f"Error getting Monte Carlo simulation results: {str(e)}", exc_info=True)
        raise DatabaseError(f"Error getting Monte Carlo simulation results: {str(e)}")

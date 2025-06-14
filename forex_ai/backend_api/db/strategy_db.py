"""
Strategy database operations for Forex AI Trading System.

This module provides functionality to interact with strategy data storage.
"""

import logging
import json
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

from app.models.strategy_models import (
    Strategy,
    StrategyBase,
    StrategyCreate,
    StrategyUpdate,
    StrategyType,
    TimeFrame,
    RiskProfile,
    ParameterType,
)

# Setup logging
logger = logging.getLogger(__name__)

# In-memory database for strategies (replace with real database in production)
strategies_db = {}


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

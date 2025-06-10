"""
Strategy Optimization Module for the AI Forex Trading System.

This module provides functionality for optimizing trading strategy parameters,
evaluating performance across different market conditions, and adapting
strategies to changing market dynamics.
"""

from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import random
import uuid
from tqdm import tqdm
import concurrent.futures

from .pine_script import PineScriptStrategy, PineScriptExecutor
from forex_ai.utils.logging import get_logger
from forex_ai.exceptions import OptimizationError

logger = get_logger(__name__)


@dataclass
class ParameterSpace:
    """Definition of a parameter's optimization space."""

    name: str
    min_value: Union[int, float]
    max_value: Union[int, float]
    step: Union[int, float] = 1
    is_categorical: bool = False
    categories: List[Any] = field(default_factory=list)

    def __post_init__(self):
        """Validate parameter space configuration."""
        if self.is_categorical and not self.categories:
            raise ValueError(
                f"Parameter {self.name} is marked as categorical but no categories provided"
            )

        if not self.is_categorical and (self.min_value > self.max_value):
            raise ValueError(f"Parameter {self.name} has min_value > max_value")


@dataclass
class OptimizationResult:
    """Results of a strategy optimization run."""

    strategy_id: str
    strategy_name: str
    best_parameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    all_results: List[Dict[str, Any]]
    optimization_time: float
    created_at: datetime = field(default_factory=datetime.now)
    unique_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """Save optimization results to a JSON file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert datetime to string for JSON serialization
        results_dict = {
            "strategy_id": self.strategy_id,
            "strategy_name": self.strategy_name,
            "best_parameters": self.best_parameters,
            "performance_metrics": self.performance_metrics,
            "all_results": self.all_results,
            "optimization_time": self.optimization_time,
            "created_at": self.created_at.isoformat(),
            "unique_id": self.unique_id,
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, indent=2)


class StrategyOptimizer:
    """Optimizer for trading strategy parameters."""

    def __init__(
        self,
        data: pd.DataFrame,
        train_test_split: float = 0.7,
        performance_metric: str = "sharpe_ratio",
        max_iterations: int = 100,
        random_seed: int = 42,
        parallel: bool = True,
        max_workers: int = None,
    ):
        """
        Initialize the strategy optimizer.

        Args:
            data: DataFrame with OHLCV data for optimization
            train_test_split: Ratio for splitting data into training/testing sets
            performance_metric: Metric to optimize ('sharpe_ratio', 'profit_factor', etc.)
            max_iterations: Maximum number of iterations for optimization
            random_seed: Random seed for reproducibility
            parallel: Whether to use parallel processing
            max_workers: Maximum number of worker processes (None = auto)
        """
        self.data = data
        self.train_test_split = train_test_split
        self.performance_metric = performance_metric
        self.max_iterations = max_iterations
        self.random_seed = random_seed
        self.parallel = parallel
        self.max_workers = max_workers

        # Split data into training and testing sets
        split_idx = int(len(data) * train_test_split)
        self.train_data = data.iloc[:split_idx]
        self.test_data = data.iloc[split_idx:]

        # Set random seed
        random.seed(random_seed)
        np.random.seed(random_seed)

    def optimize_grid_search(
        self, strategy: PineScriptStrategy, parameter_spaces: List[ParameterSpace]
    ) -> OptimizationResult:
        """
        Optimize strategy parameters using grid search.

        Args:
            strategy: PineScriptStrategy to optimize
            parameter_spaces: List of parameter spaces to search

        Returns:
            OptimizationResult containing the best parameters and performance metrics

        Raises:
            OptimizationError: If optimization fails
        """
        try:
            start_time = datetime.now()

            # Generate all parameter combinations
            param_combinations = self._generate_parameter_combinations(parameter_spaces)
            logger.info(
                f"Generated {len(param_combinations)} parameter combinations for grid search"
            )

            # Limit number of combinations if too large
            if len(param_combinations) > self.max_iterations:
                logger.warning(
                    f"Too many parameter combinations ({len(param_combinations)}), "
                    f"limiting to {self.max_iterations}"
                )
                param_combinations = random.sample(
                    param_combinations, self.max_iterations
                )

            # Evaluate all combinations
            results = []

            if self.parallel:
                # Parallel evaluation
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=self.max_workers
                ) as executor:
                    future_to_params = {
                        executor.submit(
                            self._evaluate_parameters, strategy, params
                        ): params
                        for params in param_combinations
                    }

                    # Show progress bar
                    for future in tqdm(
                        concurrent.futures.as_completed(future_to_params),
                        total=len(param_combinations),
                        desc="Optimizing strategy",
                    ):
                        params = future_to_params[future]
                        try:
                            metrics = future.result()
                            results.append({"parameters": params, "metrics": metrics})
                        except Exception as e:
                            logger.error(
                                f"Error evaluating parameters {params}: {str(e)}"
                            )
            else:
                # Sequential evaluation
                for params in tqdm(param_combinations, desc="Optimizing strategy"):
                    try:
                        metrics = self._evaluate_parameters(strategy, params)
                        results.append({"parameters": params, "metrics": metrics})
                    except Exception as e:
                        logger.error(f"Error evaluating parameters {params}: {str(e)}")

            # Find best parameters
            best_result = max(
                results,
                key=lambda r: r["metrics"].get(self.performance_metric, float("-inf")),
            )

            # Calculate optimization time
            optimization_time = (datetime.now() - start_time).total_seconds()

            # Create optimization result
            opt_result = OptimizationResult(
                strategy_id=strategy.unique_id,
                strategy_name=strategy.name,
                best_parameters=best_result["parameters"],
                performance_metrics=best_result["metrics"],
                all_results=results,
                optimization_time=optimization_time,
            )

            return opt_result
        except Exception as e:
            logger.error(f"Error in grid search optimization: {str(e)}")
            raise OptimizationError(f"Failed to optimize strategy: {str(e)}") from e

    def optimize_random_search(
        self,
        strategy: PineScriptStrategy,
        parameter_spaces: List[ParameterSpace],
        num_iterations: int = None,
    ) -> OptimizationResult:
        """
        Optimize strategy parameters using random search.

        Args:
            strategy: PineScriptStrategy to optimize
            parameter_spaces: List of parameter spaces to search
            num_iterations: Number of random parameter sets to evaluate

        Returns:
            OptimizationResult containing the best parameters and performance metrics

        Raises:
            OptimizationError: If optimization fails
        """
        try:
            start_time = datetime.now()

            # Use default max_iterations if not specified
            if num_iterations is None:
                num_iterations = self.max_iterations

            # Generate random parameter combinations
            param_combinations = []
            for _ in range(num_iterations):
                params = {}
                for param_space in parameter_spaces:
                    if param_space.is_categorical:
                        params[param_space.name] = random.choice(param_space.categories)
                    else:
                        # Handle both integer and float parameters
                        if isinstance(param_space.step, int) and isinstance(
                            param_space.min_value, int
                        ):
                            # Integer parameter
                            params[param_space.name] = random.randint(
                                param_space.min_value, param_space.max_value
                            )
                        else:
                            # Float parameter
                            params[param_space.name] = random.uniform(
                                param_space.min_value, param_space.max_value
                            )
                param_combinations.append(params)

            logger.info(
                f"Generated {len(param_combinations)} random parameter combinations"
            )

            # Evaluate all combinations
            results = []

            if self.parallel:
                # Parallel evaluation
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=self.max_workers
                ) as executor:
                    future_to_params = {
                        executor.submit(
                            self._evaluate_parameters, strategy, params
                        ): params
                        for params in param_combinations
                    }

                    # Show progress bar
                    for future in tqdm(
                        concurrent.futures.as_completed(future_to_params),
                        total=len(param_combinations),
                        desc="Optimizing strategy",
                    ):
                        params = future_to_params[future]
                        try:
                            metrics = future.result()
                            results.append({"parameters": params, "metrics": metrics})
                        except Exception as e:
                            logger.error(
                                f"Error evaluating parameters {params}: {str(e)}"
                            )
            else:
                # Sequential evaluation
                for params in tqdm(param_combinations, desc="Optimizing strategy"):
                    try:
                        metrics = self._evaluate_parameters(strategy, params)
                        results.append({"parameters": params, "metrics": metrics})
                    except Exception as e:
                        logger.error(f"Error evaluating parameters {params}: {str(e)}")

            # Find best parameters
            best_result = max(
                results,
                key=lambda r: r["metrics"].get(self.performance_metric, float("-inf")),
            )

            # Calculate optimization time
            optimization_time = (datetime.now() - start_time).total_seconds()

            # Create optimization result
            opt_result = OptimizationResult(
                strategy_id=strategy.unique_id,
                strategy_name=strategy.name,
                best_parameters=best_result["parameters"],
                performance_metrics=best_result["metrics"],
                all_results=results,
                optimization_time=optimization_time,
            )

            return opt_result
        except Exception as e:
            logger.error(f"Error in random search optimization: {str(e)}")
            raise OptimizationError(f"Failed to optimize strategy: {str(e)}") from e

    def _generate_parameter_combinations(
        self, parameter_spaces: List[ParameterSpace]
    ) -> List[Dict[str, Any]]:
        """
        Generate all possible parameter combinations for grid search.

        Args:
            parameter_spaces: List of parameter spaces

        Returns:
            List of parameter dictionaries
        """
        # Initialize with empty combination
        combinations = [{}]

        # Generate combinations for each parameter
        for param_space in parameter_spaces:
            new_combinations = []

            if param_space.is_categorical:
                # Categorical parameter
                for combo in combinations:
                    for category in param_space.categories:
                        new_combo = combo.copy()
                        new_combo[param_space.name] = category
                        new_combinations.append(new_combo)
            else:
                # Numerical parameter
                values = []
                current = param_space.min_value

                # Generate values with step
                while current <= param_space.max_value:
                    values.append(current)
                    current += param_space.step

                # Create combinations
                for combo in combinations:
                    for value in values:
                        new_combo = combo.copy()
                        new_combo[param_space.name] = value
                        new_combinations.append(new_combo)

            combinations = new_combinations

        return combinations

    def _evaluate_parameters(
        self, strategy: PineScriptStrategy, parameters: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Evaluate a set of parameters on the training data.

        Args:
            strategy: PineScriptStrategy to evaluate
            parameters: Parameter values to test

        Returns:
            Dictionary of performance metrics
        """
        try:
            # Create executor with parameters
            executor = PineScriptExecutor(strategy, parameters)

            # Execute strategy on training data
            execution_results = executor.execute(self.train_data)

            # Extract and calculate performance metrics
            metrics = self._calculate_performance_metrics(execution_results)

            return metrics
        except Exception as e:
            logger.error(f"Error evaluating parameters {parameters}: {str(e)}")
            raise OptimizationError(f"Failed to evaluate parameters: {str(e)}") from e

    def _calculate_performance_metrics(
        self, execution_results: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate performance metrics from strategy execution results.

        Args:
            execution_results: Results from strategy execution

        Returns:
            Dictionary of performance metrics
        """
        # Extract signals and performance data
        signals = execution_results.get("entry_signals", [])

        # This is a simplified placeholder implementation
        # In a real system, you'd calculate these metrics from actual backtest results

        # Extract existing metrics if available
        performance = execution_results.get("performance", {})

        # Set default metrics
        metrics = {
            "signal_count": len(signals),
            "win_rate": performance.get("win_rate", 0.5),
            "avg_profit": performance.get("avg_profit", 0.0),
            "profit_factor": 1.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "recovery_factor": 0.0,
            "expectancy": 0.0,
        }

        # Simulate some randomness for testing
        # In a real implementation, these would be calculated from actual backtest results
        random_factor = np.random.normal(1.0, 0.1)  # Add some noise
        metrics["win_rate"] = min(0.95, max(0.05, metrics["win_rate"] * random_factor))
        metrics["avg_profit"] = metrics["avg_profit"] * random_factor

        # Calculate derived metrics
        if metrics["win_rate"] > 0:
            # Simulate profit factor (winners/losers)
            avg_win = metrics["avg_profit"] * (1 + np.random.normal(0, 0.2))
            avg_loss = metrics["avg_profit"] * 0.7 * (1 + np.random.normal(0, 0.2))

            win_count = int(metrics["signal_count"] * metrics["win_rate"])
            loss_count = metrics["signal_count"] - win_count

            total_wins = win_count * avg_win
            total_losses = loss_count * avg_loss

            metrics["profit_factor"] = (
                abs(total_wins / total_losses) if total_losses != 0 else total_wins
            )

            # Simulate Sharpe ratio
            metrics["sharpe_ratio"] = (metrics["win_rate"] - 0.5) * 3 * random_factor

            # Simulate max drawdown
            metrics["max_drawdown"] = 0.1 + 0.2 * (1 - metrics["win_rate"])

            # Recovery factor = net profit / max drawdown
            metrics["recovery_factor"] = (
                (metrics["avg_profit"] * metrics["signal_count"])
                / metrics["max_drawdown"]
                if metrics["max_drawdown"] > 0
                else 0
            )

            # Expectancy = (Win Rate × Average Win) - (Loss Rate × Average Loss)
            metrics["expectancy"] = (metrics["win_rate"] * avg_win) - (
                (1 - metrics["win_rate"]) * avg_loss
            )

        return metrics


class MarketConditionAnalyzer:
    """Analyzer for detecting and categorizing market conditions."""

    def __init__(self, data: pd.DataFrame):
        """
        Initialize the market condition analyzer.

        Args:
            data: DataFrame with OHLCV data
        """
        self.data = data

    def detect_conditions(self, window_size: int = 20) -> Dict[str, Any]:
        """
        Detect market conditions in the provided data.

        Args:
            window_size: Window size for calculating metrics

        Returns:
            Dictionary containing detected market conditions and characteristics
        """
        try:
            # Calculate basic metrics
            returns = self.data["close"].pct_change().dropna()
            volatility = returns.rolling(window=window_size).std().dropna() * np.sqrt(
                252
            )  # Annualized

            # Calculate trend metrics
            sma20 = self.data["close"].rolling(window=20).mean()
            sma50 = self.data["close"].rolling(window=50).mean()
            sma200 = self.data["close"].rolling(window=200).mean()

            # Identify trend direction
            uptrend = (sma20 > sma50) & (sma50 > sma200)
            downtrend = (sma20 < sma50) & (sma50 < sma200)
            sideways = ~(uptrend | downtrend)

            # Calculate average daily range
            adr = (
                (self.data["high"] - self.data["low"])
                .rolling(window=window_size)
                .mean()
            )

            # Calculate momentum
            roc = (self.data["close"] / self.data["close"].shift(window_size) - 1) * 100

            # Determine volatility regime
            recent_volatility = volatility.iloc[-1] if not volatility.empty else 0
            volatility_regime = (
                "high"
                if recent_volatility > 0.2
                else "low" if recent_volatility < 0.1 else "medium"
            )

            # Determine trend regime
            if uptrend.iloc[-1]:
                trend_regime = "uptrend"
            elif downtrend.iloc[-1]:
                trend_regime = "downtrend"
            else:
                trend_regime = "sideways"

            # Determine momentum regime
            recent_momentum = roc.iloc[-1] if not roc.empty else 0
            momentum_regime = (
                "bullish"
                if recent_momentum > 5
                else "bearish" if recent_momentum < -5 else "neutral"
            )

            return {
                "volatility_regime": volatility_regime,
                "trend_regime": trend_regime,
                "momentum_regime": momentum_regime,
                "metrics": {
                    "volatility": recent_volatility,
                    "adr": adr.iloc[-1] if not adr.empty else 0,
                    "momentum": recent_momentum,
                },
                "market_condition": f"{trend_regime}_{volatility_regime}_{momentum_regime}",
            }
        except Exception as e:
            logger.error(f"Error detecting market conditions: {str(e)}")
            return {
                "volatility_regime": "unknown",
                "trend_regime": "unknown",
                "momentum_regime": "unknown",
                "metrics": {},
                "market_condition": "unknown",
            }

    def segment_data_by_condition(
        self, condition_key: str = "trend_regime"
    ) -> Dict[str, pd.DataFrame]:
        """
        Segment data into different market condition periods.

        Args:
            condition_key: Market condition key to segment by

        Returns:
            Dictionary mapping condition values to DataFrames
        """
        try:
            segments = {}

            # Calculate required indicators based on condition key
            if condition_key == "trend_regime":
                sma20 = self.data["close"].rolling(window=20).mean()
                sma50 = self.data["close"].rolling(window=50).mean()
                sma200 = self.data["close"].rolling(window=200).mean()

                uptrend = (sma20 > sma50) & (sma50 > sma200)
                downtrend = (sma20 < sma50) & (sma50 < sma200)
                sideways = ~(uptrend | downtrend)

                # Create segments
                segments["uptrend"] = self.data[uptrend]
                segments["downtrend"] = self.data[downtrend]
                segments["sideways"] = self.data[sideways]

            elif condition_key == "volatility_regime":
                returns = self.data["close"].pct_change().dropna()
                volatility = returns.rolling(window=20).std() * np.sqrt(
                    252
                )  # Annualized

                high_vol = volatility > 0.2
                low_vol = volatility < 0.1
                medium_vol = ~(high_vol | low_vol)

                # Create segments - align indices
                high_vol_data = self.data.iloc[
                    high_vol.index.get_indexer(high_vol[high_vol].index)
                ]
                low_vol_data = self.data.iloc[
                    low_vol.index.get_indexer(low_vol[low_vol].index)
                ]
                medium_vol_data = self.data.iloc[
                    medium_vol.index.get_indexer(medium_vol[medium_vol].index)
                ]

                segments["high"] = high_vol_data
                segments["low"] = low_vol_data
                segments["medium"] = medium_vol_data

            elif condition_key == "momentum_regime":
                roc = (self.data["close"] / self.data["close"].shift(20) - 1) * 100

                bullish = roc > 5
                bearish = roc < -5
                neutral = ~(bullish | bearish)

                # Create segments - align indices
                bullish_data = self.data.iloc[
                    bullish.index.get_indexer(bullish[bullish].index)
                ]
                bearish_data = self.data.iloc[
                    bearish.index.get_indexer(bearish[bearish].index)
                ]
                neutral_data = self.data.iloc[
                    neutral.index.get_indexer(neutral[neutral].index)
                ]

                segments["bullish"] = bullish_data
                segments["bearish"] = bearish_data
                segments["neutral"] = neutral_data

            return segments
        except Exception as e:
            logger.error(f"Error segmenting data by condition: {str(e)}")
            return {}


class AdaptiveStrategyManager:
    """Manager for adapting strategies to changing market conditions."""

    def __init__(self, repository_path: Union[str, Path]):
        """
        Initialize the adaptive strategy manager.

        Args:
            repository_path: Path to strategy repository
        """
        from .pine_script import StrategyRepository

        self.repository_path = Path(repository_path)
        self.strategy_repository = StrategyRepository(repository_path)
        self.optimizations = {}  # Market condition -> strategy ID -> OptimizationResult

        # Create directory for optimization results
        self.optimization_path = self.repository_path / "optimizations"
        self.optimization_path.mkdir(parents=True, exist_ok=True)

    def optimize_for_condition(
        self,
        strategy_id: str,
        data: pd.DataFrame,
        market_condition: str,
        parameter_spaces: List[ParameterSpace],
        optimization_method: str = "random_search",
        **optimizer_kwargs,
    ) -> OptimizationResult:
        """
        Optimize a strategy for a specific market condition.

        Args:
            strategy_id: ID of the strategy to optimize
            data: DataFrame with OHLCV data for the market condition
            market_condition: Name of the market condition
            parameter_spaces: List of parameter spaces to search
            optimization_method: Method to use ('grid_search' or 'random_search')
            **optimizer_kwargs: Additional arguments for optimizer

        Returns:
            OptimizationResult containing the best parameters

        Raises:
            OptimizationError: If optimization fails
        """
        try:
            # Get strategy from repository
            strategy = self.strategy_repository.get_strategy(strategy_id)
            if not strategy:
                raise OptimizationError(f"Strategy not found: {strategy_id}")

            # Create optimizer
            optimizer = StrategyOptimizer(data, **optimizer_kwargs)

            # Run optimization
            if optimization_method == "grid_search":
                result = optimizer.optimize_grid_search(strategy, parameter_spaces)
            else:
                result = optimizer.optimize_random_search(strategy, parameter_spaces)

            # Save result to disk
            result_path = (
                self.optimization_path
                / f"{market_condition}_{strategy_id}_{result.unique_id}.json"
            )
            result.save_to_file(result_path)

            # Store result in memory
            if market_condition not in self.optimizations:
                self.optimizations[market_condition] = {}
            self.optimizations[market_condition][strategy_id] = result

            return result
        except Exception as e:
            logger.error(f"Error optimizing strategy for condition: {str(e)}")
            raise OptimizationError(f"Failed to optimize strategy: {str(e)}") from e

    def get_best_strategy(
        self, market_condition: str, metric: str = "sharpe_ratio"
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Get the best strategy for a given market condition.

        Args:
            market_condition: Name of the market condition
            metric: Performance metric to use for ranking

        Returns:
            Tuple of (strategy_id, parameters) for the best strategy, or (None, None) if none found
        """
        if market_condition not in self.optimizations:
            logger.warning(
                f"No optimizations found for market condition: {market_condition}"
            )
            return None, None

        # Find strategy with best performance on the specified metric
        best_strategy_id = None
        best_parameters = None
        best_value = float("-inf")

        for strategy_id, result in self.optimizations[market_condition].items():
            value = result.performance_metrics.get(metric, float("-inf"))
            if value > best_value:
                best_value = value
                best_strategy_id = strategy_id
                best_parameters = result.best_parameters

        return best_strategy_id, best_parameters

    def select_strategy(
        self, data: pd.DataFrame
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Select the best strategy for current market conditions.

        Args:
            data: Recent OHLCV data for market condition analysis

        Returns:
            Tuple of (strategy_id, parameters) for the selected strategy, or (None, None) if none found
        """
        try:
            # Analyze current market condition
            analyzer = MarketConditionAnalyzer(data)
            conditions = analyzer.detect_conditions()
            market_condition = conditions["market_condition"]

            logger.info(f"Detected market condition: {market_condition}")

            # Select best strategy for this condition
            strategy_id, parameters = self.get_best_strategy(market_condition)

            if not strategy_id:
                # Try to find strategy for similar conditions
                components = market_condition.split("_")
                if len(components) == 3:
                    # Try with just trend and volatility
                    simpler_condition = f"{components[0]}_{components[1]}"
                    strategy_id, parameters = self.get_best_strategy(simpler_condition)

                    if not strategy_id:
                        # Try with just trend
                        strategy_id, parameters = self.get_best_strategy(components[0])

            return strategy_id, parameters
        except Exception as e:
            logger.error(f"Error selecting strategy: {str(e)}")
            return None, None

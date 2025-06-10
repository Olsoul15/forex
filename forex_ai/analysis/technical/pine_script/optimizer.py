"""
Pine Script optimizer.

Optimizes Pine Script strategy parameters based on historical performance
and current market conditions.
"""

import re
import json
import logging
import itertools
import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Dict, List, Any, Optional, Union, Tuple, Callable

from forex_ai.exceptions import StrategyError, StrategyNotFoundError, OptimizationError
from forex_ai.custom_types import CurrencyPair, TimeFrame, MarketCondition
from forex_ai.analysis.technical.pine_script.manager import PineScriptStrategyManager

logger = logging.getLogger(__name__)


class PineScriptOptimizer:
    """
    Optimizes Pine Script strategy parameters based on historical performance
    and current market conditions.
    """

    def __init__(
        self, db_connection=None, market_data_client=None, pandas_mcp_client=None
    ):
        self.db = db_connection  # PostgreSQL MCP connection
        self.market_data = market_data_client  # Market data client
        self.pandas = pandas_mcp_client  # Pandas MCP connection
        self.strategy_manager = PineScriptStrategyManager(db_connection)

    def extract_parameters(self, pine_script_code: str) -> Dict[str, Dict[str, Any]]:
        """
        Extract parameters from Pine Script code.

        Args:
            pine_script_code: Pine Script code.

        Returns:
            Dictionary of parameter details, including name, default value, and range.
        """
        try:
            parameters = {}

            # Match input parameters with their type and default value
            # Example: input(title="ATR Period", type=input.integer, defval=14)
            param_pattern = r'input\s*\(\s*title\s*=\s*["\']([^"\']+)["\'](?:\s*,\s*type\s*=\s*input\.([^,]+))?\s*,\s*defval\s*=\s*([^,\)]+)(?:\s*,\s*minval\s*=\s*([^,\)]+))?(?:\s*,\s*maxval\s*=\s*([^,\)]+))?'

            matches = re.finditer(param_pattern, pine_script_code)
            for match in matches:
                title, param_type, defval = match.groups()[:3]
                param_id = re.sub(r"[^a-z0-9_]", "_", title.lower())

                # Parse default value based on type
                value = self._parse_value(defval, param_type)

                # Extract min/max values if available
                minval = match.group(4)
                maxval = match.group(5)

                param_info = {
                    "title": title,
                    "type": param_type or "float",
                    "default": value,
                }

                # Add min/max values if available
                if minval:
                    param_info["min"] = self._parse_value(minval, param_type)

                if maxval:
                    param_info["max"] = self._parse_value(maxval, param_type)

                # If no min/max provided, create reasonable ranges based on type and default
                if "min" not in param_info or "max" not in param_info:
                    self._add_default_ranges(param_info)

                parameters[param_id] = param_info

            return parameters

        except Exception as e:
            logger.error(f"Error extracting parameters: {str(e)}")
            raise OptimizationError(f"Failed to extract parameters: {str(e)}")

    def replace_parameters(
        self, pine_script_code: str, parameters: Dict[str, Any]
    ) -> str:
        """
        Update Pine Script code with new parameters.

        Args:
            pine_script_code: Original Pine Script code.
            parameters: Dictionary of parameter names and new values.

        Returns:
            Updated Pine Script code.
        """
        try:
            updated_code = pine_script_code

            # First extract parameter details to know their titles
            param_details = self.extract_parameters(pine_script_code)

            # For each parameter we want to replace
            for param_id, new_value in parameters.items():
                if param_id not in param_details:
                    logger.warning(
                        f"Parameter '{param_id}' not found in Pine Script code"
                    )
                    continue

                param_info = param_details[param_id]
                title = param_info["title"]

                # Format the new value based on type
                formatted_value = self._format_value(new_value, param_info["type"])

                # Create a pattern to match the input declaration
                pattern = rf'(input\s*\(\s*title\s*=\s*["\']({re.escape(title)})["\'][^,]*,\s*defval\s*=\s*)([^,\)]+)(\s*[,\)])'

                # Replace the defval in the pattern
                updated_code = re.sub(pattern, rf"\1{formatted_value}\4", updated_code)

            return updated_code

        except Exception as e:
            logger.error(f"Error replacing parameters: {str(e)}")
            raise OptimizationError(f"Failed to replace parameters: {str(e)}")

    def optimize_strategy(
        self,
        strategy_id: str,
        market_conditions: Union[
            str, MarketCondition, List[Union[str, MarketCondition]]
        ],
        currency_pair: CurrencyPair,
        timeframe: TimeFrame,
        optimization_metric: str = "sharpe_ratio",
        param_grid: Optional[Dict[str, List[Any]]] = None,
        max_combinations: int = 100,
        historical_days: int = 90,
    ) -> Dict[str, Any]:
        """
        Optimize a strategy for the given market conditions.

        Args:
            strategy_id: Unique identifier of the strategy to optimize.
            market_conditions: Market conditions to optimize for.
            currency_pair: Currency pair to optimize for.
            timeframe: Timeframe to optimize for.
            optimization_metric: Metric to optimize ('sharpe_ratio', 'profit_factor', 'net_profit').
            param_grid: Optional custom parameter grid. If None, auto-generates.
            max_combinations: Maximum number of parameter combinations to test.
            historical_days: Number of days of historical data to use.

        Returns:
            Dictionary with optimized parameters and performance metrics.

        Raises:
            StrategyNotFoundError: If strategy not found.
            OptimizationError: If optimization fails.
        """
        try:
            # Retrieve the strategy
            strategy = self.strategy_manager.get_strategy(strategy_id)

            # Extract parameters
            params = self.extract_parameters(strategy.code)

            # If no custom grid provided, generate one
            if not param_grid:
                param_grid = self._generate_param_grid(params, max_combinations)

            # Get historical data
            historical_data = self._get_historical_data(
                currency_pair, timeframe, days=historical_days
            )

            # Generate parameter combinations
            param_keys = list(param_grid.keys())
            param_values = list(param_grid.values())

            # Limit the number of combinations to avoid excessive computation
            combinations = list(itertools.product(*param_values))
            if len(combinations) > max_combinations:
                logger.warning(
                    f"Too many parameter combinations ({len(combinations)}), limiting to {max_combinations}"
                )
                np.random.shuffle(combinations)
                combinations = combinations[:max_combinations]

            # Execute backtests for each parameter set
            results = []
            for combination in combinations:
                params_dict = dict(zip(param_keys, combination))
                backtest_result = self.backtest_parameters(
                    strategy_id, params_dict, historical_data
                )

                # Add parameters to results
                backtest_result["parameters"] = params_dict
                results.append(backtest_result)

            # Sort results by optimization metric
            sorted_results = sorted(
                results,
                key=lambda x: x.get(optimization_metric, float("-inf")),
                reverse=True,
            )

            # Get the best parameters
            best_result = sorted_results[0] if sorted_results else None

            if not best_result:
                raise OptimizationError("No valid optimization results found")

            # Update the strategy with optimized parameters
            updated_code = self.replace_parameters(
                strategy.code, best_result["parameters"]
            )

            # Prepare result
            optimization_result = {
                "strategy_id": strategy_id,
                "currency_pair": str(currency_pair),
                "timeframe": str(timeframe),
                "market_conditions": (
                    market_conditions
                    if isinstance(market_conditions, list)
                    else [market_conditions]
                ),
                "optimization_metric": optimization_metric,
                "parameters": best_result["parameters"],
                "metrics": {k: v for k, v in best_result.items() if k != "parameters"},
                "updated_code": updated_code,
                "all_results": sorted_results[:10],  # Return top 10 results
            }

            return optimization_result

        except Exception as e:
            logger.error(f"Error optimizing strategy '{strategy_id}': {str(e)}")
            if isinstance(e, StrategyNotFoundError):
                raise
            raise OptimizationError(
                f"Failed to optimize strategy '{strategy_id}': {str(e)}"
            )

    def backtest_parameters(
        self,
        strategy_id: str,
        parameters: Dict[str, Any],
        historical_data: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Run backtest with specific parameters.

        Args:
            strategy_id: Unique identifier of the strategy.
            parameters: Dictionary of parameter names and values.
            historical_data: Historical market data for backtesting.

        Returns:
            Dictionary with backtest performance metrics.
        """
        try:
            # Retrieve the strategy
            strategy = self.strategy_manager.get_strategy(strategy_id)

            # Update the strategy with new parameters
            updated_code = self.replace_parameters(strategy.code, parameters)

            # Normally we would execute the Pine Script here
            # As a placeholder, we'll implement a simple backtesting system
            metrics = self._simulate_backtest(updated_code, historical_data, parameters)

            return metrics

        except Exception as e:
            logger.error(f"Error backtesting parameters for '{strategy_id}': {str(e)}")
            if isinstance(e, StrategyNotFoundError):
                raise
            raise OptimizationError(f"Failed to backtest parameters: {str(e)}")

    def adaptive_selection(
        self,
        currency_pair: CurrencyPair,
        timeframe: TimeFrame,
        current_market_data: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Select and optimize the most appropriate strategy for current market conditions.

        Args:
            currency_pair: Currency pair to select strategy for.
            timeframe: Timeframe to select strategy for.
            current_market_data: Recent market data for condition detection.

        Returns:
            Dictionary with selected strategy and optimized parameters.
        """
        try:
            # Detect current market conditions
            market_condition = self._detect_market_condition(current_market_data)

            # List compatible strategies
            strategies = self.strategy_manager.list_strategies(
                {
                    "currency_pair": currency_pair,
                    "timeframe": timeframe,
                    "market_condition": market_condition,
                }
            )

            # If no strategies found, try with all market conditions
            if not strategies:
                logger.warning(
                    f"No strategies found for {market_condition}, listing all compatible strategies"
                )
                strategies = self.strategy_manager.list_strategies(
                    {"currency_pair": currency_pair, "timeframe": timeframe}
                )

            if not strategies:
                raise OptimizationError(
                    f"No compatible strategies found for {currency_pair}, {timeframe}"
                )

            # For now, just select the first strategy
            # In a real implementation, would rank strategies by past performance
            selected_strategy = strategies[0]

            # Optimize selected strategy
            optimized_result = self.optimize_strategy(
                selected_strategy.id, market_condition, currency_pair, timeframe
            )

            return {
                "strategy": selected_strategy.to_dict(),
                "market_condition": str(market_condition),
                "optimization_result": optimized_result,
            }

        except Exception as e:
            logger.error(f"Error in adaptive selection: {str(e)}")
            raise OptimizationError(f"Failed to select appropriate strategy: {str(e)}")

    # Helper methods

    def _parse_value(self, value_str: str, param_type: Optional[str] = None) -> Any:
        """Parse parameter value based on type."""
        value_str = value_str.strip()

        if param_type == "integer":
            return int(float(value_str))
        elif param_type == "bool":
            return value_str.lower() in ("true", "1", "yes")
        elif param_type == "string":
            # Remove quotes
            if (value_str.startswith('"') and value_str.endswith('"')) or (
                value_str.startswith("'") and value_str.endswith("'")
            ):
                return value_str[1:-1]
            return value_str
        else:
            # Default to float
            try:
                return float(value_str)
            except ValueError:
                return value_str

    def _format_value(self, value: Any, param_type: str) -> str:
        """Format value for insertion into Pine Script code."""
        if param_type == "string":
            return f'"{value}"'
        elif param_type == "bool":
            return "true" if value else "false"
        else:
            return str(value)

    def _add_default_ranges(self, param_info: Dict[str, Any]) -> None:
        """Add default parameter ranges if not specified."""
        if param_info["type"] == "integer":
            default = param_info["default"]
            if "min" not in param_info:
                param_info["min"] = max(1, int(default / 2))
            if "max" not in param_info:
                param_info["max"] = default * 2

        elif param_info["type"] == "float":
            default = param_info["default"]
            if "min" not in param_info:
                param_info["min"] = default * 0.5
            if "max" not in param_info:
                param_info["max"] = default * 2.0

        elif param_info["type"] == "bool":
            # Boolean has only two values
            param_info["min"] = False
            param_info["max"] = True

    def _generate_param_grid(
        self, params: Dict[str, Dict[str, Any]], max_combinations: int
    ) -> Dict[str, List[Any]]:
        """Generate parameter grid for optimization."""
        param_grid = {}

        for param_id, param_info in params.items():
            param_type = param_info["type"]

            if param_type == "integer":
                min_val = param_info.get("min", 1)
                max_val = param_info.get("max", 20)

                # Generate reasonable number of steps based on range
                range_size = max_val - min_val

                # Determine number of steps
                num_steps = min(10, range_size + 1)

                # Generate evenly spaced values
                if num_steps > 1:
                    values = np.linspace(
                        min_val, max_val, num_steps, dtype=int
                    ).tolist()
                else:
                    values = [min_val]

                param_grid[param_id] = values

            elif param_type == "float":
                min_val = param_info.get("min", 0.1)
                max_val = param_info.get("max", 10.0)

                # Generate 5-10 values in the range
                num_steps = min(10, int((max_val - min_val) / 0.1) + 1)

                if num_steps > 1:
                    values = np.linspace(min_val, max_val, num_steps).tolist()
                else:
                    values = [min_val]

                param_grid[param_id] = values

            elif param_type == "bool":
                param_grid[param_id] = [False, True]

            elif param_type == "string":
                # Strings are tricky to optimize, use only the default for now
                param_grid[param_id] = [param_info["default"]]

        # If we have too many combinations, reduce the grid size
        total_combinations = 1
        for values in param_grid.values():
            total_combinations *= len(values)

        # If we have too many combinations, reduce the number of values for each parameter
        if total_combinations > max_combinations:
            logger.info(
                f"Too many parameter combinations ({total_combinations}), reducing grid size"
            )
            reduction_factor = (max_combinations / total_combinations) ** (
                1 / len(param_grid)
            )

            for param_id, values in param_grid.items():
                if len(values) > 2:
                    # Keep at least 2 values per parameter
                    new_size = max(2, int(len(values) * reduction_factor))

                    # Select evenly spaced values including the first and last
                    if new_size < len(values):
                        indices = np.round(
                            np.linspace(0, len(values) - 1, new_size)
                        ).astype(int)
                        param_grid[param_id] = [values[i] for i in indices]

        return param_grid

    def _get_historical_data(
        self, currency_pair: CurrencyPair, timeframe: TimeFrame, days: int = 90
    ) -> pd.DataFrame:
        """Get historical data for backtesting."""
        try:
            # Use the market data client if available
            if self.market_data:
                # This would call the real market data client
                logger.info(
                    f"Getting historical data from market data client for {currency_pair} {timeframe}"
                )
                # return self.market_data.get_historical_data(currency_pair, timeframe, days)

            # Otherwise, generate simulated data
            logger.warning("Using simulated market data for backtesting")

            # Generate a date range
            periods = days * 24  # Assuming hourly data
            if str(timeframe).endswith("m"):
                # Minutes data
                minutes = int(str(timeframe)[:-1])
                periods = (days * 24 * 60) // minutes
            elif str(timeframe).endswith("h"):
                # Hourly data
                hours = int(str(timeframe)[:-1])
                periods = (days * 24) // hours
            elif str(timeframe) == "D":
                # Daily data
                periods = days

            end_date = pd.Timestamp.now()
            start_date = end_date - pd.Timedelta(days=days)

            # Create date range
            date_range = pd.date_range(start=start_date, end=end_date, periods=periods)

            # Generate random price data with a trend
            np.random.seed(42)  # For reproducibility

            # Start with a base price
            base_price = 1.0
            if currency_pair.base == "BTC":
                base_price = 30000.0
            elif currency_pair.base == "ETH":
                base_price = 2000.0
            elif currency_pair.base == "EUR":
                base_price = 1.1  # vs USD
            elif currency_pair.base == "GBP":
                base_price = 1.3  # vs USD

            # Generate price series with some randomness and trend
            trend = 0.0001  # Small upward trend
            volatility = 0.005  # Moderate volatility

            # Generate random walks
            log_returns = np.random.normal(trend, volatility, periods)
            log_prices = np.cumsum(log_returns)

            # Convert to prices
            prices = base_price * np.exp(log_prices)

            # Generate OHLC data
            df = pd.DataFrame(index=date_range)
            df["open"] = prices

            # Generate high, low, close
            daily_volatility = volatility * np.sqrt(1 / 252)  # Scale volatility
            df["high"] = df["open"] * np.exp(
                np.random.normal(0, daily_volatility, len(df))
            )
            df["low"] = df["open"] * np.exp(
                np.random.normal(0, -daily_volatility, len(df))
            )
            df["close"] = df["open"] * np.exp(log_returns)

            # Ensure high >= open, close and low <= open, close
            df["high"] = df[["high", "open", "close"]].max(axis=1)
            df["low"] = df[["low", "open", "close"]].min(axis=1)

            # Add volume
            df["volume"] = np.random.lognormal(10, 1, len(df))

            return df

        except Exception as e:
            logger.error(f"Error getting historical data: {str(e)}")
            raise OptimizationError(f"Failed to get historical data: {str(e)}")

    def _detect_market_condition(self, market_data: pd.DataFrame) -> MarketCondition:
        """Detect market condition from recent market data."""
        try:
            # Simple detection logic
            if len(market_data) < 20:
                return MarketCondition.UNKNOWN

            # Calculate returns
            returns = market_data["close"].pct_change().dropna()

            # Calculate some basic metrics
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility

            # Calculate trend (simple moving average slope)
            ma20 = market_data["close"].rolling(20).mean().dropna()
            ma_slope = (
                (ma20.iloc[-1] - ma20.iloc[0]) / ma20.iloc[0] if len(ma20) > 0 else 0
            )

            # Detect market condition based on volatility and trend
            if volatility > 0.2:  # High volatility
                return MarketCondition.VOLATILE
            elif abs(ma_slope) > 0.05:  # Strong trend
                return (
                    MarketCondition.TRENDING
                    if ma_slope > 0
                    else MarketCondition.BEARISH
                )
            else:
                return MarketCondition.RANGING

        except Exception as e:
            logger.error(f"Error detecting market condition: {str(e)}")
            return MarketCondition.UNKNOWN

    def _simulate_backtest(
        self,
        pine_script_code: str,
        historical_data: pd.DataFrame,
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Simulate backtest execution for a Pine Script and parameters.

        This is a placeholder for an actual backtest execution. In a real implementation,
        this would either execute the Pine Script or parse it to Python and execute it.
        """
        try:
            # Extract strategy name from Pine Script
            strategy_name_match = re.search(
                r'strategy\s*\(\s*["\']([^"\']+)["\']', pine_script_code
            )
            strategy_name = (
                strategy_name_match.group(1)
                if strategy_name_match
                else "Unknown Strategy"
            )

            # Get parameter values for the simulation
            atr_period = parameters.get("atr_period", 14)
            stop_loss = parameters.get("stop_loss", 2.0)
            take_profit = parameters.get("take_profit", 3.0)
            ema_period = parameters.get("ema_period", 50)

            # Generate simple trading signals based on parameters
            # This is a very simplified simulation

            # Calculate EMA
            historical_data["ema"] = (
                historical_data["close"].ewm(span=ema_period, adjust=False).mean()
            )

            # Calculate ATR
            high_low = historical_data["high"] - historical_data["low"]
            high_close = np.abs(
                historical_data["high"] - historical_data["close"].shift()
            )
            low_close = np.abs(
                historical_data["low"] - historical_data["close"].shift()
            )
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            historical_data["atr"] = tr.rolling(atr_period).mean()

            # Generate signals
            # Buy when price crosses above EMA, sell when it crosses below
            historical_data["signal"] = 0
            historical_data.loc[
                historical_data["close"] > historical_data["ema"], "signal"
            ] = 1
            historical_data.loc[
                historical_data["close"] < historical_data["ema"], "signal"
            ] = -1

            # Calculate position - enter when signal changes
            historical_data["position"] = historical_data["signal"].diff().fillna(0)

            # Simplified trade management with stop loss and take profit
            trades = []
            current_trade = None

            for i, row in historical_data.iterrows():
                if current_trade is None:
                    # No open trade, check for entry
                    if row["position"] == 1:  # Buy entry
                        current_trade = {
                            "entry_date": i,
                            "entry_price": row["close"],
                            "stop_loss": row["close"] - row["atr"] * stop_loss,
                            "take_profit": row["close"] + row["atr"] * take_profit,
                            "direction": "long",
                        }
                    elif row["position"] == -1:  # Sell entry
                        current_trade = {
                            "entry_date": i,
                            "entry_price": row["close"],
                            "stop_loss": row["close"] + row["atr"] * stop_loss,
                            "take_profit": row["close"] - row["atr"] * take_profit,
                            "direction": "short",
                        }
                else:
                    # Have an open trade, check for exit
                    if current_trade["direction"] == "long":
                        # Check for exit conditions
                        if row["low"] <= current_trade["stop_loss"]:
                            # Stop loss hit
                            current_trade["exit_date"] = i
                            current_trade["exit_price"] = current_trade["stop_loss"]
                            current_trade["exit_type"] = "stop_loss"
                            current_trade["profit"] = (
                                current_trade["exit_price"]
                                - current_trade["entry_price"]
                            ) / current_trade["entry_price"]
                            trades.append(current_trade)
                            current_trade = None
                        elif row["high"] >= current_trade["take_profit"]:
                            # Take profit hit
                            current_trade["exit_date"] = i
                            current_trade["exit_price"] = current_trade["take_profit"]
                            current_trade["exit_type"] = "take_profit"
                            current_trade["profit"] = (
                                current_trade["exit_price"]
                                - current_trade["entry_price"]
                            ) / current_trade["entry_price"]
                            trades.append(current_trade)
                            current_trade = None
                        elif row["position"] == -1:
                            # Signal reversal
                            current_trade["exit_date"] = i
                            current_trade["exit_price"] = row["close"]
                            current_trade["exit_type"] = "signal"
                            current_trade["profit"] = (
                                current_trade["exit_price"]
                                - current_trade["entry_price"]
                            ) / current_trade["entry_price"]
                            trades.append(current_trade)
                            current_trade = None
                    else:  # Short trade
                        # Check for exit conditions
                        if row["high"] >= current_trade["stop_loss"]:
                            # Stop loss hit
                            current_trade["exit_date"] = i
                            current_trade["exit_price"] = current_trade["stop_loss"]
                            current_trade["exit_type"] = "stop_loss"
                            current_trade["profit"] = (
                                current_trade["entry_price"]
                                - current_trade["exit_price"]
                            ) / current_trade["entry_price"]
                            trades.append(current_trade)
                            current_trade = None
                        elif row["low"] <= current_trade["take_profit"]:
                            # Take profit hit
                            current_trade["exit_date"] = i
                            current_trade["exit_price"] = current_trade["take_profit"]
                            current_trade["exit_type"] = "take_profit"
                            current_trade["profit"] = (
                                current_trade["entry_price"]
                                - current_trade["exit_price"]
                            ) / current_trade["entry_price"]
                            trades.append(current_trade)
                            current_trade = None
                        elif row["position"] == 1:
                            # Signal reversal
                            current_trade["exit_date"] = i
                            current_trade["exit_price"] = row["close"]
                            current_trade["exit_type"] = "signal"
                            current_trade["profit"] = (
                                current_trade["entry_price"]
                                - current_trade["exit_price"]
                            ) / current_trade["entry_price"]
                            trades.append(current_trade)
                            current_trade = None

            # Close any open trade at the end
            if current_trade is not None:
                current_trade["exit_date"] = historical_data.index[-1]
                current_trade["exit_price"] = historical_data["close"].iloc[-1]
                current_trade["exit_type"] = "end_of_data"
                if current_trade["direction"] == "long":
                    current_trade["profit"] = (
                        current_trade["exit_price"] - current_trade["entry_price"]
                    ) / current_trade["entry_price"]
                else:
                    current_trade["profit"] = (
                        current_trade["entry_price"] - current_trade["exit_price"]
                    ) / current_trade["entry_price"]
                trades.append(current_trade)

            # Calculate performance metrics
            if not trades:
                return {
                    "strategy_name": strategy_name,
                    "total_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "win_rate": 0.0,
                    "net_profit": 0.0,
                    "profit_factor": 0.0,
                    "sharpe_ratio": 0.0,
                    "max_drawdown": 0.0,
                }

            total_trades = len(trades)
            winning_trades = sum(1 for trade in trades if trade["profit"] > 0)
            losing_trades = sum(1 for trade in trades if trade["profit"] <= 0)

            win_rate = winning_trades / total_trades if total_trades > 0 else 0

            total_profit = sum(
                trade["profit"] for trade in trades if trade["profit"] > 0
            )
            total_loss = abs(
                sum(trade["profit"] for trade in trades if trade["profit"] <= 0)
            )

            net_profit = total_profit - total_loss
            profit_factor = (
                total_profit / total_loss if total_loss > 0 else float("inf")
            )

            # Calculate equity curve
            equity = 10000  # Starting capital
            equity_curve = [equity]

            for trade in trades:
                trade_profit = equity * trade["profit"]
                equity += trade_profit
                equity_curve.append(equity)

            # Calculate max drawdown
            peak = equity_curve[0]
            max_drawdown = 0

            for value in equity_curve:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)

            # Calculate Sharpe ratio
            returns = [
                (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
                for i in range(1, len(equity_curve))
            ]
            sharpe_ratio = (
                np.mean(returns) / np.std(returns) * np.sqrt(252)
                if np.std(returns) > 0
                else 0
            )

            return {
                "strategy_name": strategy_name,
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": win_rate,
                "net_profit": net_profit,
                "profit_factor": profit_factor,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "final_equity": equity_curve[-1],
                "return_pct": (equity_curve[-1] - equity_curve[0])
                / equity_curve[0]
                * 100,
            }

        except Exception as e:
            logger.error(f"Error simulating backtest: {str(e)}")

            # Return default metrics
            return {
                "strategy_name": "Error in Backtest",
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "net_profit": 0.0,
                "profit_factor": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "error": str(e),
            }

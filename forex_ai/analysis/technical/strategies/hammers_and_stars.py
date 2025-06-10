"""
Python implementation of the Hammers and Stars trading strategy.

This module adapts the TradingView Pine Script strategy 'Hammers & Stars Strategy'
into a Python implementation that can be integrated with the AI Forex Trading System.
It detects hammer and shooting star candlestick patterns for trading signals.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta

from forex_ai.exceptions import StrategyError, PatternError
from forex_ai.custom_types import CurrencyPair, TimeFrame

logger = logging.getLogger(__name__)


class HammersAndStarsStrategy:
    """
    Python implementation of the Hammers and Stars strategy from TradingView.
    Integrated with the AI optimization system.

    The strategy identifies hammer and shooting star candlestick patterns
    to generate trading signals based on price action. It includes various
    filters and customizable parameters for optimizing performance.

    Original strategy by BallsBulls and ZenAndTheArtOfTrading / PineScriptMastery
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the strategy with customizable parameters.

        Args:
            params: Optional dictionary of strategy parameters. If None, uses defaults.
        """
        # Default parameters (from the original Pine Script)
        self.params = params or {
            "atr_min_filter_size": 0.0,
            "atr_max_filter_size": 3.0,
            "stop_multiplier": 1.0,
            "risk_reward": 1.0,
            "fib_level": 0.333,
            "ema_filter": 0,
        }

        # Additional parameters for strategy management
        self.name = "Hammers & Stars Strategy"
        self.version = "1.0.0"
        self.description = (
            "Detects hammer and shooting star candlestick patterns for trading signals"
        )
        self.author = "Forex AI System (adapted from BallsBulls/ZenAndTheArtOfTrading)"
        self.id = "hammers_and_stars"

        # Strategy state variables
        self.current_position = 0  # 0: no position, 1: long, -1: short
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.entry_time = None

    def generate_pine_script(self) -> str:
        """
        Generate Pine Script code with current parameters.

        Returns:
            String containing Pine Script implementation with current parameters.
        """
        # This template is based on the original Pine Script with parameterized values
        pine_script = f"""// This source code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org/MPL/2.0/
// Adapted by Forex AI System from BallsBulls/ZenAndTheArtOfTrading
// @version=4
strategy("Hammers & Stars Strategy [v1.1]", shorttitle="HSS[v1.1]", overlay=true)

// Strategy Settings
var g_strategy      = "Strategy Settings"
atrMinFilterSize    = input(title=">= ATR Filter", type=input.float, defval={self.params['atr_min_filter_size']}, minval=0.0, group=g_strategy, tooltip="Minimum size of entry candle compared to ATR")
atrMaxFilterSize    = input(title="<= ATR Filter", type=input.float, defval={self.params['atr_max_filter_size']}, minval=0.0, group=g_strategy, tooltip="Maximum size of entry candle compared to ATR")
stopMultiplier      = input(title="Stop Loss ATR", type=input.float, defval={self.params['stop_multiplier']}, group=g_strategy, tooltip="Stop loss multiplier (x ATR)")
rr                  = input(title="R:R", type=input.float, defval={self.params['risk_reward']}, group=g_strategy, tooltip="Risk:Reward profile")
fibLevel            = input(title="Fib Level", type=input.float, defval={self.params['fib_level']}, group=g_strategy, tooltip="Used to calculate upper/lower third of candle.")
// Filter Settings
var g_filter        = "Filter Settings"
emaFilter           = input(title="EMA Filter", type=input.integer, defval={self.params['ema_filter']}, group=g_filter, tooltip="EMA length to filter trades - set to zero to disable")

// Get indicator values
atr = atr(14)
ema = ema(close, emaFilter == 0 ? 1 : emaFilter)

// Calculate 33.3% fibonacci level for current candle
bullFib = (low - high) * fibLevel + high
bearFib = (high - low) * fibLevel + low

// Check EMA Filter
emaFilterLong = emaFilter == 0 or close > ema
emaFilterShort = emaFilter == 0 or close < ema

// Check ATR filter
atrMinFilter = abs(high - low) >= (atrMinFilterSize * atr) or atrMinFilterSize == 0.0
atrMaxFilter = abs(high - low) <= (atrMaxFilterSize * atr) or atrMaxFilterSize == 0.0
atrFilter = atrMinFilter and atrMaxFilter

// Determine which price source closes or opens highest/lowest
lowestBody = close < open ? close : open
highestBody = close > open ? close : open

// Determine if we have a valid setup
validHammer = lowestBody >= bullFib and atrFilter and close != open and not na(atr) and emaFilterLong
validStar = highestBody <= bearFib and atrFilter and close != open and not na(atr) and emaFilterShort

// Check if we have confirmation for our setup
validLong = validHammer and strategy.position_size == 0
validShort = validStar and strategy.position_size == 0

// Calculate our stop distance & size for the current bar
stopSize = atr * stopMultiplier
longStopPrice = low < low[1] ? low - stopSize : low[1] - stopSize
longStopDistance = close - longStopPrice
longTargetPrice = close + (longStopDistance * rr)
shortStopPrice = high > high[1] ? high + stopSize : high[1] + stopSize
shortStopDistance = shortStopPrice - close
shortTargetPrice = close - (shortStopDistance * rr)

// Execute strategy
strategy.entry(id="Long", long=true, when=validLong)
strategy.entry(id="Short", long=false, when=validShort)

// Exit trades whenever our stop or target is hit
strategy.exit(id="Long Exit", from_entry="Long", limit=longTargetPrice, stop=longStopPrice)
strategy.exit(id="Short Exit", from_entry="Short", limit=shortTargetPrice, stop=shortStopPrice)

// Draw setup arrows
plotshape(validLong ? 1 : na, style=shape.triangleup, location=location.belowbar, color=color.green, title="Bullish Setup")
plotshape(validShort ? 1 : na, style=shape.triangledown, location=location.abovebar, color=color.red, title="Bearish Setup")
"""
        return pine_script

    def analyze_candle(
        self, candle: Dict[str, float], atr: float, ema: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Analyze a candle for hammer or shooting star patterns.

        Args:
            candle: Dictionary with OHLC values for the candle.
            atr: Average True Range value.
            ema: Optional EMA value for filtering.

        Returns:
            Dictionary with analysis results including pattern type and entry/exit levels.
        """
        try:
            # Extract OHLC values
            open_price = candle["open"]
            high_price = candle["high"]
            low_price = candle["low"]
            close_price = candle["close"]

            # Check for zero-sized candle
            if close_price == open_price:
                return {
                    "pattern": "none",
                    "valid_setup": False,
                    "message": "Zero-sized candle",
                }

            # Calculate candle size and check ATR filter
            candle_size = abs(high_price - low_price)
            atr_min_filter = (
                candle_size >= (self.params["atr_min_filter_size"] * atr)
                or self.params["atr_min_filter_size"] == 0.0
            )
            atr_max_filter = (
                candle_size <= (self.params["atr_max_filter_size"] * atr)
                or self.params["atr_max_filter_size"] == 0.0
            )
            atr_filter = atr_min_filter and atr_max_filter

            if not atr_filter:
                return {
                    "pattern": "none",
                    "valid_setup": False,
                    "message": "Failed ATR filter",
                }

            # Check EMA filter if provided
            ema_filter_long = (
                self.params["ema_filter"] == 0 or ema is None or close_price > ema
            )
            ema_filter_short = (
                self.params["ema_filter"] == 0 or ema is None or close_price < ema
            )

            # Calculate Fibonacci levels for pattern detection
            fib_level = self.params["fib_level"]
            bull_fib = (low_price - high_price) * fib_level + high_price
            bear_fib = (high_price - low_price) * fib_level + low_price

            # Determine body high/low points
            lowest_body = min(close_price, open_price)
            highest_body = max(close_price, open_price)

            # Determine if we have hammer or shooting star pattern
            is_hammer = lowest_body >= bull_fib and ema_filter_long
            is_star = highest_body <= bear_fib and ema_filter_short

            # Calculate stop loss and take profit levels
            stop_size = atr * self.params["stop_multiplier"]

            # Calculate previous candle data (this would need historical data in real implementation)
            # For now, we'll just use current candle information
            prev_low = low_price
            prev_high = high_price

            # Calculate long setup levels
            long_stop_price = min(low_price, prev_low) - stop_size
            long_stop_distance = close_price - long_stop_price
            long_target_price = close_price + (
                long_stop_distance * self.params["risk_reward"]
            )

            # Calculate short setup levels
            short_stop_price = max(high_price, prev_high) + stop_size
            short_stop_distance = short_stop_price - close_price
            short_target_price = close_price - (
                short_stop_distance * self.params["risk_reward"]
            )

            # Determine pattern type and validity
            pattern = "none"
            valid_setup = False
            message = "No valid pattern detected"

            if is_hammer:
                pattern = "hammer"
                valid_setup = True
                message = "Bullish Hammer detected"
            elif is_star:
                pattern = "shooting_star"
                valid_setup = True
                message = "Bearish Shooting Star detected"

            # Prepare result
            result = {
                "pattern": pattern,
                "valid_setup": valid_setup,
                "message": message,
                "long": {
                    "valid": is_hammer,
                    "entry": close_price,
                    "stop_loss": long_stop_price,
                    "take_profit": long_target_price,
                    "risk_pips": long_stop_distance,
                    "reward_pips": long_target_price - close_price,
                },
                "short": {
                    "valid": is_star,
                    "entry": close_price,
                    "stop_loss": short_stop_price,
                    "take_profit": short_target_price,
                    "risk_pips": short_stop_distance,
                    "reward_pips": close_price - short_target_price,
                },
            }

            return result

        except Exception as e:
            logger.error(f"Error analyzing candle: {str(e)}")
            return {
                "pattern": "error",
                "valid_setup": False,
                "message": f"Error analyzing candle: {str(e)}",
            }

    def calculate_entry_exit(
        self, candle: Dict[str, float], atr: float, pattern_type: str
    ) -> Dict[str, float]:
        """
        Calculate entry, stop loss, and take profit levels.

        Args:
            candle: Dictionary with OHLC values.
            atr: Average True Range value.
            pattern_type: 'hammer' or 'shooting_star'.

        Returns:
            Dictionary with entry, stop loss and take profit prices.
        """
        try:
            # Extract prices
            close_price = candle["close"]
            high_price = candle["high"]
            low_price = candle["low"]

            # Calculate stop size based on ATR
            stop_size = atr * self.params["stop_multiplier"]

            if pattern_type == "hammer":
                # Long trade
                stop_loss = low_price - stop_size
                stop_distance = close_price - stop_loss
                take_profit = close_price + (stop_distance * self.params["risk_reward"])

                return {
                    "entry": close_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "direction": "long",
                    "risk_pips": stop_distance,
                    "reward_pips": take_profit - close_price,
                }

            elif pattern_type == "shooting_star":
                # Short trade
                stop_loss = high_price + stop_size
                stop_distance = stop_loss - close_price
                take_profit = close_price - (stop_distance * self.params["risk_reward"])

                return {
                    "entry": close_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "direction": "short",
                    "risk_pips": stop_distance,
                    "reward_pips": close_price - take_profit,
                }

            else:
                raise ValueError(f"Invalid pattern type: {pattern_type}")

        except Exception as e:
            logger.error(f"Error calculating entry/exit levels: {str(e)}")
            raise StrategyError(f"Failed to calculate entry/exit levels: {str(e)}")

    def analyze_market_data(
        self, df: pd.DataFrame, timeframe: TimeFrame
    ) -> pd.DataFrame:
        """
        Analyze market data to identify patterns and generate signals.

        Args:
            df: DataFrame with OHLCV data.
            timeframe: TimeFrame of the data.

        Returns:
            DataFrame with added columns for pattern identification and trade signals.
        """
        try:
            # Ensure required columns exist
            required_columns = ["open", "high", "low", "close"]
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' missing from DataFrame")

            # Calculate ATR
            df["atr"] = self._calculate_atr(df, 14)

            # Calculate EMA if enabled
            if self.params["ema_filter"] > 0:
                df["ema"] = (
                    df["close"].ewm(span=self.params["ema_filter"], adjust=False).mean()
                )
            else:
                df["ema"] = float("nan")

            # Calculate Fibonacci levels
            df["bull_fib"] = (df["low"] - df["high"]) * self.params["fib_level"] + df[
                "high"
            ]
            df["bear_fib"] = (df["high"] - df["low"]) * self.params["fib_level"] + df[
                "low"
            ]

            # Determine body high/low
            df["lowest_body"] = df[["open", "close"]].min(axis=1)
            df["highest_body"] = df[["open", "close"]].max(axis=1)

            # Check ATR filter
            df["candle_size"] = abs(df["high"] - df["low"])
            df["atr_min_filter"] = (
                df["candle_size"] >= (self.params["atr_min_filter_size"] * df["atr"])
            ) | (self.params["atr_min_filter_size"] == 0.0)
            df["atr_max_filter"] = (
                df["candle_size"] <= (self.params["atr_max_filter_size"] * df["atr"])
            ) | (self.params["atr_max_filter_size"] == 0.0)
            df["atr_filter"] = df["atr_min_filter"] & df["atr_max_filter"]

            # Check EMA filter
            df["ema_filter_long"] = (self.params["ema_filter"] == 0) | (
                df["close"] > df["ema"]
            )
            df["ema_filter_short"] = (self.params["ema_filter"] == 0) | (
                df["close"] < df["ema"]
            )

            # Identify patterns
            df["hammer"] = (
                (df["lowest_body"] >= df["bull_fib"])
                & df["atr_filter"]
                & (df["open"] != df["close"])
                & df["ema_filter_long"]
            )
            df["shooting_star"] = (
                (df["highest_body"] <= df["bear_fib"])
                & df["atr_filter"]
                & (df["open"] != df["close"])
                & df["ema_filter_short"]
            )

            # Calculate stop loss and take profit levels
            df["stop_size"] = df["atr"] * self.params["stop_multiplier"]

            # For long setups
            df["long_stop"] = df.apply(
                lambda row: (
                    min(row["low"], df["low"].shift(1).loc[row.name]) - row["stop_size"]
                    if not pd.isna(df["low"].shift(1).loc[row.name])
                    else row["low"] - row["stop_size"]
                ),
                axis=1,
            )
            df["long_stop_distance"] = df["close"] - df["long_stop"]
            df["long_target"] = df["close"] + (
                df["long_stop_distance"] * self.params["risk_reward"]
            )

            # For short setups
            df["short_stop"] = df.apply(
                lambda row: (
                    max(row["high"], df["high"].shift(1).loc[row.name])
                    + row["stop_size"]
                    if not pd.isna(df["high"].shift(1).loc[row.name])
                    else row["high"] + row["stop_size"]
                ),
                axis=1,
            )
            df["short_stop_distance"] = df["short_stop"] - df["close"]
            df["short_target"] = df["close"] - (
                df["short_stop_distance"] * self.params["risk_reward"]
            )

            # Generate signals
            df["long_signal"] = df["hammer"]
            df["short_signal"] = df["shooting_star"]

            # Generate trade details
            df["trade_direction"] = 0  # 0: no trade, 1: long, -1: short
            df.loc[df["long_signal"], "trade_direction"] = 1
            df.loc[df["short_signal"], "trade_direction"] = -1

            df["entry_price"] = df["close"]
            df["stop_loss"] = df.apply(
                lambda row: (
                    row["long_stop"]
                    if row["trade_direction"] == 1
                    else (
                        row["short_stop"]
                        if row["trade_direction"] == -1
                        else float("nan")
                    )
                ),
                axis=1,
            )
            df["take_profit"] = df.apply(
                lambda row: (
                    row["long_target"]
                    if row["trade_direction"] == 1
                    else (
                        row["short_target"]
                        if row["trade_direction"] == -1
                        else float("nan")
                    )
                ),
                axis=1,
            )

            return df

        except Exception as e:
            logger.error(f"Error analyzing market data: {str(e)}")
            raise StrategyError(f"Failed to analyze market data: {str(e)}")

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range.

        Args:
            df: DataFrame with OHLC data.
            period: ATR period.

        Returns:
            Series with ATR values.
        """
        high_low = df["high"] - df["low"]
        high_close = abs(df["high"] - df["close"].shift())
        low_close = abs(df["low"] - df["close"].shift())

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)

        atr = true_range.rolling(period).mean()
        return atr

    def backtest(
        self,
        df: pd.DataFrame,
        initial_capital: float = 10000.0,
        risk_per_trade_pct: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Run a backtest of the strategy on historical data.

        Args:
            df: DataFrame with OHLCV data.
            initial_capital: Starting capital amount.
            risk_per_trade_pct: Risk per trade as percentage of capital.

        Returns:
            Dictionary with backtest results and performance metrics.
        """
        try:
            # Analyze market data to get signals
            analysis_df = self.analyze_market_data(df.copy(), TimeFrame.HOURLY)

            # Initialize backtest variables
            balance = initial_capital
            max_balance = initial_capital
            equity_curve = [initial_capital]
            trades = []
            open_trade = None

            # Run through each bar
            for i, row in analysis_df.iterrows():
                # Check if we need to enter a new trade
                if open_trade is None and row["trade_direction"] != 0:
                    # Calculate position size based on risk
                    risk_amount = balance * (risk_per_trade_pct / 100)
                    if row["trade_direction"] == 1:  # Long
                        risk_pips = row["long_stop_distance"]
                    else:  # Short
                        risk_pips = row["short_stop_distance"]

                    if risk_pips <= 0:
                        continue  # Skip invalid setup

                    position_size = risk_amount / risk_pips

                    # Create trade record
                    open_trade = {
                        "entry_time": i,
                        "direction": "long" if row["trade_direction"] == 1 else "short",
                        "entry_price": row["entry_price"],
                        "stop_loss": row["stop_loss"],
                        "take_profit": row["take_profit"],
                        "position_size": position_size,
                        "risk_amount": risk_amount,
                    }

                # Check if we need to close an existing trade
                elif open_trade is not None:
                    # Check if stop loss or take profit hit
                    exit_price = None
                    exit_type = None

                    if open_trade["direction"] == "long":
                        # Check stop loss
                        if row["low"] <= open_trade["stop_loss"]:
                            exit_price = open_trade["stop_loss"]
                            exit_type = "stop_loss"
                        # Check take profit
                        elif row["high"] >= open_trade["take_profit"]:
                            exit_price = open_trade["take_profit"]
                            exit_type = "take_profit"
                    else:  # Short
                        # Check stop loss
                        if row["high"] >= open_trade["stop_loss"]:
                            exit_price = open_trade["stop_loss"]
                            exit_type = "stop_loss"
                        # Check take profit
                        elif row["low"] <= open_trade["take_profit"]:
                            exit_price = open_trade["take_profit"]
                            exit_type = "take_profit"

                    # Close the trade if exit conditions met
                    if exit_price is not None:
                        # Calculate profit/loss
                        if open_trade["direction"] == "long":
                            pips_gained = exit_price - open_trade["entry_price"]
                        else:
                            pips_gained = open_trade["entry_price"] - exit_price

                        profit_loss = pips_gained * open_trade["position_size"]

                        # Update balance
                        balance += profit_loss
                        max_balance = max(max_balance, balance)
                        equity_curve.append(balance)

                        # Complete trade record
                        open_trade["exit_time"] = i
                        open_trade["exit_price"] = exit_price
                        open_trade["exit_type"] = exit_type
                        open_trade["pips_gained"] = pips_gained
                        open_trade["profit_loss"] = profit_loss

                        # Add to trades list
                        trades.append(open_trade)

                        # Reset open trade
                        open_trade = None

            # Handle open trade at end of backtest
            if open_trade is not None:
                # Close at last price
                last_row = analysis_df.iloc[-1]

                if open_trade["direction"] == "long":
                    pips_gained = last_row["close"] - open_trade["entry_price"]
                else:
                    pips_gained = open_trade["entry_price"] - last_row["close"]

                profit_loss = pips_gained * open_trade["position_size"]

                # Update balance
                balance += profit_loss
                max_balance = max(max_balance, balance)
                equity_curve.append(balance)

                # Complete trade record
                open_trade["exit_time"] = analysis_df.index[-1]
                open_trade["exit_price"] = last_row["close"]
                open_trade["exit_type"] = "end_of_test"
                open_trade["pips_gained"] = pips_gained
                open_trade["profit_loss"] = profit_loss

                # Add to trades list
                trades.append(open_trade)

            # Calculate performance metrics
            total_trades = len(trades)
            winning_trades = sum(1 for t in trades if t["profit_loss"] > 0)
            losing_trades = sum(1 for t in trades if t["profit_loss"] <= 0)

            if total_trades > 0:
                win_rate = winning_trades / total_trades
            else:
                win_rate = 0

            total_profit = sum(t["profit_loss"] for t in trades if t["profit_loss"] > 0)
            total_loss = abs(
                sum(t["profit_loss"] for t in trades if t["profit_loss"] <= 0)
            )

            if total_loss > 0:
                profit_factor = total_profit / total_loss
            else:
                profit_factor = float("inf") if total_profit > 0 else 0

            net_profit = balance - initial_capital
            percent_return = (net_profit / initial_capital) * 100

            # Calculate max drawdown
            peak = equity_curve[0]
            drawdown = 0
            max_drawdown = 0

            for value in equity_curve:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)

            # Calculate Sharpe ratio
            if len(equity_curve) > 1:
                returns = [
                    (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
                    for i in range(1, len(equity_curve))
                ]
                if len(returns) > 0 and np.std(returns) > 0:
                    sharpe_ratio = (
                        np.mean(returns) / np.std(returns) * np.sqrt(252)
                    )  # Annualized
                else:
                    sharpe_ratio = 0
            else:
                sharpe_ratio = 0

            # Prepare result
            result = {
                "strategy_name": self.name,
                "initial_capital": initial_capital,
                "final_capital": balance,
                "net_profit": net_profit,
                "percent_return": percent_return,
                "max_drawdown": max_drawdown * 100,  # As percentage
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": win_rate * 100,  # As percentage
                "profit_factor": profit_factor,
                "sharpe_ratio": sharpe_ratio,
                "trades": trades,
                "equity_curve": equity_curve,
                "parameters": self.params,
            }

            return result

        except Exception as e:
            logger.error(f"Error in backtest: {str(e)}")
            raise StrategyError(f"Backtest failed: {str(e)}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert strategy to dictionary for serialization.

        Returns:
            Dictionary representation of the strategy.
        """
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "params": self.params,
            "type": "hammers_and_stars",
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HammersAndStarsStrategy":
        """
        Create strategy instance from dictionary.

        Args:
            data: Dictionary with strategy data.

        Returns:
            Strategy instance.
        """
        strategy = cls(params=data.get("params"))
        strategy.id = data.get("id", "hammers_and_stars")
        strategy.name = data.get("name", "Hammers & Stars Strategy")
        strategy.version = data.get("version", "1.0.0")
        strategy.description = data.get(
            "description", "Detects hammer and shooting star patterns"
        )
        strategy.author = data.get("author", "Forex AI System")

        return strategy

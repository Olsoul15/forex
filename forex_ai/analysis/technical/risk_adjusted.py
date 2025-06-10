"""
Risk-Adjusted Returns Analysis Module for the AI Forex Trading System.

This module implements various risk-adjusted performance metrics to evaluate
trading strategies, positions, and portfolios on a risk-normalized basis.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
from enum import Enum
from datetime import datetime
import math

from forex_ai.utils.logging import get_logger

logger = get_logger(__name__)


class PerformanceMetric(Enum):
    """Enumeration of risk-adjusted performance metrics."""

    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    INFORMATION_RATIO = "information_ratio"
    OMEGA_RATIO = "omega_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WINNING_PERCENTAGE = "winning_percentage"
    PROFIT_FACTOR = "profit_factor"
    EXPECTANCY = "expectancy"
    RISK_REWARD_RATIO = "risk_reward_ratio"


def calculate_returns(price_data: pd.DataFrame, method: str = "percent") -> pd.Series:
    """
    Calculate returns from price data.

    Args:
        price_data: DataFrame or Series with price data
        method: Method to calculate returns ('percent', 'log')

    Returns:
        Series with calculated returns
    """
    if isinstance(price_data, pd.DataFrame):
        if "close" in price_data.columns:
            prices = price_data["close"]
        else:
            prices = price_data.iloc[:, 0]  # Use first column
    else:
        prices = price_data

    if method == "percent":
        returns = prices.pct_change().dropna()
    elif method == "log":
        returns = np.log(prices / prices.shift(1)).dropna()
    else:
        raise ValueError(f"Unknown returns calculation method: {method}")

    return returns


def calculate_drawdowns(returns: pd.Series) -> pd.Series:
    """
    Calculate drawdowns from returns.

    Args:
        returns: Series of returns

    Returns:
        Series with drawdowns
    """
    # Calculate cumulative returns
    cum_returns = (1 + returns).cumprod()

    # Calculate running maximum
    running_max = cum_returns.cummax()

    # Calculate drawdowns
    drawdowns = (cum_returns / running_max) - 1

    return drawdowns


def max_drawdown(returns: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    """
    Calculate maximum drawdown and its start/end dates.

    Args:
        returns: Series of returns

    Returns:
        Tuple of (max_drawdown, peak_date, valley_date)
    """
    # Calculate cumulative returns
    cum_returns = (1 + returns).cumprod()

    # Calculate running maximum
    running_max = cum_returns.cummax()

    # Calculate drawdowns
    drawdowns = (cum_returns / running_max) - 1

    # Find maximum drawdown and its index
    max_dd = drawdowns.min()
    valley_idx = drawdowns.idxmin()

    # Find the preceding peak
    peak_idx = running_max.loc[:valley_idx].idxmax()

    return max_dd, peak_idx, valley_idx


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    period: str = "daily",
    annualization: Optional[float] = None,
) -> float:
    """
    Calculate Sharpe ratio.

    Args:
        returns: Series of returns
        risk_free_rate: Risk-free rate (annualized)
        period: Period of returns ('daily', 'weekly', 'monthly')
        annualization: Annualization factor (if None, inferred from period)

    Returns:
        Sharpe ratio
    """
    if annualization is None:
        if period == "daily":
            annualization = 252
        elif period == "weekly":
            annualization = 52
        elif period == "monthly":
            annualization = 12
        else:
            raise ValueError(f"Unknown period: {period}")

    # Adjust risk-free rate to match period
    period_rf = ((1 + risk_free_rate) ** (1 / annualization)) - 1

    # Calculate excess returns
    excess_returns = returns - period_rf

    # Calculate Sharpe ratio
    sharpe = excess_returns.mean() / excess_returns.std()

    # Annualize Sharpe ratio
    annualized_sharpe = sharpe * np.sqrt(annualization)

    return annualized_sharpe


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    period: str = "daily",
    annualization: Optional[float] = None,
    target_return: float = 0.0,
) -> float:
    """
    Calculate Sortino ratio (penalizes only downside volatility).

    Args:
        returns: Series of returns
        risk_free_rate: Risk-free rate (annualized)
        period: Period of returns ('daily', 'weekly', 'monthly')
        annualization: Annualization factor (if None, inferred from period)
        target_return: Minimum acceptable return

    Returns:
        Sortino ratio
    """
    if annualization is None:
        if period == "daily":
            annualization = 252
        elif period == "weekly":
            annualization = 52
        elif period == "monthly":
            annualization = 12
        else:
            raise ValueError(f"Unknown period: {period}")

    # Adjust risk-free rate to match period
    period_rf = ((1 + risk_free_rate) ** (1 / annualization)) - 1

    # Calculate excess returns
    excess_returns = returns - period_rf

    # Calculate downside deviation (only returns below target)
    downside_returns = excess_returns[excess_returns < target_return]
    if len(downside_returns) == 0:
        return float("inf")  # No downside returns

    downside_deviation = np.sqrt(
        ((downside_returns - target_return) ** 2).sum() / len(excess_returns)
    )

    if downside_deviation == 0:
        return float("inf")  # No downside deviation

    # Calculate Sortino ratio
    sortino = (excess_returns.mean() - target_return) / downside_deviation

    # Annualize Sortino ratio
    annualized_sortino = sortino * np.sqrt(annualization)

    return annualized_sortino


def calmar_ratio(
    returns: pd.Series, period: str = "daily", annualization: Optional[float] = None
) -> float:
    """
    Calculate Calmar ratio (annualized return / maximum drawdown).

    Args:
        returns: Series of returns
        period: Period of returns ('daily', 'weekly', 'monthly')
        annualization: Annualization factor (if None, inferred from period)

    Returns:
        Calmar ratio
    """
    if annualization is None:
        if period == "daily":
            annualization = 252
        elif period == "weekly":
            annualization = 52
        elif period == "monthly":
            annualization = 12
        else:
            raise ValueError(f"Unknown period: {period}")

    # Calculate annualized return
    ann_return = ((1 + returns.mean()) ** annualization) - 1

    # Calculate maximum drawdown
    max_dd, _, _ = max_drawdown(returns)

    if max_dd == 0:
        return float("inf")  # No drawdown

    # Calculate Calmar ratio
    calmar = ann_return / abs(max_dd)

    return calmar


def omega_ratio(
    returns: pd.Series,
    threshold: float = 0.0,
    period: str = "daily",
    annualization: Optional[float] = None,
) -> float:
    """
    Calculate Omega ratio (probability-weighted ratio of gains vs. losses).

    Args:
        returns: Series of returns
        threshold: Return threshold
        period: Period of returns ('daily', 'weekly', 'monthly')
        annualization: Annualization factor (if None, inferred from period)

    Returns:
        Omega ratio
    """
    if annualization is None:
        if period == "daily":
            annualization = 252
        elif period == "weekly":
            annualization = 52
        elif period == "monthly":
            annualization = 12
        else:
            raise ValueError(f"Unknown period: {period}")

    # Adjust threshold to match period
    period_threshold = ((1 + threshold) ** (1 / annualization)) - 1

    # Calculate excess returns over threshold
    excess_returns = returns - period_threshold

    # Separate positive and negative excess returns
    positive_excess = excess_returns[excess_returns > 0]
    negative_excess = excess_returns[excess_returns < 0]

    if len(negative_excess) == 0:
        return float("inf")  # No negative excess returns

    # Calculate Omega ratio
    omega = positive_excess.sum() / abs(negative_excess.sum())

    return omega


def information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    period: str = "daily",
    annualization: Optional[float] = None,
) -> float:
    """
    Calculate Information ratio (excess return over benchmark / tracking error).

    Args:
        returns: Series of returns
        benchmark_returns: Series of benchmark returns
        period: Period of returns ('daily', 'weekly', 'monthly')
        annualization: Annualization factor (if None, inferred from period)

    Returns:
        Information ratio
    """
    if annualization is None:
        if period == "daily":
            annualization = 252
        elif period == "weekly":
            annualization = 52
        elif period == "monthly":
            annualization = 12
        else:
            raise ValueError(f"Unknown period: {period}")

    # Align returns and benchmark returns
    returns, benchmark_returns = returns.align(benchmark_returns, join="inner")

    # Calculate excess returns over benchmark
    excess_returns = returns - benchmark_returns

    # Calculate tracking error (standard deviation of excess returns)
    tracking_error = excess_returns.std()

    if tracking_error == 0:
        return 0  # No tracking error

    # Calculate Information ratio
    info_ratio = excess_returns.mean() / tracking_error

    # Annualize Information ratio
    annualized_ir = info_ratio * np.sqrt(annualization)

    return annualized_ir


def winning_percentage(returns: pd.Series) -> float:
    """
    Calculate the percentage of winning periods.

    Args:
        returns: Series of returns

    Returns:
        Winning percentage (0.0 to 1.0)
    """
    if len(returns) == 0:
        return 0.0

    wins = (returns > 0).sum()
    total = len(returns)

    return wins / total


def profit_factor(returns: pd.Series) -> float:
    """
    Calculate profit factor (gross profit / gross loss).

    Args:
        returns: Series of returns

    Returns:
        Profit factor
    """
    positive_returns = returns[returns > 0]
    negative_returns = returns[returns < 0]

    gross_profit = positive_returns.sum()
    gross_loss = abs(negative_returns.sum())

    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0

    return gross_profit / gross_loss


def expectancy(returns: pd.Series) -> float:
    """
    Calculate expectancy (average return per trade).

    Args:
        returns: Series of returns

    Returns:
        Expectancy
    """
    if len(returns) == 0:
        return 0.0

    win_rate = winning_percentage(returns)

    # Average win and loss
    avg_win = returns[returns > 0].mean() if any(returns > 0) else 0
    avg_loss = abs(returns[returns < 0].mean()) if any(returns < 0) else 0

    if avg_loss == 0:
        return avg_win if win_rate > 0 else 0.0

    # Calculate win/loss ratio
    win_loss_ratio = avg_win / avg_loss

    # Calculate expectancy
    expectancy = (win_rate * win_loss_ratio) - (1 - win_rate)

    return expectancy


def risk_reward_ratio(returns: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate rolling risk-reward ratio (average gain / average loss).

    Args:
        returns: Series of returns
        window: Rolling window size

    Returns:
        Series with rolling risk-reward ratio
    """
    # Rolling average of positive returns
    rolling_gains = returns[returns > 0].rolling(window=window).mean()

    # Rolling average of negative returns (absolute value)
    rolling_losses = returns[returns < 0].abs().rolling(window=window).mean()

    # Calculate risk-reward ratio
    risk_reward = rolling_gains / rolling_losses

    return risk_reward


class RiskAdjustedReturns:
    """Class for calculating risk-adjusted performance metrics."""

    def __init__(self, risk_free_rate: float = 0.0, period: str = "daily"):
        """
        Initialize RiskAdjustedReturns calculator.

        Args:
            risk_free_rate: Annualized risk-free rate
            period: Period of returns ('daily', 'weekly', 'monthly')
        """
        self.risk_free_rate = risk_free_rate
        self.period = period

        # Set annualization factor
        if period == "daily":
            self.annualization = 252
        elif period == "weekly":
            self.annualization = 52
        elif period == "monthly":
            self.annualization = 12
        else:
            raise ValueError(f"Unknown period: {period}")

    def calculate_metrics(
        self, returns: pd.Series, benchmark_returns: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Calculate all risk-adjusted performance metrics.

        Args:
            returns: Series of returns
            benchmark_returns: Optional benchmark returns for comparative metrics

        Returns:
            Dictionary with calculated metrics
        """
        metrics = {}

        try:
            # Basic metrics
            metrics["total_return"] = (1 + returns).prod() - 1
            metrics["annualized_return"] = (
                1 + returns.mean()
            ) ** self.annualization - 1
            metrics["volatility"] = returns.std() * np.sqrt(self.annualization)

            # Calculate drawdowns
            dd, peak_date, valley_date = max_drawdown(returns)
            metrics["max_drawdown"] = dd
            metrics["max_drawdown_peak_date"] = peak_date
            metrics["max_drawdown_valley_date"] = valley_date

            # Risk-adjusted metrics
            metrics["sharpe_ratio"] = sharpe_ratio(
                returns, self.risk_free_rate, self.period, self.annualization
            )

            metrics["sortino_ratio"] = sortino_ratio(
                returns, self.risk_free_rate, self.period, self.annualization
            )

            metrics["calmar_ratio"] = calmar_ratio(
                returns, self.period, self.annualization
            )

            metrics["omega_ratio"] = omega_ratio(
                returns, 0.0, self.period, self.annualization
            )

            # Trading metrics
            metrics["winning_percentage"] = winning_percentage(returns)
            metrics["profit_factor"] = profit_factor(returns)
            metrics["expectancy"] = expectancy(returns)

            # If benchmark returns provided, calculate comparative metrics
            if benchmark_returns is not None:
                metrics["information_ratio"] = information_ratio(
                    returns, benchmark_returns, self.period, self.annualization
                )

                # Calculate beta and alpha
                # Align returns and benchmark returns
                aligned_returns, aligned_benchmark = returns.align(
                    benchmark_returns, join="inner"
                )

                # Calculate beta (covariance / benchmark variance)
                covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
                benchmark_variance = aligned_benchmark.var()
                metrics["beta"] = (
                    covariance / benchmark_variance if benchmark_variance != 0 else 1.0
                )

                # Calculate alpha (annualized)
                period_rf = ((1 + self.risk_free_rate) ** (1 / self.annualization)) - 1
                metrics["alpha"] = (
                    aligned_returns.mean()
                    - period_rf
                    - metrics["beta"] * (aligned_benchmark.mean() - period_rf)
                ) * self.annualization

        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            # Try to salvage any calculated metrics

        return metrics

    def evaluate_strategy(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        capital: float = 10000.0,
    ) -> Dict[str, Any]:
        """
        Comprehensive strategy evaluation with metrics and insights.

        Args:
            strategy_returns: Series of strategy returns
            benchmark_returns: Optional benchmark returns
            capital: Initial capital for calculating absolute metrics

        Returns:
            Dictionary with evaluation results and insights
        """
        # Calculate metrics
        metrics = self.calculate_metrics(strategy_returns, benchmark_returns)

        # Calculate absolute performance
        performance = {}
        performance["initial_capital"] = capital
        performance["final_capital"] = capital * (1 + metrics["total_return"])
        performance["absolute_return"] = performance["final_capital"] - capital
        performance["max_drawdown_amount"] = capital * abs(metrics["max_drawdown"])

        # Calculate drawdown recovery time if ended
        if metrics["max_drawdown_valley_date"] < strategy_returns.index[-1]:
            # Get cumulative returns after valley
            cum_returns_after_valley = (
                1 + strategy_returns.loc[metrics["max_drawdown_valley_date"] :]
            ).cumprod()

            # Find when cumulative returns exceed the value at the previous peak
            peak_value = (
                (1 + strategy_returns.loc[: metrics["max_drawdown_peak_date"]])
                .cumprod()
                .iloc[-1]
            )
            recovery_threshold = peak_value / cum_returns_after_valley.iloc[0]

            recovery_dates = cum_returns_after_valley[
                cum_returns_after_valley >= recovery_threshold
            ].index

            if len(recovery_dates) > 0:
                recovery_date = recovery_dates[0]
                recovery_time = recovery_date - metrics["max_drawdown_valley_date"]
                performance["drawdown_recovery_time_days"] = recovery_time.days
            else:
                performance["drawdown_recovery_time_days"] = None  # Not yet recovered
        else:
            performance["drawdown_recovery_time_days"] = None  # Still in drawdown

        # Prepare insights
        insights = {}

        # Overall assessment
        if metrics["sharpe_ratio"] > 1.0:
            if metrics["max_drawdown"] > -0.2:
                insights["overall_assessment"] = (
                    "Good risk-adjusted performance with reasonable drawdowns."
                )
            else:
                insights["overall_assessment"] = (
                    "Good returns but with significant drawdowns. Consider risk controls."
                )
        elif metrics["sharpe_ratio"] > 0.5:
            insights["overall_assessment"] = "Moderate risk-adjusted performance."
        else:
            insights["overall_assessment"] = (
                "Poor risk-adjusted performance. Re-evaluate strategy."
            )

        # Risk assessment
        risk_level = "low"
        if metrics["volatility"] > 0.25 or abs(metrics["max_drawdown"]) > 0.25:
            risk_level = "high"
        elif metrics["volatility"] > 0.15 or abs(metrics["max_drawdown"]) > 0.15:
            risk_level = "medium"

        insights["risk_assessment"] = f"{risk_level.capitalize()} risk strategy."

        # Benchmark comparison if available
        if benchmark_returns is not None:
            benchmark_return = (1 + benchmark_returns).prod() - 1

            if metrics["total_return"] > benchmark_return:
                outperformance = metrics["total_return"] - benchmark_return
                insights["benchmark_comparison"] = (
                    f"Strategy outperformed benchmark by {outperformance:.2%}."
                )

                if metrics["information_ratio"] < 0.3:
                    insights[
                        "benchmark_comparison"
                    ] += " However, risk-adjusted outperformance is weak."
            else:
                underperformance = benchmark_return - metrics["total_return"]
                insights["benchmark_comparison"] = (
                    f"Strategy underperformed benchmark by {underperformance:.2%}."
                )

        # Return combined results
        return {
            "metrics": metrics,
            "performance": performance,
            "insights": insights,
            "evaluation_date": datetime.now().isoformat(),
        }

    def backtest_comparison(
        self,
        strategies_returns: Dict[str, pd.Series],
        benchmark_returns: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Compare multiple strategy backtests.

        Args:
            strategies_returns: Dictionary mapping strategy names to return series
            benchmark_returns: Optional benchmark returns

        Returns:
            Dictionary with comparison results
        """
        comparison = {
            "strategies": {},
            "ranking": {},
            "best_overall": None,
            "best_risk_adjusted": None,
            "comparison_date": datetime.now().isoformat(),
        }

        # Calculate metrics for each strategy
        for strategy_name, returns in strategies_returns.items():
            comparison["strategies"][strategy_name] = self.calculate_metrics(
                returns, benchmark_returns
            )

        # Calculate metrics for benchmark if provided
        if benchmark_returns is not None:
            comparison["benchmark"] = self.calculate_metrics(benchmark_returns)

        # Rank strategies by different metrics
        metrics_to_rank = [
            "total_return",
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
            "max_drawdown",
            "winning_percentage",
            "profit_factor",
        ]

        for metric in metrics_to_rank:
            # For drawdown, lower (less negative) is better
            reverse = metric != "max_drawdown"

            ranked_strategies = sorted(
                strategies_returns.keys(),
                key=lambda s: comparison["strategies"][s].get(
                    metric, -float("inf") if reverse else float("inf")
                ),
                reverse=reverse,
            )

            comparison["ranking"][metric] = ranked_strategies

        # Determine best overall strategy (highest Sharpe ratio)
        if comparison["ranking"].get("sharpe_ratio"):
            comparison["best_risk_adjusted"] = comparison["ranking"]["sharpe_ratio"][0]

        # Determine best absolute return strategy
        if comparison["ranking"].get("total_return"):
            comparison["best_overall"] = comparison["ranking"]["total_return"][0]

        return comparison


def analyze_trades(trades: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze a set of trades with risk-adjusted metrics.

    Args:
        trades: DataFrame with trade data (must have 'entry_price', 'exit_price',
               'direction', 'entry_time', 'exit_time', 'profit_loss' columns)

    Returns:
        Dictionary with trade analysis
    """
    if trades.empty:
        return {"error": "No trades to analyze"}

    required_columns = [
        "entry_price",
        "exit_price",
        "direction",
        "entry_time",
        "exit_time",
        "profit_loss",
    ]

    missing_columns = [col for col in required_columns if col not in trades.columns]
    if missing_columns:
        return {"error": f"Missing required columns: {missing_columns}"}

    # Basic trade statistics
    total_trades = len(trades)
    winning_trades = (trades["profit_loss"] > 0).sum()
    losing_trades = (trades["profit_loss"] < 0).sum()
    breakeven_trades = total_trades - winning_trades - losing_trades

    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    # Calculate returns
    total_profit = trades[trades["profit_loss"] > 0]["profit_loss"].sum()
    total_loss = trades[trades["profit_loss"] < 0]["profit_loss"].sum()

    net_profit = total_profit + total_loss

    # Calculate average metrics
    avg_profit = total_profit / winning_trades if winning_trades > 0 else 0
    avg_loss = total_loss / losing_trades if losing_trades > 0 else 0

    profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float("inf")

    # Calculate drawdown
    cumulative_pl = trades["profit_loss"].cumsum()
    peak = cumulative_pl.cummax()
    drawdown = (cumulative_pl - peak) / peak if peak.max() > 0 else cumulative_pl - peak
    max_dd = drawdown.min()

    # Calculate time-based metrics
    trades["duration"] = (
        trades["exit_time"] - trades["entry_time"]
    ).dt.total_seconds() / 3600  # hours

    avg_trade_duration = trades["duration"].mean()
    avg_win_duration = (
        trades[trades["profit_loss"] > 0]["duration"].mean()
        if winning_trades > 0
        else 0
    )
    avg_loss_duration = (
        trades[trades["profit_loss"] < 0]["duration"].mean() if losing_trades > 0 else 0
    )

    # Calculate metrics by direction
    long_trades = trades[trades["direction"] == "long"]
    short_trades = trades[trades["direction"] == "short"]

    long_win_rate = (
        (long_trades["profit_loss"] > 0).sum() / len(long_trades)
        if len(long_trades) > 0
        else 0
    )
    short_win_rate = (
        (short_trades["profit_loss"] > 0).sum() / len(short_trades)
        if len(short_trades) > 0
        else 0
    )

    # Calculate expectancy
    expectancy_value = (
        (win_rate * avg_profit) - ((1 - win_rate) * abs(avg_loss))
        if avg_loss != 0
        else 0
    )

    # Assemble results
    return {
        "trade_statistics": {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "breakeven_trades": breakeven_trades,
            "win_rate": win_rate,
            "net_profit": net_profit,
            "total_profit": total_profit,
            "total_loss": total_loss,
            "profit_factor": profit_factor,
            "max_drawdown": max_dd,
        },
        "average_metrics": {
            "avg_profit": avg_profit,
            "avg_loss": avg_loss,
            "avg_trade_duration_hours": avg_trade_duration,
            "avg_win_duration_hours": avg_win_duration,
            "avg_loss_duration_hours": avg_loss_duration,
            "risk_reward_ratio": (
                abs(avg_profit / avg_loss) if avg_loss != 0 else float("inf")
            ),
            "expectancy": expectancy_value,
        },
        "direction_analysis": {
            "long_trades": len(long_trades),
            "short_trades": len(short_trades),
            "long_win_rate": long_win_rate,
            "short_win_rate": short_win_rate,
            "long_profit": long_trades["profit_loss"].sum(),
            "short_profit": short_trades["profit_loss"].sum(),
        },
    }


def calculate_optimal_position_size(
    account_balance: float,
    risk_per_trade: float,
    stop_loss_pips: float,
    pip_value: float,
) -> float:
    """
    Calculate optimal position size based on risk management parameters.

    Args:
        account_balance: Account balance in account currency
        risk_per_trade: Percentage of account to risk per trade (e.g., 0.02 for 2%)
        stop_loss_pips: Stop loss distance in pips
        pip_value: Value of 1 pip in account currency

    Returns:
        Position size (lot size)
    """
    if stop_loss_pips <= 0 or pip_value <= 0:
        return 0.0

    # Calculate risk amount
    risk_amount = account_balance * risk_per_trade

    # Calculate position size
    position_size = risk_amount / (stop_loss_pips * pip_value)

    # Round to standard lot sizes
    # For forex, standard lot sizes are typically 0.01, 0.1, 1.0
    standard_lot = 1.0
    mini_lot = 0.1
    micro_lot = 0.01

    if position_size >= standard_lot:
        return math.floor(position_size * 100) / 100  # Round down to 2 decimal places
    elif position_size >= mini_lot:
        return math.floor(position_size * 100) / 100
    elif position_size >= micro_lot:
        return math.floor(position_size * 100) / 100
    else:
        return micro_lot  # Minimum position size

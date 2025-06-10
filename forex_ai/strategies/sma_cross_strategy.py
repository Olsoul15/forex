import pandas as pd
import pandas_ta as ta # Import pandas-ta
from typing import Dict, Any

from .strategy import BaseStrategy, TradingSignal

class SmaCrossStrategy(BaseStrategy):
    \"\"\"
    A simple Moving Average Crossover strategy.

    Generates BUY signals when the fast MA crosses above the slow MA.
    Generates SELL signals when the fast MA crosses below the slow MA.
    Requires 'fast_period' and 'slow_period' in parameters.
    \"\"\"

    def _validate_parameters(self):
        \"\"\"
        Validate parameters for SMA Crossover.
        \"\"\"
        required = ['fast_period', 'slow_period']
        for param in required:
            if param not in self.parameters:
                raise ValueError(f"Missing required parameter '{param}' for SmaCrossStrategy")
            if not isinstance(self.parameters[param], int) or self.parameters[param] <= 0:
                raise ValueError(f"Parameter '{param}' must be a positive integer.")

        if self.parameters['fast_period'] >= self.parameters['slow_period']:
            raise ValueError("'fast_period' must be less than 'slow_period'")

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        \"\"\"
        Generate SMA Crossover signals.
        \"\"\"
        if not isinstance(data, pd.DataFrame) or data.empty:
            return pd.Series(dtype=int)
        if 'close' not in data.columns:
            raise ValueError("Dataframe must contain 'close' column for SMA calculation.")

        fast_period = self.get_parameter('fast_period')
        slow_period = self.get_parameter('slow_period')

        # Calculate SMAs using pandas_ta
        # Ensure the strategy receives enough data for the longest period
        data.ta.sma(length=fast_period, append=True) # Appends column like SMA_10
        data.ta.sma(length=slow_period, append=True) # Appends column like SMA_30

        fast_sma_col = f'SMA_{fast_period}'
        slow_sma_col = f'SMA_{slow_period}'

        # Check if SMA columns were added (pandas-ta might fail silently on very short data)
        if fast_sma_col not in data.columns or slow_sma_col not in data.columns:
             # Not enough data to calculate indicators for the given periods
             return pd.Series(index=data.index, dtype=int, data=TradingSignal.HOLD)

        # Initialize signals Series
        signals = pd.Series(index=data.index, dtype=int, data=TradingSignal.HOLD)

        # Determine crossover points
        # Condition for fast MA crossing above slow MA
        buy_condition = (data[fast_sma_col].shift(1) < data[slow_sma_col].shift(1)) & \
                        (data[fast_sma_col] > data[slow_sma_col])

        # Condition for fast MA crossing below slow MA
        sell_condition = (data[fast_sma_col].shift(1) > data[slow_sma_col].shift(1)) & \
                         (data[fast_sma_col] < data[slow_sma_col])

        # Apply signals based on conditions
        signals[buy_condition] = TradingSignal.BUY
        signals[sell_condition] = TradingSignal.SELL

        # Clean up added columns? Optional, depends if engine needs them later
        # data.drop(columns=[fast_sma_col, slow_sma_col], inplace=True)

        return signals 
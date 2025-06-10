from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Optional


# Define potential signal types
class TradingSignal:
    HOLD = 0
    BUY = 1
    SELL = -1
    CLOSE = 2  # Signal to close existing position


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.

    Subclasses must implement the generate_signals method.
    """

    def __init__(self, parameters: Dict[str, Any]):
        """
        Initialize the strategy with specific parameters.

        Args:
            parameters: A dictionary containing strategy-specific parameters
                        extracted from the main strategy configuration.
        """
        self.parameters = parameters
        self._validate_parameters()

    @abstractmethod
    def _validate_parameters(self):
        """
        Validate the necessary parameters are present and valid.
        Should be implemented by subclasses. Raise ValueError if invalid.
        """
        pass

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on the provided historical data.

        Args:
            data: A pandas DataFrame containing OHLCV data, potentially with
                  pre-calculated indicators if the engine adds them, or the
                  strategy calculates them internally. The index should be datetime.

        Returns:
            A pandas Series with the same index as the input data, containing
            trading signals (e.g., TradingSignal.BUY, TradingSignal.SELL,
            TradingSignal.HOLD, TradingSignal.CLOSE).
        """
        pass

    def get_parameter(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Helper method to safely get a parameter value.
        """
        return self.parameters.get(key, default)


# Example usage (in subclasses):
# class MyStrategy(BaseStrategy):
#     def _validate_parameters(self):
#         if 'period' not in self.parameters:
#             raise ValueError("Missing 'period' parameter for MyStrategy")
#
#     def generate_signals(self, data: pd.DataFrame) -> pd.Series:
#         period = self.get_parameter('period')
#         # ... calculate indicators using data and period ...
#         signals = pd.Series(index=data.index, dtype=int, data=TradingSignal.HOLD)
#         # ... logic to set BUY/SELL signals based on indicators ...
#         return signals

"""
Base classes and utilities for data connectors.
"""

import abc
from typing import Any, Dict, List, Optional
import pandas as pd
from forex_ai.custom_types import CurrencyPair, TimeFrame

class DataConnector(abc.ABC):
    """Abstract base class for all data connectors."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @abc.abstractmethod
    def connect(self):
        """Establish connection to the data source."""
        raise NotImplementedError

    @abc.abstractmethod
    def disconnect(self):
        """Disconnect from the data source."""
        raise NotImplementedError

    @abc.abstractmethod
    def fetch_historical_data(
        self,
        currency_pair: CurrencyPair,
        timeframe: TimeFrame,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data."""
        raise NotImplementedError

    # Add other common methods needed by connectors if known
    # e.g., fetch_realtime_tick, subscribe_to_stream, etc. 
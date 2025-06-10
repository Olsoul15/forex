# conftest.py for integration tests

import pytest
from unittest.mock import MagicMock
import pandas as pd

# TODO: Add shared fixtures here, for example:
# - A fixture for a default AnalysisConfig object.
# - A fixture for a mock MarketDataFetcher.
# - A fixture for a sample AutoAgentOrchestrator instance. 

@pytest.fixture
def mock_analysis_config():
    """Provides a mock AnalysisConfig object."""
    config = MagicMock()
    config.timeframes = ["H1", "H4"]
    config.indicators = ["RSI", "MACD"]
    config.lookback_periods = 100
    return config

@pytest.fixture
def mock_data_fetcher():
    """Provides a mock MarketDataFetcher instance."""
    fetcher = MagicMock()
    
    # Create a realistic-looking mock candle
    mock_candle = {
        "time": "2024-01-01T00:00:00Z",
        "volume": 100,
        "mid": {"o": "1.1000", "h": "1.1050", "l": "1.0990", "c": "1.1030"}
    }
    
    # Mock the get_candles method
    fetcher.get_candles.return_value = {
        "instrument": "EUR_USD",
        "granularity": "H1",
        "candles": [mock_candle for _ in range(100)] # Return 100 candles
    }
    return fetcher

@pytest.fixture
def mock_orchestrator_config(mock_analysis_config, mock_data_fetcher):
    """Provides a configuration dictionary for the orchestrator."""
    return {
        "config": mock_analysis_config,
        "data_fetcher": mock_data_fetcher
    } 
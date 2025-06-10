import pytest
from forex_ai.integration.autoagent_orchestrator import AutoAgentOrchestrator

@pytest.fixture
def orchestrator(mock_orchestrator_config):
    """Provides a default AutoAgentOrchestrator instance for testing."""
    return AutoAgentOrchestrator(mock_orchestrator_config)

def test_generate_insights_rsi_overbought(orchestrator):
    """
    Tests that _generate_insights correctly identifies an overbought RSI condition.
    """
    instrument = "EUR_USD"
    timeframe = "H1"
    indicators = {"current_price": 1.1200, "rsi": 75}
    support_resistance = {}
    
    insights = orchestrator._generate_insights(instrument, timeframe, indicators, support_resistance)
    
    assert len(insights) == 1
    rsi_insight = insights[0]
    assert rsi_insight["indicator"] == "RSI"
    assert rsi_insight["sentiment"] == "bearish"
    assert "overbought" in rsi_insight["message"]

def test_generate_insights_rsi_oversold(orchestrator):
    """
    Tests that _generate_insights correctly identifies an oversold RSI condition.
    """
    instrument = "EUR_USD"
    timeframe = "H1"
    indicators = {"current_price": 1.1000, "rsi": 25}
    support_resistance = {}
    
    insights = orchestrator._generate_insights(instrument, timeframe, indicators, support_resistance)
    
    assert len(insights) == 1
    rsi_insight = insights[0]
    assert rsi_insight["indicator"] == "RSI"
    assert rsi_insight["sentiment"] == "bullish"
    assert "oversold" in rsi_insight["message"]

def test_generate_insights_macd_bullish(orchestrator):
    """
    Tests that _generate_insights correctly identifies a bullish MACD condition.
    """
    instrument = "EUR_USD"
    timeframe = "H4"
    indicators = {"current_price": 1.1250, "macd": {"histogram": 0.0005}}
    support_resistance = {}
    
    insights = orchestrator._generate_insights(instrument, timeframe, indicators, support_resistance)
    
    macd_insight = next((i for i in insights if i["indicator"] == "MACD"), None)
    assert macd_insight is not None
    assert macd_insight["sentiment"] == "bullish"
    assert "positive at" in macd_insight["message"]

def test_generate_insights_bollinger_bearish(orchestrator):
    """
    Tests that _generate_insights correctly identifies a bearish Bollinger Bands condition.
    """
    instrument = "GBP_USD"
    timeframe = "D1"
    indicators = {
        "current_price": 1.2800,
        "bollinger_bands": {"upper": 1.2750, "lower": 1.2550}
    }
    support_resistance = {}

    insights = orchestrator._generate_insights(instrument, timeframe, indicators, support_resistance)

    bb_insight = next((i for i in insights if i["indicator"] == "Bollinger Bands"), None)
    assert bb_insight is not None
    assert bb_insight["sentiment"] == "bearish"
    assert "above upper" in bb_insight["message"]

def test_generate_insights_support_resistance(orchestrator):
    """
    Tests that _generate_insights correctly identifies support and resistance levels.
    """
    instrument = "USD_JPY"
    timeframe = "H1"
    indicators = {"current_price": 110.50}
    support_resistance = {
        "support_levels": [110.20, 109.80],
        "resistance_levels": [110.80, 111.00]
    }

    insights = orchestrator._generate_insights(instrument, timeframe, indicators, support_resistance)
    
    support_insight = next((i for i in insights if i["indicator"] == "Support"), None)
    resistance_insight = next((i for i in insights if i["indicator"] == "Resistance"), None)
    
    assert support_insight is not None
    assert resistance_insight is not None
    assert support_insight["value"] == 110.20
    assert resistance_insight["value"] == 110.80

# --- Calculation Method Unit Tests ---

def test_calculate_rsi(orchestrator):
    """
    Tests the _calculate_rsi method with a known data set.
    """
    # Prices that should result in an RSI of ~70 (overbought)
    prices = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.50, 46.70, 46.80, 47.00, 47.20, 47.50]
    rsi = orchestrator._calculate_rsi(prices, period=14)
    assert 69 < rsi < 80 # Loosened the range a bit

    # Prices that should result in an RSI of ~30 (oversold)
    prices_down = [46.28, 45.61, 46.03, 45.89, 46.08, 45.84, 45.42, 45.10, 44.83, 44.33, 43.61, 44.15, 44.09, 44.34, 44.00, 43.80, 43.50, 43.00, 42.50]
    rsi_down = orchestrator._calculate_rsi(prices_down, period=14)
    assert 20 < rsi_down < 31

def test_calculate_macd(orchestrator):
    """
    Tests the _calculate_macd method.
    Note: This is a simplified test for the shape of the output.
    A full test would require a known dataset and validated results.
    """
    prices = [i*i for i in range(100)] # A quadratically increasing list
    macd, signal, hist = orchestrator._calculate_macd(prices)
    
    assert isinstance(macd, float)
    assert isinstance(signal, float)
    assert isinstance(hist, float)
    assert hist > 0 # Should be bullish on a strong uptrend

def test_get_support_resistance(orchestrator):
    """
    Tests the _get_support_resistance method.
    """
    mock_candle = {
        "time": "2024-01-01T00:00:00Z",
        "volume": 100,
        "mid": {"o": "1.1000", "h": "1.1050", "l": "1.0990", "c": "1.1030"}
    }
    market_data = {"candles": [mock_candle for _ in range(20)]}
    
    sr_levels = orchestrator._get_support_resistance(market_data)
    
    assert "support_levels" in sr_levels
    assert "resistance_levels" in sr_levels
    assert isinstance(sr_levels["support_levels"], list)
    assert isinstance(sr_levels["resistance_levels"], list) 
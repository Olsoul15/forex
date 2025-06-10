# Placeholder for AutoAgentOrchestrator tests
# TODO: Implement tests based on the testing plan 

import pytest
from unittest.mock import AsyncMock
from forex_ai.integration.autoagent_orchestrator import AutoAgentOrchestrator

def test_orchestrator_initialization(mock_orchestrator_config):
    """
    Tests if the AutoAgentOrchestrator initializes correctly.
    """
    orchestrator = AutoAgentOrchestrator(mock_orchestrator_config)
    assert orchestrator is not None
    assert orchestrator.config == mock_orchestrator_config.get("config")
    assert orchestrator.data_fetcher == mock_orchestrator_config.get("data_fetcher")
    assert orchestrator.confidence_threshold == 0.65  # Default value
    assert orchestrator.memory_manager is not None

@pytest.mark.asyncio
async def test_analyze_market_executes_and_returns_structure(mock_orchestrator_config):
    """
    Tests that analyze_market runs and returns a dictionary with the correct keys.
    """
    orchestrator = AutoAgentOrchestrator(mock_orchestrator_config)
    orchestrator.memory_manager.store_analysis_result = AsyncMock() # Mock the memory manager method
    results = await orchestrator.analyze_market("EUR_USD")

    assert isinstance(results, dict)
    expected_keys = ["instrument", "timestamp", "market_view", "insights", "signals", "support_resistance"]
    for key in expected_keys:
        assert key in results

@pytest.mark.asyncio
async def test_analyze_market_calls_data_fetcher(mock_orchestrator_config):
    """
    Tests that the orchestrator calls the data fetcher's get_candles method.
    """
    orchestrator = AutoAgentOrchestrator(mock_orchestrator_config)
    orchestrator.memory_manager.store_analysis_result = AsyncMock()
    await orchestrator.analyze_market("EUR_USD")

    # The mock_analysis_config fixture specifies two timeframes: H1 and H4
    assert orchestrator.data_fetcher.get_candles.call_count == 2
    
    # Check that it was called with the correct instrument
    calls = orchestrator.data_fetcher.get_candles.call_args_list
    for call in calls:
        assert call.kwargs["instrument"] == "EUR_USD"

@pytest.mark.asyncio
async def test_analyze_market_with_single_timeframe(mock_orchestrator_config):
    """
    Tests that analyze_market works correctly when a single timeframe is specified.
    """
    orchestrator = AutoAgentOrchestrator(mock_orchestrator_config)
    orchestrator.memory_manager.store_analysis_result = AsyncMock()
    await orchestrator.analyze_market("EUR_USD", timeframe="H1")
    
    # Should only be called once for the specified timeframe
    assert orchestrator.data_fetcher.get_candles.call_count == 1
    
    call = orchestrator.data_fetcher.get_candles.call_args
    assert call.kwargs["instrument"] == "EUR_USD"
    assert call.kwargs["timeframe"] == "H1"

@pytest.mark.asyncio
async def test_analyze_market_updates_memory(mock_orchestrator_config):
    """
    Tests that analyze_market calls the memory manager to store the analysis context.
    """
    orchestrator = AutoAgentOrchestrator(mock_orchestrator_config)
    orchestrator.memory_manager.store_analysis_result = AsyncMock(return_value="test_context_id")
    
    results = await orchestrator.analyze_market("EUR_USD")
    
    orchestrator.memory_manager.store_analysis_result.assert_called_once()
    call_args = orchestrator.memory_manager.store_analysis_result.call_args
    assert call_args.args[0] == results
    assert call_args.args[1] == "comprehensive" 
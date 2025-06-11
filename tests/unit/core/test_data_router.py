import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock

from forex_ai.core.data_router import DataRouter, DataType, ProcessingTier


@pytest.fixture
def router():
    return DataRouter()


def test_initialization(router):
    assert isinstance(router, DataRouter)
    assert router.routing_rules is not None
    assert router.metrics is not None
    assert isinstance(router.orchestration_queue, asyncio.Queue)
    assert isinstance(router.deep_research_queue, asyncio.Queue)


def test_load_routing_rules(router):
    rules = router._load_routing_rules()
    assert DataType.PRICE_TICK.value in rules
    assert rules[DataType.PRICE_TICK.value]['tier'] == ProcessingTier.DIRECT.value


@pytest.mark.parametrize("data, expected_type", [
    ({'type': 'CUSTOM_TYPE'}, 'CUSTOM_TYPE'),
    ({'price': 1.23}, DataType.PRICE_TICK.value),
    ({'pattern': 'abc'}, DataType.PATTERN_CONFIRMATION.value),
    ({'headline': 'xyz'}, DataType.NEWS_HEADLINE.value),
    ({'economic': 'report'}, DataType.ECONOMIC_INDICATOR.value),
    ({'sentiment': 'positive'}, DataType.SOCIAL_SENTIMENT.value),
    ({'research': 'paper'}, DataType.STRATEGY_RESEARCH.value),
    ({}, DataType.UNKNOWN.value),
])
def test_infer_data_type(router, data, expected_type):
    assert router._infer_data_type(data) == expected_type


def test_register_direct_handler(router):
    handler = MagicMock()
    router.register_direct_handler(DataType.PRICE_TICK.value, handler)
    assert DataType.PRICE_TICK.value in router.direct_handlers
    assert router.direct_handlers[DataType.PRICE_TICK.value] == handler


def test_register_orchestration_handler(router):
    handler = MagicMock()
    router.register_orchestration_handler(handler)
    assert router.orchestration_handler == handler


def test_register_deep_research_handler(router):
    handler = MagicMock()
    router.register_deep_research_handler(handler)
    assert router.deep_research_handler == handler


@pytest.mark.asyncio
async def test_route_to_direct(router):
    handler = AsyncMock()
    router.register_direct_handler(DataType.PRICE_TICK.value, handler)
    data = {'price': 1.23}
    await router.route(data)
    handler.assert_called_once_with(data)


@pytest.mark.asyncio
async def test_route_to_orchestration(router):
    router.orchestration_handler = AsyncMock()
    data = {'economic': 'report'}
    await router.route(data, data_type=DataType.ECONOMIC_INDICATOR.value)
    # The handler is called by a worker, so we check the queue
    queued_item = await router.orchestration_queue.get()
    assert queued_item[0] == data


@pytest.mark.asyncio
async def test_route_to_deep_research(router):
    router.deep_research_handler = AsyncMock()
    data = {'research': 'paper'}
    await router.route(data, data_type=DataType.STRATEGY_RESEARCH.value)
    # The handler is called by a worker, so we check the queue
    queued_item = await router.deep_research_queue.get()
    assert queued_item[0] == data


@pytest.mark.asyncio
async def test_start_and_stop(router):
    router.orchestration_handler = AsyncMock()
    router.deep_research_handler = AsyncMock()
    await router.start()
    assert router.running
    assert len(router.workers) == 3
    await router.stop()
    assert not router.running
    assert len(router.workers) == 0 
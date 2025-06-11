import pytest
from forex_ai.agents.communication import MessageBus

def test_message_bus_initialization():
    bus = MessageBus()
    assert bus.queues == {}
    assert bus.handlers == {} 
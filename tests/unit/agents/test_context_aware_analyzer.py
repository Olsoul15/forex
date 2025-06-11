import pytest
from forex_ai.agents.context_aware_analyzer import ContextAwareAnalyzer

@pytest.fixture
def analyzer():
    return ContextAwareAnalyzer()

def test_context_aware_analyzer_initialization(analyzer):
    assert analyzer.memory is not None 
import pytest
from forex_ai.agents.base import BaseAgent
from typing import Dict, Any

class ConcreteAgent(BaseAgent):
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "processed"}

def test_base_agent_initialization():
    agent = ConcreteAgent(name="Test Agent")
    assert agent.name == "Test Agent" 
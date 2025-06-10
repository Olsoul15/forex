"""
AutoAgent integration for AI Forex trading system.

This module provides integration between the AutoAgent system and the AI Forex
trading system, enabling cross-analysis coordination, context-aware analysis, and
refined trading decisions.
"""

from forex_ai.integration.enhanced_memory_manager import EnhancedMemoryManager
from forex_ai.integration.autoagent_orchestrator import AutoAgentOrchestrator
from forex_ai.integration.tools.technical_tools import get_technical_tools
from forex_ai.integration.tools.fundamental_tools import get_fundamental_tools
from forex_ai.integration.tools.correlation_tools import get_correlation_tools
from forex_ai.integration.tools.signal_tools import get_signal_tools

__all__ = [
    "EnhancedMemoryManager",
    "AutoAgentOrchestrator",
    "get_technical_tools",
    "get_fundamental_tools",
    "get_correlation_tools",
    "get_signal_tools",
]

# Default configuration
DEFAULT_CONFIG = {
    "memory_config": {"schema_prefix": "autoagent_", "cache_size": 100},
    "model": "gpt-4",
    "temperature": 0.2,
    "max_tokens": 2048,
    "technical_tools_config": {},
    "fundamental_tools_config": {},
    "correlation_tools_config": {},
    "signal_tools_config": {},
    "confidence_threshold": 0.7,
}


async def create_orchestrator(config=None):
    """
    Create and initialize an AutoAgent orchestrator.

    Args:
        config: Optional configuration dictionary

    Returns:
        Initialized orchestrator
    """
    # Merge provided config with defaults
    merged_config = DEFAULT_CONFIG.copy()
    if config:
        merged_config.update(config)

    # Create orchestrator
    orchestrator = AutoAgentOrchestrator(merged_config)

    # Initialize
    await orchestrator.start()

    return orchestrator

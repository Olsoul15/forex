"""
AutoAgent integration for AI Forex trading system.

This module provides integration between the AutoAgent system and the AI Forex
trading system, enabling cross-analysis coordination, context-aware analysis, and
refined trading decisions.
"""

from forex_ai.integration.enhanced_memory_manager import EnhancedMemoryManager
# from forex_ai.integration.autoagent_orchestrator import AutoAgentOrchestrator

__all__ = [
    "EnhancedMemoryManager",
    # "AutoAgentOrchestrator",
]

# Default configuration
DEFAULT_CONFIG = {
    "memory_config": {"schema_prefix": "autoagent_", "cache_size": 100},
    "model": "gpt-4",
    "temperature": 0.2,
    "max_tokens": 2048,
    "confidence_threshold": 0.7,
}


# async def create_orchestrator(config=None):
#     """
#     Create and initialize an AutoAgent orchestrator.
#
#     Args:
#         config: Optional configuration dictionary
#
#     Returns:
#         Initialized orchestrator
#     """
#     # Merge provided config with defaults
#     merged_config = DEFAULT_CONFIG.copy()
#     if config:
#         merged_config.update(config)
#
#     # Create orchestrator
#     orchestrator = AutoAgentOrchestrator(merged_config)
#
#     # Initialize
#     await orchestrator.start()
#
#     return orchestrator

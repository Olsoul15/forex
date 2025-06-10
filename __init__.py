"""
Forex AI Trading System

A comprehensive trading system for forex markets with AI-powered analysis.
"""

__version__ = "0.1.0"
__author__ = "Forex AI Team"
__email__ = "info@forexai.example.com"

from forex_ai.config import settings
from forex_ai.exceptions import AgentError
from forex_ai.custom_types import TimeFrame, CurrencyPair

__all__ = ["get_settings", "ForexAiError", "TimeFrame", "CurrencyPair"]

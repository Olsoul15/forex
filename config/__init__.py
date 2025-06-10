"""
Configuration package for the Forex AI Trading System.

This package provides access to various configuration settings,
environment variables, and constants used throughout the system.
"""

from forex_ai.config.settings import get_settings, Settings

__all__ = ["get_settings", "Settings"]

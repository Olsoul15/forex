"""
Market data connectors for the Forex AI Trading System.

This package contains connectors for various market data providers,
including TradingView, Alpha Vantage, and financial news sources.
"""

# Remove import for missing TradingViewConnector
# from forex_ai.data.connectors.trading_view import TradingViewConnector
from .base import DataConnector
from forex_ai.data.connectors.alpha_vantage import AlphaVantageConnector
from forex_ai.data.connectors.news_api import NewsApiConnector
from forex_ai.data.connectors.youtube import YouTubeConnector

__all__ = [
    'DataConnector',
    # 'TradingViewConnector',
    'AlphaVantageConnector',
    'NewsApiConnector',
    'YouTubeConnector',
] 
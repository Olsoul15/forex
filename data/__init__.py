"""
Data management components for the Forex AI Trading System.

This package includes modules for:
- Market data connectors (TradingView, Alpha Vantage)
- News and sentiment analysis (News APIs, YouTube)
- Data storage (PostgreSQL, Redis)
- Data processing pipelines
"""

# Import main components
from forex_ai.data.connectors import (
    # TradingViewConnector, # Removed as the module doesn't exist
    AlphaVantageConnector,
    NewsApiConnector,
    YouTubeConnector,
)

from forex_ai.data.storage import (
    PostgresClient,
    get_postgres_client,
    RedisClient,
    get_redis_client,
)

# Import from pipelines
from forex_ai.data.pipelines.market_data import (
    fetch_market_data,
    import_from_csv,
    export_to_csv,
    export_to_json,
    export_to_excel,
    convert_timeframe,
    merge_data_sources,
    normalize_data,
    export_trade_history
)

__all__ = [
    # Data connectors
    # 'TradingViewConnector', # Removed as the module doesn't exist
    'AlphaVantageConnector',
    'NewsApiConnector',
    'YouTubeConnector',
    
    # Storage clients
    'PostgresClient',
    'get_postgres_client',
    'RedisClient',
    'get_redis_client',
    
    # Pipelines
    "fetch_market_data",
    "import_from_csv",
    "export_to_csv",
    "export_to_json",
    "export_to_excel",
    "convert_timeframe",
    "merge_data_sources",
    "normalize_data",
    "export_trade_history"
] 
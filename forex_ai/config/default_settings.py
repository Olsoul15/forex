"""
Default settings for the AI Forex system.

This module provides the default configuration values for the system.
These can be overridden by environment variables or a .env file.
"""

# Default settings dictionary
DEFAULT_SETTINGS = {
    # Database settings
    "database": {
        "type": "postgres",
        "host": "localhost",
        "port": 5432,
        "name": "forex_ai",
        "user": "forex_user",
        "password": "password123",
        "connection_timeout": 30,
        "pool_size": 20,
    },
    # API settings
    "api": {
        "host": "0.0.0.0",
        "port": 8000,
        "debug": False,
        "cors_origins": ["*"],
        "rate_limit": 100,
        "timeout": 60,
    },
    # Data provider settings
    "data_providers": {
        "default": "oanda",
        "oanda": {
            "api_key": "",
            "account_id": "",
            "use_practice": True,
            "request_timeout": 30,
        },
        "alpha_vantage": {"api_key": "", "request_timeout": 30},
    },
    # Agent settings
    "agents": {
        "technical_analysis": {
            "indicators_to_use": [
                "sma",
                "ema",
                "rsi",
                "macd",
                "bollinger",
                "atr",
                "stochastic",
            ],
            "max_lookback": 1000,
            "use_multiple_timeframes": True,
        },
        "sentiment_analysis": {
            "use_social_media": True,
            "use_news": True,
            "sentiment_threshold": 0.6,
            "max_age_hours": 24,
        },
        "strategy": {
            "default_risk_per_trade": 0.02,
            "max_correlation": 0.7,
            "min_win_rate": 0.55,
        },
        "context_analyzer": {
            "max_contexts_per_query": 10,
            "default_lookback_days": 30,
            "confidence_threshold": 0.6,
            "storage_days": 90,
            "context_table_name": "analysis_contexts",
        },
    },
    # Feature flags
    "features": {
        "multi_timeframe_analysis": True,
        "advanced_pattern_recognition": True,
        "neural_network_predictions": True,
        "social_sentiment_analysis": True,
        "economic_indicators": True,
        "backtesting": True,
        "strategy_optimization": True,
        "risk_management": True,
        "context_aware_analysis": True,  # New feature flag for context-aware analysis
    },
    # AutoAgent integration settings
    "integrations": {
        "autoagent": {
            "enabled": True,
            "api_key": "",
            "base_url": "https://api.autoagent.ai/v1",
            "default_model": "gpt-4",
            "request_timeout": 60,
            "max_tokens": 4000,
        }
    },
    # Logging settings
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": "logs/forex_ai.log",
        "max_size": 10485760,  # 10MB
        "backup_count": 5,
    },
    # Cache settings
    "cache": {
        "type": "redis",
        "host": "localhost",
        "port": 6379,
        "password": "",
        "ttl": 3600,  # 1 hour
        "key_prefix": "forex_ai:",
    },
}

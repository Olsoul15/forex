"""
Core settings and configuration for the Forex AI Trading System.

This module provides a centralized configuration management system
using Pydantic's Settings management. It handles loading configuration
from environment variables, with appropriate type conversion and validation.
"""

import os
from functools import lru_cache
from typing import Dict, List, Optional, Any, Union

from pydantic import (
    PostgresDsn,
    RedisDsn,
    Field,
    field_validator,
    model_validator,
    HttpUrl,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings.

    This class uses Pydantic's BaseSettings to handle configuration
    from environment variables, with appropriate validation.
    """

    # Environment settings
    ENVIRONMENT: str = Field("development", env="ENVIRONMENT")
    DEBUG: bool = Field(False, env="DEBUG")
    LOG_LEVEL: str = Field("INFO", env="LOG_LEVEL")

    # Database settings
    POSTGRES_USER: str = Field("postgres", env="POSTGRES_USER")
    POSTGRES_PASSWORD: str = Field("password", env="POSTGRES_PASSWORD")
    POSTGRES_DB: str = Field("forex_ai", env="POSTGRES_DB")
    POSTGRES_HOST: str = Field("localhost", env="POSTGRES_HOST")
    POSTGRES_PORT: int = Field(5432, env="POSTGRES_PORT")
    DATABASE_URL: Optional[PostgresDsn] = Field(None, env="DATABASE_URL")

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def assemble_db_connection(cls, v: Optional[str], info) -> Any:
        """Build the DATABASE_URL if not provided."""
        if isinstance(v, str):
            return v

        values = info.data
        required_keys = [
            "POSTGRES_USER",
            "POSTGRES_PASSWORD",
            "POSTGRES_HOST",
            "POSTGRES_PORT",
            "POSTGRES_DB",
        ]
        missing_keys = [k for k in required_keys if k not in values or not values[k]]
        if missing_keys:
            missing = ", ".join(missing_keys)
            raise ValueError(f"Missing required database configuration: {missing}")

        return PostgresDsn.build(
            scheme="postgresql",
            username=values.get("POSTGRES_USER"),
            password=values.get("POSTGRES_PASSWORD"),
            host=values.get("POSTGRES_HOST"),
            port=int(values.get("POSTGRES_PORT")),
            path=f"{values.get('POSTGRES_DB') or ''}",
        )

    # Redis settings
    REDIS_HOST: str = Field("localhost", env="REDIS_HOST")
    REDIS_PORT: int = Field(6379, env="REDIS_PORT")
    REDIS_DB: int = Field(0, env="REDIS_DB")
    REDIS_PASSWORD: str = Field("", env="REDIS_PASSWORD")
    REDIS_SSL: bool = Field(False, env="REDIS_SSL")
    REDIS_URL: Optional[RedisDsn] = Field(None, env="REDIS_URL")
    REDIS_NOTIFICATION_CHANNEL: str = Field(
        "backtest_notifications", env="REDIS_NOTIFICATION_CHANNEL"
    )

    @field_validator("REDIS_URL", mode="before")
    @classmethod
    def assemble_redis_connection(cls, v: Optional[str], info) -> Any:
        """Build the REDIS_URL if not provided."""
        if isinstance(v, str):
            return v

        values = info.data
        required_keys = ["REDIS_HOST", "REDIS_PORT", "REDIS_PASSWORD"]
        missing_keys = [k for k in required_keys if k not in values or not values[k]]
        if missing_keys:
            missing = ", ".join(missing_keys)
            raise ValueError(f"Missing required Redis configuration: {missing}")

        return RedisDsn.build(
            scheme="redis",
            password=values.get("REDIS_PASSWORD"),
            host=values.get("REDIS_HOST"),
            port=int(values.get("REDIS_PORT")),
            path="0",
        )

    # Supabase settings
    SUPABASE_URL: str = Field(
        "https://your-project-url.supabase.co", env="SUPABASE_URL"
    )
    SUPABASE_KEY: str = Field("your-supabase-anon-key", env="SUPABASE_KEY")

    # OANDA Proxy URL (used by TA Service)
    OANDA_PROXY_URL: str = Field("http://localhost:8002", env="OANDA_PROXY_URL")

    # TA Service Port (used by TA Service server.py)
    TA_SERVICE_PORT: int = Field(8003, env="TA_PORT")

    # External API settings
    ALPHA_VANTAGE_API_KEY: str = Field("placeholder", env="ALPHA_VANTAGE_API_KEY")
    TRADING_VIEW_API_KEY: str = Field("placeholder", env="TRADING_VIEW_API_KEY")
    NEWS_API_KEY: str = Field("placeholder", env="NEWS_API_KEY")
    YOUTUBE_API_KEY: str = Field("placeholder", env="YOUTUBE_API_KEY")

    # AI model settings
    AZURE_OPENAI_KEY: str = Field("placeholder", env="AZURE_OPENAI_KEY")
    AZURE_OPENAI_ENDPOINT: str = Field("placeholder", env="AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_VERSION: str = Field(
        "2024-02-15-preview", env="AZURE_OPENAI_API_VERSION"
    )
    OPENAI_API_KEY: str = Field("placeholder", env="OPENAI_API_KEY")
    GROQ_API_KEY: str = Field("placeholder", env="GROQ_API_KEY")
    OPENROUTER_API_KEY: str = Field("placeholder", env="OPENROUTER_API_KEY")
    GOOGLE_API_KEY: str = Field("placeholder", env="GOOGLE_API_KEY")

    # Default model selections
    REASONING_MODEL: str = Field("gpt4", env="REASONING_MODEL")  # Azure GPT-4
    CHAT_MODEL: str = Field("gpt35turbo", env="CHAT_MODEL")  # Azure GPT-3.5
    VISION_MODEL: str = Field("gemini-1.5-flash-001", env="VISION_MODEL")  # Google
    EMBEDDING_MODEL: str = Field(
        "text-embedding-3-small", env="EMBEDDING_MODEL"
    )  # OpenAI

    # Model provider preferences
    DEFAULT_PROVIDER: str = Field("azure", env="DEFAULT_PROVIDER")
    FALLBACK_PROVIDERS: List[str] = Field(
        ["openai", "groq", "openrouter"], env="FALLBACK_PROVIDERS"
    )

    # Provider-specific settings
    AZURE_DEPLOYMENTS: Dict[str, str] = Field(
        default={
            "gpt4": "gpt-4",
            "gpt35turbo": "gpt-35-turbo",
            "embedding": "text-embedding-3-small",
        },
        env="AZURE_DEPLOYMENTS",
    )

    OPENROUTER_MODELS: Dict[str, str] = Field(
        default={
            "claude3": "anthropic/claude-3-sonnet",
            "mixtral": "mistralai/mixtral-8x7b",
            "llama2": "meta/llama2-70b",
        },
        env="OPENROUTER_MODELS",
    )

    GROQ_MODELS: Dict[str, str] = Field(
        default={"mixtral": "mixtral-8x7b-32768", "llama2": "llama2-70b-4096"},
        env="GROQ_MODELS",
    )

    # MCP Server settings
    MCP_HOST: str = Field("localhost", env="MCP_HOST")
    MCP_PORT: int = Field(8080, env="MCP_PORT")
    MCP_API_KEY: str = Field("placeholder", env="MCP_API_KEY")
    MCP_SECRET: str = Field("placeholder", env="MCP_SECRET")

    # Trading settings
    TRADING_ENABLED: bool = Field(False, env="TRADING_ENABLED")
    MAX_POSITION_SIZE: float = Field(0.01, env="MAX_POSITION_SIZE")
    MAX_OPEN_POSITIONS: int = Field(3, env="MAX_OPEN_POSITIONS")
    DEFAULT_STOP_LOSS_PIPS: int = Field(50, env="DEFAULT_STOP_LOSS_PIPS")
    DEFAULT_TAKE_PROFIT_PIPS: int = Field(100, env="DEFAULT_TAKE_PROFIT_PIPS")
    RISK_PER_TRADE_PERCENT: float = Field(1.0, env="RISK_PER_TRADE_PERCENT")

    # Web dashboard settings
    WEB_PORT: int = Field(8000, env="WEB_PORT")
    JWT_SECRET_KEY: str = Field(
        "placeholder_jwt_secret_key_for_development_only", env="JWT_SECRET_KEY"
    )
    JWT_ALGORITHM: str = Field("HS256", env="JWT_ALGORITHM")
    JWT_EXPIRE_MINUTES: int = Field(60, env="JWT_EXPIRE_MINUTES")
    ENABLE_DOCS: bool = Field(True, env="ENABLE_DOCS")

    # N8N settings
    N8N_HOST: str = Field("localhost", env="N8N_HOST")
    N8N_PORT: int = Field(5678, env="N8N_PORT")
    N8N_ENCRYPTION_KEY: str = Field("placeholder", env="N8N_ENCRYPTION_KEY")

    # Logging and monitoring
    SENTRY_DSN: Optional[str] = Field(None, env="SENTRY_DSN")
    ENABLE_PERFORMANCE_LOGGING: bool = Field(True, env="ENABLE_PERFORMANCE_LOGGING")

    # System constants - not configurable via env vars
    VERSION: str = "0.1.0"
    API_PREFIX: str = "/api/v1"
    DEFAULT_CURRENCY_PAIRS: List[str] = [
        "EUR/USD",
        "GBP/USD",
        "USD/JPY",
        "USD/CHF",
        "AUD/USD",
        "USD/CAD",
        "NZD/USD",
    ]
    SUPPORTED_TIMEFRAMES: List[str] = [
        "1m",
        "5m",
        "15m",
        "30m",
        "1h",
        "4h",
        "1d",
        "1w",
        "1M",
    ]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="allow",
    )


@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings.

    Uses lru_cache to cache settings for performance.

    Returns:
        Settings: Application settings
    """
    return Settings()

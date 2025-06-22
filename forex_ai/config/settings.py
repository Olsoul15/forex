"""
Core settings and configuration for the Forex AI Trading System.

This module provides a centralized configuration management system
using Pydantic's Settings management. It handles loading configuration
from environment variables, with appropriate type conversion and validation.
"""

import os
import logging
from functools import lru_cache
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from pydantic import (
    PostgresDsn,
    RedisDsn,
    Field,
    field_validator,
    model_validator,
    HttpUrl,
    validator,
    root_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict
import warnings

logger = logging.getLogger(__name__)

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
    FOREX_AI_DEV_MODE: bool = Field(False, env="FOREX_AI_DEV_MODE")  # Disable development mode by default to use real data

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
    SUPABASE_SERVICE_KEY: str = Field("your-supabase-service-key", env="SUPABASE_SERVICE_KEY")
    SUPABASE_JWT_SECRET: Optional[str] = Field(None, env="SUPABASE_JWT_SECRET")
    SUPABASE_JWT_EXPIRY: int = Field(3600, env="SUPABASE_JWT_EXPIRY")  # 1 hour

    # OANDA Proxy URL (used by TA Service)
    OANDA_PROXY_URL: str = Field("http://localhost:8002", env="OANDA_PROXY_URL")
    
    # OANDA API settings
    OANDA_API_KEY: Optional[str] = Field(None, env="OANDA_API_KEY")
    OANDA_ACCESS_TOKEN: Optional[str] = Field(None, env="OANDA_ACCESS_TOKEN")
    OANDA_ACCOUNT_ID: str = Field("", env="OANDA_ACCOUNT_ID")
    
    @field_validator("OANDA_ACCESS_TOKEN", mode="before")
    @classmethod
    def set_oanda_access_token(cls, v: Optional[str], info) -> Any:
        """Use OANDA_API_KEY as fallback for OANDA_ACCESS_TOKEN."""
        if v:
            return v
        return info.data.get("OANDA_API_KEY", "")

    # TA Service Port (used by TA Service server.py)
    TA_SERVICE_PORT: int = Field(8003, env="TA_PORT")

    # External API settings
    ALPHA_VANTAGE_API_KEY: str = Field("placeholder", env="ALPHA_VANTAGE_API_KEY")
    TRADING_VIEW_API_KEY: str = Field("placeholder", env="TRADING_VIEW_API_KEY")
    THE_NEWS_API_KEY: str = Field("placeholder", env="THE_NEWS_API_KEY")
    YOUTUBE_API_KEY: Optional[str] = Field(None, env="YOUTUBE_API_KEY")

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
    GCP_LOCATION: str = Field("us-central1", env="GCP_LOCATION")

    # Default model selections
    REASONING_MODEL: str = Field("gpt4", env="REASONING_MODEL")  # Azure GPT-4
    CHAT_MODEL: str = Field("gpt35turbo", env="CHAT_MODEL")  # Azure GPT-3.5
    VISION_MODEL: str = Field("gemini-1.5-flash-001", env="VISION_MODEL")  # Google
    EMBEDDING_MODEL: str = Field(
        "text-embedding-3-small", env="EMBEDDING_MODEL"
    )  # OpenAI

    # Model provider preferences
    DEFAULT_PROVIDER: str = Field("mcp", env="DEFAULT_PROVIDER")

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

    # Workflow engine settings
    WORKFLOW_ENABLED: bool = Field(True, env="WORKFLOW_ENABLED")
    WORKFLOW_MAX_WORKERS: int = Field(4, env="WORKFLOW_MAX_WORKERS")
    WORKFLOW_RESULTS_TTL: int = Field(30, env="WORKFLOW_RESULTS_TTL")  # Days to keep results

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

    # New settings
    API_HOST: str = Field("0.0.0.0", env="API_HOST")
    API_PORT: int = Field(8000, env="API_PORT")
    API_WORKERS: int = Field(4, env="API_WORKERS")
    API_TIMEOUT: int = Field(60, env="API_TIMEOUT")
    API_RELOAD: bool = Field(True, env="API_RELOAD")
    API_DEBUG: bool = Field(True, env="API_DEBUG")
    API_ROOT_PATH: str = Field("", env="API_ROOT_PATH")
    API_DOCS_URL: str = Field("/docs", env="API_DOCS_URL")
    API_REDOC_URL: str = Field("/redoc", env="API_REDOC_URL")
    API_OPENAPI_URL: str = Field("/openapi.json", env="API_OPENAPI_URL")
    API_TITLE: str = Field("Forex AI Trading System API", env="API_TITLE")
    API_DESCRIPTION: str = Field("API for the Forex AI Trading System", env="API_DESCRIPTION")
    API_VERSION: str = Field("1.0.0", env="API_VERSION")

    # Security settings
    SECRET_KEY: str = Field("forex_ai_development_secret_key_for_testing_only", env="SECRET_KEY")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(60 * 24, env="ACCESS_TOKEN_EXPIRE_MINUTES")  # 1 day
    ALGORITHM: str = Field("HS256", env="ALGORITHM")

    # Proxy settings
    PROXY_HOST: str = Field("0.0.0.0", env="PROXY_HOST")
    PROXY_PORT: int = Field(8001, env="PROXY_PORT")
    PROXY_WORKERS: int = Field(2, env="PROXY_WORKERS")

    # Workflow engine settings
    WORKFLOW_ENABLED: bool = Field(True, env="WORKFLOW_ENABLED")
    WORKFLOW_MAX_WORKERS: int = Field(4, env="WORKFLOW_MAX_WORKERS")
    WORKFLOW_RESULTS_TTL: int = Field(30, env="WORKFLOW_RESULTS_TTL")  # Days to keep results

    # Path settings
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = Field(None)
    LOG_DIR: Path = Field(None)

    @validator("DATA_DIR", pre=True, always=True)
    def set_data_dir(cls, v, values):
        """Set the data directory."""
        if v is None:
            return values["BASE_DIR"] / "data" / "storage"
        return Path(v)
    
    @validator("LOG_DIR", pre=True, always=True)
    def set_log_dir(cls, v, values):
        """Set the log directory."""
        if v is None:
            return values["BASE_DIR"].parent / "logs"
        return Path(v)
    
    @validator("OANDA_ACCESS_TOKEN", pre=True, always=True)
    def validate_oanda_token(cls, v, values):
        """Validate OANDA access token, using API_KEY as fallback."""
        if v is None and "OANDA_API_KEY" in values and values["OANDA_API_KEY"] is not None:
            if values.get("FOREX_AI_DEV_MODE", True):
                logger.info(
                    "Development mode: Using OANDA_API_KEY from environment variables as fallback for OANDA_ACCESS_TOKEN."
                )
                return values["OANDA_API_KEY"]
            else:
                logger.warning(
                    "Production mode: OANDA credentials should come from the frontend, not environment variables."
                )
        return v
    
    @root_validator(skip_on_failure=True)
    def check_oanda_credentials(cls, values):
        """Check if OANDA credentials are set."""
        if values.get("FOREX_AI_DEV_MODE", True):
            if not values.get("OANDA_ACCESS_TOKEN") and not values.get("OANDA_API_KEY"):
                logger.warning(
                    "Development mode: OANDA credentials not set in environment variables. "
                    "OANDA functionality will be limited."
                )
        else:
            # In production mode, credentials should come from frontend
            if values.get("OANDA_ACCESS_TOKEN") or values.get("OANDA_API_KEY"):
                logger.warning(
                    "Production mode: OANDA credentials found in environment variables. "
                    "In production, these should come from the frontend instead."
                )
        return values

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

def get_env_var(primary_name: str, aliases: List[str] = None, default: Any = None) -> Any:
    """
    Get an environment variable with fallbacks to aliases.
    
    Args:
        primary_name: Primary environment variable name.
        aliases: List of alias names to try if primary is not found.
        default: Default value if no variables are found.
        
    Returns:
        Environment variable value or default.
    """
    # Try primary name first
    value = os.environ.get(primary_name)
    if value is not None:
        return value
    
    # Try aliases if provided
    if aliases:
        for alias in aliases:
            value = os.environ.get(alias)
            if value is not None:
                logger.warning(
                    f"Using {alias} as fallback for {primary_name}. "
                    f"Consider using {primary_name} directly."
                )
                return value
    
    # Return default if nothing found
    return default

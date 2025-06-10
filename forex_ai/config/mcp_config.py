"""
MCP (Market Connection Protocol) server connection settings.

This module provides configuration and utilities for connecting to the
MCP server, which serves as the bridge between the Forex AI system and
trading platforms like MT4/MT5.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, validator

from forex_ai.config.settings import get_settings


class MCPConnectionConfig(BaseModel):
    """
    MCP Server connection configuration.

    This class defines the configuration needed to connect to
    the MCP server, which acts as a bridge to trading platforms.
    """

    host: str = Field(..., description="MCP server hostname")
    port: int = Field(..., description="MCP server port")
    api_key: str = Field(..., description="API key for authentication")
    secret: str = Field(..., description="Secret for API key authentication")
    use_ssl: bool = Field(True, description="Whether to use SSL for the connection")
    connection_timeout: int = Field(10, description="Connection timeout in seconds")
    keep_alive: bool = Field(True, description="Whether to keep the connection alive")
    reconnect_attempts: int = Field(5, description="Number of reconnection attempts")
    reconnect_delay: int = Field(
        5, description="Delay between reconnection attempts in seconds"
    )

    @validator("port")
    def validate_port(cls, v: int) -> int:
        """Validate that the port is in a valid range."""
        if not 1 <= v <= 65535:
            raise ValueError(f"Port must be between 1 and 65535, got {v}")
        return v


class MCPOrderConfig(BaseModel):
    """
    MCP Order execution configuration.

    This class defines the configuration for order execution
    through the MCP server.
    """

    default_slippage: int = Field(3, description="Default slippage in points")
    max_retries: int = Field(3, description="Maximum order placement retries")
    retry_delay: int = Field(1, description="Delay between retries in seconds")
    emergency_close_enabled: bool = Field(
        True, description="Whether emergency close is enabled"
    )
    validate_orders: bool = Field(
        True, description="Whether to validate orders before sending"
    )
    default_comment: str = Field("ForexAI", description="Default comment for orders")
    default_magic_number: int = Field(
        123456, description="Default magic number for orders"
    )


class MCPConfig(BaseModel):
    """
    Complete MCP configuration.

    This class combines connection and order execution configuration
    for the MCP server.
    """

    connection: MCPConnectionConfig
    order: MCPOrderConfig
    enabled_pairs: List[str] = Field(
        ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "USD/CAD", "NZD/USD"],
        description="Enabled currency pairs",
    )
    logging_enabled: bool = Field(True, description="Whether to log MCP operations")
    simulation_mode: bool = Field(
        False, description="Whether to run in simulation mode"
    )

    @validator("enabled_pairs")
    def validate_pairs(cls, v: List[str]) -> List[str]:
        """Validate that the pairs are in the correct format."""
        for pair in v:
            if not "/" in pair or len(pair) != 7:
                raise ValueError(f"Currency pair must be in format XXX/YYY, got {pair}")
        return v


def get_mcp_config() -> MCPConfig:
    """
    Get MCP configuration from application settings.

    Returns:
        MCPConfig: MCP configuration
    """
    settings = get_settings()

    connection_config = MCPConnectionConfig(
        host=settings.MCP_HOST,
        port=settings.MCP_PORT,
        api_key=settings.MCP_API_KEY,
        secret=settings.MCP_SECRET,
    )

    order_config = MCPOrderConfig(
        default_slippage=3,
        max_retries=3,
        retry_delay=1,
        emergency_close_enabled=True,
        validate_orders=True,
        default_comment="ForexAI",
        default_magic_number=123456,
    )

    return MCPConfig(
        connection=connection_config,
        order=order_config,
        enabled_pairs=settings.DEFAULT_CURRENCY_PAIRS,
        logging_enabled=True,
        simulation_mode=not settings.TRADING_ENABLED,
    )

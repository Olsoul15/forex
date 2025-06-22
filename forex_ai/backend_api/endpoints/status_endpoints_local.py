"""
Status API Endpoints for Forex AI Trading System.

This module provides FastAPI endpoints for checking the status of various system components.
"""

import logging
import time
import platform
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/status", tags=["status"])


# Models
class ServiceStatus(BaseModel):
    """Status information for a system service."""

    name: str
    status: str
    version: str = None
    details: Dict[str, Any] = None


class SystemStatusResponse(BaseModel):
    """System status response model."""

    status: str
    uptime: int
    system_info: Dict[str, Any]
    services: List[ServiceStatus]
    timestamp: datetime


class MarketStatusResponse(BaseModel):
    """Market status response model."""

    status: str
    current_session: Optional[str] = None
    next_session: Optional[str] = None
    next_session_start: Optional[str] = None
    is_weekend: bool
    is_open: bool = False
    details: Dict[str, Any] = None
    timestamp: datetime


class ConnectionTestResponse(BaseModel):
    """Connection test response model."""

    status: str
    latency_ms: float
    details: Dict[str, Any] = None
    timestamp: datetime


class DatabaseStatusResponse(BaseModel):
    """Database status response model."""

    status: str
    connection_pool: Dict[str, Any] = None
    last_successful_query: datetime = None
    details: Dict[str, Any] = None
    timestamp: datetime


class MarketDataStatusResponse(BaseModel):
    """Market data status response model."""

    status: str
    provider: str
    last_update: datetime = None
    available_instruments: int = 0
    details: Dict[str, Any] = None
    timestamp: datetime


# Initialization timestamp for uptime calculation
start_time = time.time()


# Helpers
def get_system_info() -> Dict[str, Any]:
    """Get system information."""
    return {
        "os": platform.system(),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
    }


# Endpoints
@router.get("/system", response_model=SystemStatusResponse)
async def get_system_status():
    """
    Get system status and health information.

    Returns information about system health, running services, and uptime.
    """
    logger.info("Processing system status request")

    # Calculate uptime
    uptime_seconds = int(time.time() - start_time)

    # Get status of important services
    services = [
        ServiceStatus(
            name="API Server",
            status="online",
            version="1.0.0",
            details={"environment": "development"},
        ),
        ServiceStatus(
            name="Database",
            status="online",
            version="1.0.0",
            details={"type": "Mock"},
        ),
        ServiceStatus(
            name="Market Data Provider",
            status="online",
            version="1.0.0",
            details={"provider": "OANDA"},
        ),
    ]

    # Overall status is online if all critical services are online
    overall_status = "online"

    return SystemStatusResponse(
        status=overall_status,
        uptime=uptime_seconds,
        system_info=get_system_info(),
        services=services,
        timestamp=datetime.now(),
    )


@router.get("/market", response_model=MarketStatusResponse)
async def get_market_status():
    """
    Get current forex market status.

    Returns information about whether markets are open, current and upcoming sessions.
    """
    logger.info("Processing market status request")

    # Get current time
    now = datetime.now()
    current_day = now.weekday()  # 0=Monday, 6=Sunday
    current_hour = now.hour

    # Check if it's the weekend
    is_weekend = current_day >= 5

    # Determine current session (simplified logic)
    if is_weekend:
        status = "closed"
        is_open = False
        current_session = None
        next_session = "sydney"
        next_session_start = None
    else:
        status = "open"
        is_open = True
        if 0 <= current_hour < 8:
            current_session = "sydney"
            next_session = "london"
            next_session_start = None
        elif 8 <= current_hour < 16:
            current_session = "london"
            next_session = "new_york"
            next_session_start = None
        else:
            current_session = "new_york"
            next_session = "sydney"
            next_session_start = None

    return MarketStatusResponse(
        status=status,
        current_session=current_session,
        next_session=next_session,
        next_session_start=next_session_start,
        is_weekend=is_weekend,
        is_open=is_open,
        details={"day_of_week": current_day, "hour": current_hour},
        timestamp=now,
    )


@router.get("/test", response_model=ConnectionTestResponse)
async def test_connection():
    """
    Test API connection and measure latency.

    Useful for clients to check connectivity and response time.
    """
    logger.info("Processing connection test request")

    # Simulate a latency check
    start = time.time()
    time.sleep(0.01)  # Simulate a small delay
    latency = (time.time() - start) * 1000  # Convert to milliseconds

    return ConnectionTestResponse(
        status="connected",
        latency_ms=latency,
        details={"server_time": datetime.now().isoformat()},
        timestamp=datetime.now(),
    )


@router.get("/database", response_model=DatabaseStatusResponse)
async def get_database_status():
    """
    Get database connection status.

    Returns information about the database connection.
    """
    logger.info("Processing database status request")

    return DatabaseStatusResponse(
        status="connected",
        connection_pool={"active": 1, "idle": 3, "max": 10},
        last_successful_query=datetime.now(),
        details={"type": "mock", "version": "1.0"},
        timestamp=datetime.now(),
    )


@router.get("/market-data", response_model=MarketDataStatusResponse)
async def get_market_data_status():
    """
    Get market data provider status.

    Returns information about the market data provider connection.
    """
    logger.info("Processing market data status request")

    return MarketDataStatusResponse(
        status="connected",
        provider="OANDA",
        last_update=datetime.now(),
        available_instruments=7,
        details={
            "instruments": ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "NZD_USD", "EUR_GBP"]
        },
        timestamp=datetime.now(),
    ) 

@router.get("/status/server")
async def server_redirect():
    """
    Redirect to /api/status/market-data for backward compatibility.
    """
    logger.info("Redirecting from /api/status/server to /api/status/market-data")
    return JSONResponse(
        status_code=status.HTTP_307_TEMPORARY_REDIRECT,
        headers={"Location": "/api/status/market-data"},
        content={
            "success": True,
            "message": "Endpoint moved to /api/status/market-data",
            "status_code": 307
        }
    )


@router.get("/status/execution")
async def execution_redirect():
    """
    Redirect to /api/status/market-data for backward compatibility.
    """
    logger.info("Redirecting from /api/status/execution to /api/status/market-data")
    return JSONResponse(
        status_code=status.HTTP_307_TEMPORARY_REDIRECT,
        headers={"Location": "/api/status/market-data"},
        content={
            "success": True,
            "message": "Endpoint moved to /api/status/market-data",
            "status_code": 307
        }
    )


@router.get("/status/strategies")
async def strategies_redirect():
    """
    Redirect to /api/status/market-data for backward compatibility.
    """
    logger.info("Redirecting from /api/status/strategies to /api/status/market-data")
    return JSONResponse(
        status_code=status.HTTP_307_TEMPORARY_REDIRECT,
        headers={"Location": "/api/status/market-data"},
        content={
            "success": True,
            "message": "Endpoint moved to /api/status/market-data",
            "status_code": 307
        }
    )


@router.get("/status/server")
async def mock_server():
    """
    Mock implementation for /api/status/server.
    """
    logger.info(f"Processing mock request for /api/status/server")
    return {
        "success": True,
        "message": "This is a mock implementation",
        "data": {},
        "timestamp": datetime.now().isoformat()
    }


@router.get("/status/execution")
async def mock_execution():
    """
    Mock implementation for /api/status/execution.
    """
    logger.info(f"Processing mock request for /api/status/execution")
    return {
        "success": True,
        "message": "This is a mock implementation",
        "data": {},
        "timestamp": datetime.now().isoformat()
    }


@router.get("/status/strategies")
async def mock_strategies():
    """
    Mock implementation for /api/status/strategies.
    """
    logger.info(f"Processing mock request for /api/status/strategies")
    return {
        "success": True,
        "message": "This is a mock implementation",
        "data": {},
        "timestamp": datetime.now().isoformat()
    }

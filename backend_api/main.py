"""
Forex AI Trading System API

Main application entry point for the Forex AI Trading System API.
This FastAPI application provides endpoints for trading, strategy management,
market data access, and AI-driven analysis.
"""

import os
import time
import logging
import platform
from typing import Dict, Any
from datetime import datetime, timedelta

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
from fastapi.staticfiles import StaticFiles

# Import routers
from app.endpoints import status_endpoints
from app.endpoints import strategy_endpoints
from app.endpoints import market_data_endpoints
from app.endpoints import account_endpoints
from app.endpoints import execution_endpoints
from app.endpoints import websocket_proxy_endpoints

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Get environment variables
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
API_VERSION = os.getenv("API_VERSION", "v1")
API_TITLE = os.getenv("API_TITLE", "Forex AI Trading System API")
API_DESCRIPTION = os.getenv(
    "API_DESCRIPTION",
    """
                          API for the Forex AI Trading System, providing endpoints for:
                          - System status monitoring
                          - Strategy management
                          - Market data access
                          - Account management
                          - Day trading functionality
                          - AI-driven analysis
                          """,
)

# Create FastAPI application
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url=None,  # Disable default docs
    redoc_url=None,  # Disable default redoc
    openapi_url="/api/openapi.json",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, this should be restricted
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],  # This helps with some WebSocket compatibility
)

# Include routers
app.include_router(status_endpoints.router)
app.include_router(strategy_endpoints.router)
app.include_router(market_data_endpoints.router)
app.include_router(account_endpoints.router)
app.include_router(execution_endpoints.router)
app.include_router(websocket_proxy_endpoints.router)
app.include_router(
    market_data_endpoints.metrics_router
)  # Include market metrics router

# Global variables
start_time = datetime.now()

# Initialize mock databases
from app.db.strategy_db import initialize_db as initialize_strategy_db
from app.db.market_data_db import initialize_instruments
from app.db.account_db import initialize_db as initialize_account_db
from app.db.execution_db import initialize_preferences

initialize_strategy_db()
initialize_instruments()
initialize_account_db()
initialize_preferences()


@app.get("/api/status/system", tags=["status"])
async def system_status() -> Dict[str, Any]:
    """
    Get system status information.

    Returns the current status of the API, uptime, and various system information.
    """
    uptime = datetime.now() - start_time

    return {
        "status": "operational",
        "uptime": str(uptime),
        "uptime_seconds": uptime.total_seconds(),
        "system_info": {
            "environment": ENVIRONMENT,
            "api_version": API_VERSION,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "services": {
            "database": "connected",
            "market_data_provider": "connected",
        },
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/status/market", tags=["status"])
async def market_status() -> Dict[str, Any]:
    """
    Get forex market status information.

    Returns information about whether the forex market is currently open,
    which session is active, and when the next session will open.
    """
    # Get current time in UTC
    now = datetime.now()

    # Check if it's a weekend
    is_weekend = now.weekday() >= 5  # 5 is Saturday, 6 is Sunday

    # Determine current session
    hour_utc = now.hour
    current_session = None

    if is_weekend:
        is_open = False
    else:
        # Simple session determination (would be more complex in production)
        if 22 <= hour_utc or hour_utc < 8:
            current_session = "Asia/Pacific"
            is_open = True
        elif 8 <= hour_utc < 16:
            current_session = "Europe/London"
            is_open = True
        elif 13 <= hour_utc < 22:
            current_session = "New York"
            is_open = True
        else:
            is_open = False

    # Determine next session
    next_session = None
    next_session_time = None

    if is_weekend:
        # Next session is Asian session on Monday
        days_to_monday = 7 - now.weekday() if now.weekday() == 6 else 1
        next_monday = now + timedelta(days=days_to_monday)
        next_session_time = datetime(
            next_monday.year, next_monday.month, next_monday.day, 22, 0, 0
        ).isoformat()
        next_session = "Asia/Pacific"
    elif not is_open:
        if 16 <= hour_utc < 22:
            next_session = "New York"
            next_session_time = datetime(
                now.year, now.month, now.day, 13, 0, 0
            ).isoformat()
        elif 8 <= hour_utc < 13:
            next_session = "Europe/London"
            next_session_time = datetime(
                now.year, now.month, now.day, 8, 0, 0
            ).isoformat()

    return {
        "is_open": is_open,
        "is_weekend": is_weekend,
        "current_session": current_session,
        "next_session": (
            {"name": next_session, "open_time": next_session_time}
            if next_session
            else None
        ),
        "timestamp": now.isoformat(),
    }


@app.get("/api/status/test", tags=["status"])
async def connection_test() -> Dict[str, Any]:
    """
    Test API connection and measure latency.

    Useful for clients to check connectivity and response time.
    """
    return {
        "status": "connected",
        "latency_ms": 15,  # Mock value
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/status/database", tags=["status"])
async def database_status() -> Dict[str, Any]:
    """
    Get database connection status.

    Returns information about the database connection.
    """
    return {
        "status": "connected",
        "type": "mock",  # Would be SQL, MongoDB, etc. in production
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/status/market-data", tags=["status"])
async def market_data_status() -> Dict[str, Any]:
    """
    Get market data provider status.

    Returns information about the market data provider connection.
    """
    return {
        "status": "connected",
        "provider": "OANDA",  # Mock value
        "available_instruments": [
            "EUR_USD",
            "GBP_USD",
            "USD_JPY",
            "AUD_USD",
            "USD_CAD",
            "NZD_USD",
            "EUR_GBP",
        ],
        "timestamp": datetime.now().isoformat(),
    }


# Custom middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests to the API."""
    start_time = time.time()

    # Get request details
    method = request.method
    url = str(request.url)
    client_host = request.client.host if request.client else "Unknown"

    logger.info(f"Request received: {method} {url} from {client_host}")

    # Process request
    try:
        response = await call_next(request)

        # Log response details
        process_time = time.time() - start_time
        status_code = response.status_code
        logger.info(
            f"Request completed: {method} {url} - Status: {status_code}, Time: {process_time:.3f}s"
        )

        return response
    except Exception as e:
        # Log exception
        process_time = time.time() - start_time
        logger.error(
            f"Request failed: {method} {url} - Error: {str(e)}, Time: {process_time:.3f}s",
            exc_info=True,
        )

        # Return error response
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"},
        )


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    logger.warning(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


# Custom OpenAPI documentation
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Custom Swagger UI documentation endpoint."""
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title=app.title + " - Swagger UI",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
    )


@app.get("/redoc", include_in_schema=False)
async def custom_redoc_html():
    """Custom ReDoc documentation endpoint."""
    return get_redoc_html(
        openapi_url="/openapi.json",
        title=app.title + " - ReDoc",
        redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.js",
    )


@app.get("/openapi.json", include_in_schema=False)
async def get_openapi_schema():
    """Endpoint to get the OpenAPI schema."""
    return get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to the Forex AI Trading System API",
        "version": app.version,
        "docs": "/docs",
        "redoc": "/redoc",
    }


if __name__ == "__main__":
    import uvicorn

    # Get the port from environment variables
    try:
        port = int(os.environ.get("PORT"))
        if not port:
            raise ValueError("PORT environment variable is not set")
    except (ValueError, TypeError) as e:
        print(f"ERROR: {e}")
        print("Please set the PORT environment variable to a valid port number.")
        exit(1)

    uvicorn.run(app, host="0.0.0.0", port=port)

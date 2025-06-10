"""
Dashboard API Endpoints.

This module provides FastAPI endpoints for dashboard configuration and performance metrics.
It integrates with the proxy-server for actual market data.
"""

import logging
import os
from typing import Dict, List, Any, Optional
import httpx
from fastapi import APIRouter, HTTPException, status, Depends

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["dashboard"])

# Configuration
PROXY_SERVER_URL = os.getenv("PROXY_SERVER_URL", "http://localhost:8002")


# Helper function to get httpx client
async def get_client():
    async with httpx.AsyncClient(timeout=10.0) as client:
        yield client


@router.get("/dashboard-config")
async def dashboard_config(client: httpx.AsyncClient = Depends(get_client)):
    """Return dashboard configuration."""
    try:
        # Try to get available pairs from proxy server
        try:
            response = await client.get(f"{PROXY_SERVER_URL}/api/v1/available-pairs")
            if response.status_code == 200:
                data = response.json()
                if "pairs" in data and len(data["pairs"]) > 0:
                    # Convert pairs format if needed (e.g., "EUR/USD" to "EUR_USD")
                    currency_pairs = [pair.replace("/", "_") for pair in data["pairs"]]

                    # Return with pairs from proxy server
                    return {
                        "name": "Forex AI Trading System",
                        "version": "1.0.0",
                        "currency_pairs": currency_pairs,
                        "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
                        "default_pair": "EUR_USD",
                        "default_timeframe": "4h",
                    }
        except Exception as proxy_error:
            logger.warning(
                f"Failed to fetch available pairs from proxy server: {str(proxy_error)}"
            )

        # Fallback to default configuration
        logger.info("Using default dashboard configuration")
        return {
            "name": "Forex AI Trading System",
            "version": "1.0.0",
            "currency_pairs": ["EUR_USD", "GBP_USD", "USD_JPY", "XAU_USD"],
            "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
            "default_pair": "EUR_USD",
            "default_timeframe": "4h",
        }
    except Exception as e:
        logger.error(f"Error fetching dashboard configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance/summary")
async def get_performance_summary(client: httpx.AsyncClient = Depends(get_client)):
    """Return performance summary."""
    try:
        # Try to get performance data from proxy server
        try:
            # Check if proxy server has a performance endpoint
            response = await client.get(f"{PROXY_SERVER_URL}/api/market/performance")
            if response.status_code == 200:
                return response.json()
        except Exception as proxy_error:
            logger.warning(
                f"Failed to fetch performance data from proxy server: {str(proxy_error)}"
            )

        # Fallback to mock data
        logger.info("Using mock performance data")
        return {
            "total_trades": 120,
            "win_rate": 0.62,
            "profit_factor": 1.75,
            "average_win": 45.2,
            "average_loss": 28.7,
            "max_drawdown": 12.5,
            "sharpe_ratio": 1.8,
            "monthly_returns": [2.1, 1.5, -0.8, 3.2, 1.1],
        }
    except Exception as e:
        logger.error(f"Error fetching performance summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

"""
System API Endpoints for Forex AI Trading System.

This module provides FastAPI endpoints for system status and configuration.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, Request
from pydantic import BaseModel

from forex_ai.auth.supabase import get_current_user
from forex_ai.data.storage.redis_cache import redis_cache

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/system", tags=["system"])

# Create a router for the API documentation endpoint at the root level
docs_router = APIRouter(tags=["documentation"])

@docs_router.get("/docs-data", summary="Get a list of all available API endpoints")
async def get_docs_data(request: Request) -> Dict[str, Any]:
    """
    Get a comprehensive list of all available API endpoints organized by tags.
    This endpoint provides a user-friendly overview of the entire API.
    """
    app = request.app
    routes = app.routes
    
    # Group endpoints by tags
    endpoints_by_tag = {}
    
    for route in routes:
        # Skip endpoints without path operations (like WebSocket routes)
        if not hasattr(route, "methods") or not route.methods:
            continue
            
        # Get endpoint information
        path = route.path
        methods = list(route.methods)
        
        # Get tags and summary from route.tags and route.summary if available
        tags = getattr(route, "tags", ["other"])
        if not tags:
            tags = ["other"]
            
        summary = getattr(route, "summary", "")
        description = getattr(route, "description", "")
        
        # Add endpoint to each of its tags
        for tag in tags:
            if tag not in endpoints_by_tag:
                endpoints_by_tag[tag] = []
                
            endpoint_info = {
                "path": path,
                "methods": methods,
                "summary": summary,
                "description": description
            }
            
            endpoints_by_tag[tag].append(endpoint_info)
    
    # Count total endpoints
    total_endpoints = sum(len(endpoints) for endpoints in endpoints_by_tag.values())
    
    return {
        "endpoints_by_tag": endpoints_by_tag,
        "total_endpoints": total_endpoints,
        "timestamp": datetime.now().isoformat()
    }

# Models
class SystemStatusResponse(BaseModel):
    """System status response model."""
    status: str
    version: str
    api_version: str
    market_status: Dict[str, Any]
    available_services: List[str]
    supported_instruments: List[Dict[str, Any]]
    supported_timeframes: List[Dict[str, Any]]
    configuration_parameters: Dict[str, Any]
    timestamp: datetime

@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status(current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Get comprehensive system status and configuration.
    
    This endpoint provides all necessary information for client setup in a single call,
    including system status, available services, supported instruments, and configuration
    parameters.
    
    Returns:
        SystemStatusResponse: Complete system status and configuration
    """
    logger.info(f"Processing system status request for user {current_user['id']}")
    
    try:
        # Try to get from cache first
        cache_key = f"system_status:{current_user['id']}"
        cached_data = await redis_cache.get(cache_key)
        
        if cached_data:
            logger.info("Returning cached system status")
            return SystemStatusResponse(**cached_data)
        
        # Determine market status
        now = datetime.now()
        hour_utc = now.hour
        is_weekend = now.weekday() >= 5  # 5 is Saturday, 6 is Sunday
        
        if is_weekend:
            market_status = {
                "is_open": False,
                "is_weekend": True,
                "current_session": None,
                "next_session": "Asia/Pacific",
                "next_session_time": "Monday 22:00 UTC"
            }
        else:
            # Simple session determination
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
                current_session = None
                is_open = False
                
            market_status = {
                "is_open": is_open,
                "is_weekend": False,
                "current_session": current_session,
                "next_session": None if is_open else "Next trading session",
                "next_session_time": None
            }
        
        # List of supported instruments
        supported_instruments = [
            {"name": "EUR_USD", "display_name": "EUR/USD", "pip_value": 0.0001, "type": "forex"},
            {"name": "GBP_USD", "display_name": "GBP/USD", "pip_value": 0.0001, "type": "forex"},
            {"name": "USD_JPY", "display_name": "USD/JPY", "pip_value": 0.01, "type": "forex"},
            {"name": "AUD_USD", "display_name": "AUD/USD", "pip_value": 0.0001, "type": "forex"},
            {"name": "USD_CAD", "display_name": "USD/CAD", "pip_value": 0.0001, "type": "forex"},
            {"name": "USD_CHF", "display_name": "USD/CHF", "pip_value": 0.0001, "type": "forex"},
            {"name": "NZD_USD", "display_name": "NZD/USD", "pip_value": 0.0001, "type": "forex"}
        ]
        
        # List of supported timeframes
        supported_timeframes = [
            {"name": "M1", "display_name": "1 Minute", "seconds": 60},
            {"name": "M5", "display_name": "5 Minutes", "seconds": 300},
            {"name": "M15", "display_name": "15 Minutes", "seconds": 900},
            {"name": "M30", "display_name": "30 Minutes", "seconds": 1800},
            {"name": "H1", "display_name": "1 Hour", "seconds": 3600},
            {"name": "H4", "display_name": "4 Hours", "seconds": 14400},
            {"name": "D1", "display_name": "1 Day", "seconds": 86400},
            {"name": "W1", "display_name": "1 Week", "seconds": 604800},
            {"name": "MN", "display_name": "1 Month", "seconds": 2592000}
        ]
        
        # Available services
        available_services = [
            "account_management",
            "market_data",
            "trading_signals",
            "auto_trading",
            "strategy_optimization",
            "technical_analysis",
            "fundamental_analysis"
        ]
        
        # Configuration parameters
        configuration_parameters = {
            "default_risk_per_trade": 1.0,
            "max_open_trades": 5,
            "default_timeframe": "H1",
            "default_instrument": "EUR_USD",
            "api_rate_limit": 60,  # requests per minute
            "websocket_heartbeat": 30,  # seconds
            "historical_data_limit": 1000,  # candles
            "signal_expiry_time": 8  # hours
        }
        
        # Create response
        response = SystemStatusResponse(
            status="online",
            version="1.0.0",
            api_version="v1",
            market_status=market_status,
            available_services=available_services,
            supported_instruments=supported_instruments,
            supported_timeframes=supported_timeframes,
            configuration_parameters=configuration_parameters,
            timestamp=datetime.now()
        )
        
        # Cache the response for 5 minutes
        await redis_cache.set(cache_key, response.model_dump(), ttl=300)
        
        return response
    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting system status: {str(e)}"
        )

@router.get("/health", summary="Simple health check endpoint")
async def health_check():
    """
    Simple health check endpoint that doesn't require authentication.
    
    This endpoint can be used to verify that the API server is running and responsive.
    
    Returns:
        Dict: Simple health status response
    """
    logger.info("Processing health check request")
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "api_version": "v1"
    } 
"""
System API Endpoints for Forex AI Trading System.

This module provides FastAPI endpoints for system status and configuration.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import platform
import psutil
import os
import time
import sys
import importlib.metadata

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/system", tags=["system"])

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
async def get_system_status():
    """
    Get comprehensive system status and configuration.
    
    This endpoint provides all necessary information for client setup in a single call,
    including system status, available services, supported instruments, and configuration
    parameters.
    
    Returns:
        SystemStatusResponse: Complete system status and configuration
    """
    logger.info("Processing system status request")
    
    try:
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
        
        return SystemStatusResponse(
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
    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting system status: {str(e)}"
        )

# Create a new router for the API menu endpoint at the root level
api_menu_router = APIRouter(tags=["documentation"])

@api_menu_router.get("/api-menu", summary="Get a list of all available API endpoints", response_class=HTMLResponse)
async def get_api_menu(request: Request) -> str:
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
    
    # Generate HTML response
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Forex AI Trading System API Menu</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            h1 {{
                color: #0066cc;
                border-bottom: 2px solid #0066cc;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #0066cc;
                margin-top: 30px;
                border-bottom: 1px solid #ddd;
                padding-bottom: 5px;
            }}
            .endpoint {{
                margin-bottom: 15px;
                padding: 10px;
                border-left: 4px solid #0066cc;
                background-color: #f8f9fa;
            }}
            .method {{
                display: inline-block;
                padding: 3px 6px;
                border-radius: 3px;
                font-weight: bold;
                margin-right: 10px;
                min-width: 60px;
                text-align: center;
            }}
            .get {{ background-color: #61affe; color: white; }}
            .post {{ background-color: #49cc90; color: white; }}
            .put {{ background-color: #fca130; color: white; }}
            .delete {{ background-color: #f93e3e; color: white; }}
            .path {{
                font-family: monospace;
                font-size: 1.1em;
                margin-right: 10px;
            }}
            .summary {{
                margin-top: 5px;
                color: #555;
            }}
            .description {{
                margin-top: 5px;
                color: #666;
                font-size: 0.9em;
            }}
            .stats {{
                margin-top: 30px;
                padding: 10px;
                background-color: #f8f9fa;
                border-radius: 5px;
            }}
            .links {{
                margin-top: 30px;
                padding: 10px;
                background-color: #f0f0f0;
                border-radius: 5px;
            }}
            .tag-count {{
                color: #666;
                font-size: 0.9em;
                margin-left: 10px;
            }}
        </style>
    </head>
    <body>
        <h1>Forex AI Trading System API Menu</h1>
        
        <div class="links">
            <strong>API Documentation:</strong>
            <ul>
                <li><a href="/docs">Swagger UI Documentation</a></li>
                <li><a href="/redoc">ReDoc Documentation</a></li>
                <li><a href="/api/openapi.json">OpenAPI Schema</a></li>
            </ul>
        </div>
        
        <div class="stats">
            <p><strong>Total Endpoints:</strong> {total_endpoints}</p>
            <p><strong>API Version:</strong> {app.version}</p>
            <p><strong>Server Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    """
    
    # Add endpoints by tag
    for tag in sorted(endpoints_by_tag.keys()):
        endpoints = endpoints_by_tag[tag]
        html += f"""
        <h2>{tag.upper()} <span class="tag-count">({len(endpoints)} endpoints)</span></h2>
        """
        
        for endpoint in endpoints:
            html += f"""
            <div class="endpoint">
            """
            
            for method in endpoint["methods"]:
                method_class = method.lower()
                html += f"""
                <span class="method {method_class}">{method}</span>
                <span class="path">{endpoint["path"]}</span><br>
                """
            
            if endpoint["summary"]:
                html += f"""
                <div class="summary"><strong>Summary:</strong> {endpoint["summary"]}</div>
                """
                
            if endpoint["description"]:
                html += f"""
                <div class="description"><strong>Description:</strong> {endpoint["description"]}</div>
                """
                
            html += """
            </div>
            """
    
    html += """
    </body>
    </html>
    """
    
    return html

@api_menu_router.get("/api-menu.json", summary="Get a list of all available API endpoints in JSON format")
async def get_api_menu_json(request: Request) -> Dict[str, Any]:
    """
    Get a comprehensive list of all available API endpoints organized by tags in JSON format.
    This endpoint provides a machine-readable overview of the entire API.
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
        "api_version": app.version,
        "timestamp": datetime.now().isoformat()
    } 

@router.get("/system/info")
async def info_redirect():
    """
    Redirect to /api/system/status for backward compatibility.
    """
    logger.info("Redirecting from /api/system/info to /api/system/status")
    return JSONResponse(
        status_code=status.HTTP_307_TEMPORARY_REDIRECT,
        headers={"Location": "/api/system/status"},
        content={
            "success": True,
            "message": "Endpoint moved to /api/system/status",
            "status_code": 307
        }
    )


@router.get("/system/logs")
async def logs_redirect():
    """
    Redirect to /api/system/status for backward compatibility.
    """
    logger.info("Redirecting from /api/system/logs to /api/system/status")
    return JSONResponse(
        status_code=status.HTTP_307_TEMPORARY_REDIRECT,
        headers={"Location": "/api/system/status"},
        content={
            "success": True,
            "message": "Endpoint moved to /api/system/status",
            "status_code": 307
        }
    )


@router.get("/system/config")
async def config_redirect():
    """
    Redirect to /api/system/status for backward compatibility.
    """
    logger.info("Redirecting from /api/system/config to /api/system/status")
    return JSONResponse(
        status_code=status.HTTP_307_TEMPORARY_REDIRECT,
        headers={"Location": "/api/system/status"},
        content={
            "success": True,
            "message": "Endpoint moved to /api/system/status",
            "status_code": 307
        }
    )


@router.get("/system/restart")
async def restart_redirect():
    """
    Redirect to /api/system/status for backward compatibility.
    """
    logger.info("Redirecting from /api/system/restart to /api/system/status")
    return JSONResponse(
        status_code=status.HTTP_307_TEMPORARY_REDIRECT,
        headers={"Location": "/api/system/status"},
        content={
            "success": True,
            "message": "Endpoint moved to /api/system/status",
            "status_code": 307
        }
    )


@router.get("/system/info")
async def mock_info():
    """
    Mock implementation for /api/system/info.
    """
    logger.info(f"Processing mock request for /api/system/info")
    return {
        "success": True,
        "message": "This is a mock implementation",
        "data": {},
        "timestamp": datetime.now().isoformat()
    }


@router.get("/system/logs")
async def mock_logs():
    """
    Mock implementation for /api/system/logs.
    """
    logger.info(f"Processing mock request for /api/system/logs")
    return {
        "success": True,
        "message": "This is a mock implementation",
        "data": {},
        "timestamp": datetime.now().isoformat()
    }


@router.get("/system/config")
async def mock_config():
    """
    Mock implementation for /api/system/config.
    """
    logger.info(f"Processing mock request for /api/system/config")
    return {
        "success": True,
        "message": "This is a mock implementation",
        "data": {},
        "timestamp": datetime.now().isoformat()
    }


@router.get("/system/restart")
async def mock_restart():
    """
    Mock implementation for /api/system/restart.
    """
    logger.info(f"Processing mock request for /api/system/restart")
    return {
        "success": True,
        "message": "This is a mock implementation",
        "data": {},
        "timestamp": datetime.now().isoformat()
    }

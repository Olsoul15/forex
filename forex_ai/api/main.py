"""
Main API for the Forex AI Trading System.

This module provides FastAPI routes for the web application and API endpoints.
"""

import logging
import os
from typing import Dict, List, Any, Optional
from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
    status,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
import websockets
import json
import asyncio

from forex_ai.custom_types import CurrencyPair, TimeFrame, MarketData, TradingSignal
from forex_ai.exceptions import (
    DatabaseError,
    APIError,
    AuthenticationError,
    BacktestingError,
)
from forex_ai.ui.dashboard.components.charts import ChartComponent
from forex_ai.ui.dashboard.components.strategies import StrategyComponent
from forex_ai.ui.dashboard.components.performance import PerformanceComponent
from forex_ai.ui.dashboard.components.signals import SignalComponent
from forex_ai.api.template_filters import (
    multiply,
    round_value,
    format_currency,
    format_percentage,
    format_date,
)
from forex_ai.api.auth import router as auth_router
from forex_ai.auth.supabase import get_current_user
from forex_ai.api.strategy_endpoints import (
    router as strategy_router,
    init_llm_strategy_integrator,
    init_strategy_repository,
)
from forex_ai.api.broker_routes import router as broker_router
from forex_ai.agents.framework.llm_strategy_integrator import LLMStrategyIntegrator
from forex_ai.agents.framework.agent_manager import AgentManager
from forex_ai.agents.framework.agent_types import UserQuery, SystemResponse

# Import new v1 endpoints
from forex_ai.api.v1.account_endpoints import router as account_router_v1
from forex_ai.api.v1.auto_trading_endpoints import router as auto_trading_router_v1
from forex_ai.api.v1.forex_optimizer_endpoints import router as forex_optimizer_router_v1
from forex_ai.api.v1.signal_endpoints import router as signal_router_v1
from forex_ai.api.v1.system_endpoints import router as system_router_v1, docs_router

# Import backend API endpoints
from forex_ai.backend_api.endpoints.status_endpoints_local import router as status_router
from forex_ai.backend_api.endpoints.market_data_endpoints import router as market_data_router
from forex_ai.backend_api.endpoints.account_endpoints import router as account_router
from forex_ai.backend_api.endpoints.execution_endpoints import router as execution_router
from forex_ai.backend_api.endpoints.analysis_endpoints import router as analysis_router
from forex_ai.backend_api.endpoints.day_trading_endpoints import router as day_trading_router
from forex_ai.backend_api.endpoints.docs_endpoints import router as backend_docs_router
from forex_ai.backend_api.endpoints.strategy_endpoints_local import router as strategy_router_local
from forex_ai.backend_api.endpoints.system_endpoints import router as system_router

# Configure logging
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Forex AI Trading System",
    description="API for the Forex AI Trading System",
    version="1.0.0",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    openapi_url="/api/v1/openapi.json",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Should be restricted in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include auth router
app.include_router(auth_router, prefix="/api/v1")

# Add our new strategy router
app.include_router(strategy_router)

# Add broker integration router
app.include_router(broker_router)

# Add v1 routers
app.include_router(account_router_v1)
app.include_router(auto_trading_router_v1)
app.include_router(forex_optimizer_router_v1)
app.include_router(signal_router_v1)
app.include_router(system_router_v1)

# Add API menu router at root level
app.include_router(docs_router)

# Add backend API routers
app.include_router(status_router)
app.include_router(market_data_router)
app.include_router(account_router, prefix="/api/backend")  # Add prefix to avoid conflicts
app.include_router(execution_router)
app.include_router(analysis_router)
app.include_router(day_trading_router)
app.include_router(backend_docs_router, prefix="/api/backend")  # Add prefix to avoid conflicts
app.include_router(strategy_router_local, prefix="/api/backend")  # Add prefix to avoid conflicts
app.include_router(system_router, prefix="/api/backend")  # Add prefix to avoid conflicts

# Dashboard components
chart_component = ChartComponent()
strategy_component = StrategyComponent()
performance_component = PerformanceComponent()
signal_component = SignalComponent()

# Initialize components with default configuration
config = {
    "charts": {"default_timeframe": "1h", "default_period": 14},
    "strategies": {"default_strategy": "moving_average_crossover"},
    "performance": {"default_period": "1m"},
    "signals": {"default_count": 10},
    "llm": {},
}

# Initialize components
chart_component.initialize(config.get("charts", {}))
strategy_component.initialize(config.get("strategies", {}))
performance_component.initialize(config.get("performance", {}))
signal_component.initialize(config.get("signals", {}))

# Configure templates
templates_directory = os.path.join(os.path.dirname(__file__), "..", "ui", "templates")
templates = Jinja2Templates(directory=templates_directory)

# Add template filters
templates.env.filters["multiply"] = multiply
templates.env.filters["round_value"] = round_value
templates.env.filters["format_currency"] = format_currency
templates.env.filters["format_percentage"] = format_percentage
templates.env.filters["format_date"] = format_date

# Static files
static_directory = os.path.join(os.path.dirname(__file__), "..", "ui", "static")
app.mount("/static", StaticFiles(directory=static_directory), name="static")

# Add global agent manager variable
agent_manager = None

# Create path aliases for backward compatibility
@app.get("/api/brokers/credentials", include_in_schema=False)
async def legacy_get_broker_credentials(request: Request):
    """Legacy endpoint that redirects to the v1 endpoint."""
    # Return a simple response for testing
    return JSONResponse(content={"message": "Legacy broker credentials endpoint"})

# Add a simple health check endpoint that doesn't require authentication
@app.get("/api/health", include_in_schema=True)
async def health_check():
    """
    Simple health check endpoint that doesn't require authentication.
    
    This endpoint can be used to verify that the API server is running and responsive.
    """
    logger.info("Processing health check request")
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "api_version": "v1"
    }

@app.post("/api/brokers/credentials", include_in_schema=False)
async def legacy_post_broker_credentials(request: Request):
    """Legacy endpoint that redirects to the v1 endpoint."""
    return RedirectResponse(url="/api/v1/brokers/credentials", status_code=307)

@app.delete("/api/brokers/credentials", include_in_schema=False)
async def legacy_delete_broker_credentials(request: Request):
    """Legacy endpoint that redirects to the v1 endpoint."""
    return RedirectResponse(url=f"/api/v1/brokers/credentials{request.url.query}", status_code=307)

@app.get("/api/brokers/accounts", include_in_schema=False)
async def legacy_get_broker_accounts(request: Request):
    """Legacy endpoint that redirects to the v1 endpoint."""
    return RedirectResponse(url=f"/api/v1/brokers/accounts{request.url.query}")

@app.get("/api/brokers/accounts/{account_id}", include_in_schema=False)
async def legacy_get_broker_account_info(account_id: str, request: Request):
    """Legacy endpoint that redirects to the v1 endpoint."""
    return RedirectResponse(url=f"/api/v1/brokers/accounts/{account_id}{request.url.query}")

@app.get("/api/market-data", include_in_schema=False)
async def legacy_get_market_data(request: Request):
    """Legacy endpoint that redirects to the v1 endpoint."""
    return RedirectResponse(url=f"/api/v1/market-data{request.url.query}")

@app.get("/api/strategies", include_in_schema=False)
async def legacy_get_strategies(request: Request):
    """Legacy endpoint that redirects to the v1 endpoint."""
    # Return a simple response for testing
    return JSONResponse(content={"strategies": [], "count": 0, "timestamp": datetime.now().isoformat()})

@app.get("/api/signals", include_in_schema=False)
async def legacy_get_signals(request: Request):
    """Legacy endpoint that redirects to the v1 endpoint."""
    return RedirectResponse(url=f"/api/v1/signals{request.url.query}")

@app.get("/api/performance", include_in_schema=False)
async def legacy_get_performance(request: Request):
    """Legacy endpoint that redirects to the v1 endpoint."""
    return RedirectResponse(url=f"/api/v1/account/performance{request.url.query}")

@app.get("/api/test", include_in_schema=False)
async def legacy_test_endpoint(request: Request):
    """Legacy endpoint that redirects to the v1 endpoint."""
    return RedirectResponse(url="/api/v1/health")

# Dashboard routes
@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    """
    Render the main dashboard.
    """
    try:
        # Get market data summary
        market_data = chart_component.get_market_summary()

        # Get performance metrics
        performance_metrics = performance_component.get_summary_metrics()

        # Get active signals
        active_signals = signal_component.get_active_signals()

        # Get signal summary
        signals = signal_component.get_recent_signals(limit=5)

        # Get strategies
        strategies = strategy_component.get_strategies()

        return templates.TemplateResponse(
            "dashboard.html",
            {
                "request": request,
                "market_data": market_data,
                "performance": performance_metrics,
                "signals": signals,
                "active_signals": active_signals,
                "strategies": strategies,
                "title": "Forex AI Dashboard",
                "user": {"name": "Demo User"},
                "version": "0.1.0",
            },
        )
    except Exception as e:
        logger.error(f"Error rendering dashboard: {str(e)}")

        # Provide fallback data to avoid template errors
        return templates.TemplateResponse(
            "dashboard.html",
            {
                "request": request,
                "error": str(e),
                "market_data": {
                    "major_pairs": [],
                    "market_status": "unknown",
                    "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
                "performance": {
                    "win_rate": 0,
                    "profit_factor": 0,
                    "drawdown_max": 0,
                    "sharpe_ratio": 0,
                    "total_trades": 0,
                    "period": "unknown",
                    "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
                "signals": [],
                "active_signals": [],
                "strategies": [],
                "title": "Forex AI Dashboard - Error",
                "user": {"name": "Demo User"},
                "version": "0.1.0",
            },
        )


@app.get("/backtest", response_class=HTMLResponse)
async def get_backtest_page(request: Request):
    """
    Render the backtest page.
    """
    try:
        # Get available strategies
        strategies = strategy_component.get_strategies()

        # Get currency pairs
        pairs = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "NZD/USD"]

        # Get timeframes
        timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1M"]

        return templates.TemplateResponse(
            "backtest.html",
            {
                "request": request,
                "strategies": strategies,
                "pairs": pairs,
                "timeframes": timeframes,
                "title": "Backtest Strategy",
            },
        )
    except Exception as e:
        logger.error(f"Error rendering backtest page: {str(e)}")
        return templates.TemplateResponse(
            "backtest.html",
            {
                "request": request,
                "error": str(e),
                "title": "Backtest Strategy",
            },
        )


# API endpoints
@app.get("/api/market-data", response_model=Dict[str, Any])
async def get_market_overview(
    pairs: Optional[str] = None,  # Optional query parameter
    period: Optional[str] = None,  # Optional query parameter
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Get market overview data for dashboard charts.
    Enhanced with filtering options.
    """
    try:
        # Get base market data
        market_data = chart_component.get_market_summary()

        # Import formatters
        from forex_ai.ui.dashboard.components.formatters import (
            format_market_data_for_charts,
        )

        # Apply filters if provided
        if pairs:
            requested_pairs = [p.strip() for p in pairs.split(",")]
            # Filter major_pairs to only include requested pairs
            if "major_pairs" in market_data:
                market_data["major_pairs"] = [
                    pair
                    for pair in market_data["major_pairs"]
                    if pair.get("pair", "") in requested_pairs
                ]

        # Format data for frontend charts
        formatted_data = format_market_data_for_charts(market_data)

        # Add response timestamp and cache headers
        response_data = {
            "success": True,
            "market_data": formatted_data,
            "timestamp": datetime.now().isoformat(),
        }

        return response_data
    except Exception as e:
        logger.error(f"Error getting market data: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting market data: {str(e)}",
        )


@app.get("/api/strategies", response_model=Dict[str, Any])
async def get_strategies(current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Get available trading strategies.
    """
    try:
        strategies = strategy_component.get_strategies()

        return {
            "success": True,
            "strategies": strategies,
            "count": len(strategies),
        }
    except Exception as e:
        logger.error(f"Error getting strategies: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting strategies: {str(e)}",
        )


@app.get("/api/signals", response_model=Dict[str, Any])
async def get_signals(
    limit: int = 10,
    pair: Optional[str] = None,
    direction: Optional[str] = None,
    timeframe: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Get recent trading signals.
    Enhanced with filtering options.
    """
    try:
        # Get both recent and active signals
        signals = signal_component.get_recent_signals(limit=limit)
        active_signals = signal_component.get_active_signals()

        # Import formatters
        from forex_ai.ui.dashboard.components.formatters import (
            format_signals_for_frontend,
        )

        # Apply filters if provided
        if pair or direction or timeframe:
            if pair:
                active_signals = [
                    s
                    for s in active_signals
                    if s.get("pair", "").lower() == pair.lower()
                ]
                signals = [
                    s for s in signals if s.get("pair", "").lower() == pair.lower()
                ]

            if direction:
                direction = direction.lower()
                active_signals = [
                    s
                    for s in active_signals
                    if s.get("direction", "").lower() == direction
                ]
                signals = [
                    s for s in signals if s.get("direction", "").lower() == direction
                ]

            if timeframe:
                timeframe = timeframe.lower()
                active_signals = [
                    s
                    for s in active_signals
                    if s.get("timeframe", "").lower() == timeframe
                ]
                signals = [
                    s for s in signals if s.get("timeframe", "").lower() == timeframe
                ]

        # Format signals for frontend
        formatted_active_signals = format_signals_for_frontend(active_signals)
        formatted_signals = format_signals_for_frontend(signals)

        return {
            "success": True,
            "signals": formatted_signals,
            "active_signals": formatted_active_signals,
            "count": len(signals),
            "active_count": len(active_signals),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting signals: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting signals: {str(e)}",
        )


@app.get("/api/performance", response_model=Dict[str, Any])
async def get_performance(
    period: str = "1m",
    strategy: Optional[str] = None,
    pair: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Get performance metrics.
    Enhanced with filtering options.
    """
    try:
        # Get metrics from performance component
        metrics = performance_component.get_performance_metrics(period=period)

        # Import formatters
        from forex_ai.ui.dashboard.components.formatters import (
            format_performance_for_charts,
        )

        # Prepare performance data for formatter
        performance_data = {"strategies": {}, "pairs": {}}

        # Add strategy data
        strategy_names = [
            "Hammers & Stars",
            "MA Crossover",
            "RSI Divergence",
            "Support/Resistance",
        ]

        import random

        for name in strategy_names:
            # In production, this would use real strategy metrics
            win_rate = round(random.uniform(0.55, 0.68), 2)
            profit_factor = round(random.uniform(1.4, 1.9), 1)

            # Filter by strategy if provided
            if strategy and name.lower() != strategy.lower():
                continue

            performance_data["strategies"][name] = {
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "total_trades": random.randint(80, 150),
            }

        # Add pair data
        pairs = [
            "EUR/USD",
            "GBP/USD",
            "USD/JPY",
            "USD/CHF",
            "AUD/USD",
            "USD/CAD",
            "NZD/USD",
        ]

        for p in pairs:
            # Filter by pair if provided
            if pair and p.lower() != pair.lower():
                continue

            # In production, this would use real pair performance data
            pips = random.randint(-200, 350)
            trades = random.randint(20, 50)

            performance_data["pairs"][p] = {
                "pips": pips,
                "trades": trades,
                "win_rate": round(random.uniform(0.5, 0.7), 2),
            }

        # Format data for charts
        chart_data = format_performance_for_charts(performance_data)

        return {
            "success": True,
            "metrics": metrics,
            "period": period,
            "strategy_performance": chart_data["strategy_performance"],
            "pair_performance": chart_data["pair_performance"],
            "timestamp": datetime.now().isoformat(),
        }
    except ValueError as e:
        logger.error(f"Invalid parameters: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid parameters: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Error getting performance metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting performance metrics: {str(e)}",
        )


@app.get("/strategies", response_class=HTMLResponse)
async def get_strategies_page(request: Request):
    """
    Get strategies page.
    """
    try:
        # Get strategies from component
        strategies = strategy_component.get_strategies()

        # Get current user
        user = {"name": "Demo User"}  # Default for development
        try:
            current_user = await get_current_user(request)
            if current_user:
                user = current_user
        except:
            pass

        return templates.TemplateResponse(
            "strategies.html",
            {
                "request": request,
                "user": user,
                "strategies": strategies,
                "version": "0.1.0",
            },
        )
    except Exception as e:
        logger.error(f"Error getting strategies page: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting strategies page: {str(e)}",
        )


@app.post("/api/strategy/{strategy_id}/activate", response_model=Dict[str, Any])
async def activate_strategy(
    strategy_id: str, current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Activate a strategy.
    """
    try:
        # Activate strategy
        strategy = strategy_component.activate_strategy(strategy_id)

        return {"success": True, "strategy": strategy}
    except Exception as e:
        logger.error(f"Error activating strategy: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error activating strategy: {str(e)}",
        )


@app.post("/api/strategy/{strategy_id}/deactivate", response_model=Dict[str, Any])
async def deactivate_strategy(
    strategy_id: str, current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Deactivate a strategy.
    """
    try:
        # Deactivate strategy
        strategy = strategy_component.deactivate_strategy(strategy_id)

        return {"success": True, "strategy": strategy}
    except Exception as e:
        logger.error(f"Error deactivating strategy: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deactivating strategy: {str(e)}",
        )


@app.get("/signals", response_class=HTMLResponse)
async def get_signals_page(request: Request):
    """
    Get signals page.
    """
    try:
        # Get current user
        user = {"name": "Demo User"}  # Default for development
        try:
            current_user = await get_current_user(request)
            if current_user:
                user = current_user
        except:
            pass

        return templates.TemplateResponse(
            "signals.html", {"request": request, "user": user, "version": "0.1.0"}
        )
    except Exception as e:
        logger.error(f"Error getting signals page: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting signals page: {str(e)}",
        )


@app.get("/performance", response_class=HTMLResponse)
async def get_performance_page(request: Request):
    """
    Get performance page.
    """
    try:
        # Get current user
        user = {"name": "Demo User"}  # Default for development
        try:
            current_user = await get_current_user(request)
            if current_user:
                user = current_user
        except:
            pass

        return templates.TemplateResponse(
            "performance.html", {"request": request, "user": user, "version": "0.1.0"}
        )
    except Exception as e:
        logger.error(f"Error getting performance page: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting performance page: {str(e)}",
        )


@app.get("/settings", response_class=HTMLResponse)
async def get_settings_page(request: Request):
    """
    Get settings page.
    """
    try:
        # Get current user
        user = {"name": "Demo User"}  # Default for development
        try:
            current_user = await get_current_user(request)
            if current_user:
                user = current_user
        except:
            pass

        return templates.TemplateResponse(
            "settings.html", {"request": request, "user": user, "version": "0.1.0"}
        )
    except Exception as e:
        logger.error(f"Error getting settings page: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting settings page: {str(e)}",
        )


@app.get("/brokers", response_class=HTMLResponse)
async def get_brokers_page(request: Request):
    """
    Render the broker connections page.
    """
    try:
        # Get current user
        user = {"name": "Demo User"}  # Default for development
        try:
            current_user = await get_current_user(request)
            if current_user:
                user = current_user
        except:
            pass

        return templates.TemplateResponse(
            "brokers.html",
            {
                "request": request,
                "title": "Broker Connections",
                "user": user,
                "version": "0.1.0",
                "brokers": [
                    {
                        "id": "oanda",
                        "name": "OANDA",
                        "description": "Connect to your OANDA trading account",
                        "logo": "/static/images/brokers/oanda-logo.png",
                        "supported_environments": ["practice", "live"],
                    },
                    # Add more brokers as they are supported
                ],
            },
        )
    except Exception as e:
        logger.error(f"Error rendering brokers page: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error rendering brokers page: {str(e)}",
        )


# Update the startup event handler
@app.on_event("startup")
async def startup_event():
    """Initialize components on application startup."""
    try:
        logger.info("Starting ForexAI API")

        # Initialize strategy repository
        init_strategy_repository()
        logger.info("Strategy repository initialized")

        # Initialize LLM strategy integrator with Google Vertex AI config
        llm_config = config.get("llm", {})

        # Set default Google Vertex AI configuration if not present
        if "provider" not in llm_config:
            llm_config["provider"] = "vertex"

        if "model_name" not in llm_config:
            llm_config["model_name"] = "gemini-1.5-pro"

        if "temperature" not in llm_config:
            llm_config["temperature"] = 0.2

        logger.info(f"Using LLM provider: {llm_config['provider']}")

        # Initialize the LLM strategy integrator - make this optional
        try:
            init_llm_strategy_integrator(llm_config)
            logger.info("LLM strategy integrator initialized")
        except Exception as llm_error:
            logger.warning(f"Failed to initialize LLM strategy integrator: {str(llm_error)}")
            logger.warning("Continuing without LLM capabilities")

        # Initialize agent manager
        try:
            global agent_manager
            agent_manager = AgentManager()
            logger.info("Agent manager initialized")
        except Exception as agent_error:
            logger.warning(f"Failed to initialize agent manager: {str(agent_error)}")
            logger.warning("Continuing without agent framework")

    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise


# Add a chat endpoint for the agent framework
@app.post("/api/chat", response_model=Dict[str, Any])
async def chat_with_agent(
    query: str,
    context: Dict[str, Any] = None,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Chat with the AI assistant.

    This endpoint processes user queries through the agent framework and returns
    responses from the appropriate agents.
    """
    try:
        if not agent_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Agent framework not initialized",
            )

        # Create user query
        user_query = UserQuery(query_text=query, context=context or {})

        # Get chat agent and process query
        chat_agent = agent_manager.get_agent("chat")
        response = await chat_agent.process(user_query)

        return {
            "success": True,
            "response": response.response_text,
            "timestamp": datetime.now().isoformat(),
            "source_agent": response.source_agent,
        }

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing chat: {str(e)}",
        )


# Simple test endpoint that doesn't require authentication
@app.get("/api/test", response_model=Dict[str, Any])
async def test_endpoint():
    """
    Simple test endpoint that doesn't require authentication.
    """
    return {
        "success": True,
        "message": "API server is running correctly",
        "timestamp": datetime.now().isoformat(),
    }


# WebSocket proxy routes
@app.websocket("/api/v1/ws/ohlc/{instrument}")
async def websocket_ohlc_proxy(
    websocket: WebSocket, instrument: str, timeframe: str = "M5"
):
    """Proxy WebSocket connections to the OANDA proxy service."""
    await websocket.accept()
    logger.info(f"New WebSocket connection for {instrument} ({timeframe})")

    try:
        # Connect to OANDA proxy WebSocket
        proxy_url = f"ws://localhost:8002/ws/ohlc/{instrument}?timeframe={timeframe}"
        async with websockets.connect(proxy_url) as proxy_ws:
            # Forward messages in both directions
            async def forward_to_client():
                try:
                    while True:
                        message = await proxy_ws.recv()
                        await websocket.send_text(message)
                except Exception as e:
                    logger.error(f"Error forwarding to client: {str(e)}")

            async def forward_to_proxy():
                try:
                    while True:
                        message = await websocket.receive_text()
                        await proxy_ws.send(message)
                except Exception as e:
                    logger.error(f"Error forwarding to proxy: {str(e)}")

            # Run both forwarding tasks
            forward_client = asyncio.create_task(forward_to_client())
            forward_proxy = asyncio.create_task(forward_to_proxy())

            # Wait for either task to complete (connection closed or error)
            done, pending = await asyncio.wait(
                [forward_client, forward_proxy], return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel pending tasks
            for task in pending:
                task.cancel()

    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {instrument}")
    except Exception as e:
        logger.error(f"WebSocket proxy error: {str(e)}")
        try:
            await websocket.close()
        except:
            pass


# Exception handlers
@app.exception_handler(AuthenticationError)
async def authentication_error_handler(request: Request, exc: AuthenticationError):
    """
    Handle authentication errors.
    """
    logger.error(f"Authentication error: {str(exc)}")
    return {"success": False, "error": str(exc), "error_type": "authentication_error"}


@app.exception_handler(DatabaseError)
async def database_error_handler(request: Request, exc: DatabaseError):
    """
    Handle database errors.
    """
    logger.error(f"Database error: {str(exc)}")
    return {"success": False, "error": str(exc), "error_type": "database_error"}


@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError):
    """
    Handle API errors.
    """
    logger.error(f"API error: {str(exc)}")
    return {"success": False, "error": str(exc), "error_type": "api_error"}


@app.exception_handler(BacktestingError)
async def backtest_error_handler(request: Request, exc: BacktestingError):
    """
    Handle backtesting errors.
    """
    logger.error(f"Backtesting error: {str(exc)}")
    return {"success": False, "error": str(exc), "error_type": "backtest_error"}


# Run the application directly if this module is executed
if __name__ == "__main__":
    import uvicorn
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the Forex AI Trading System API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    # Configure logging level based on debug flag
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    # Run the server
    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

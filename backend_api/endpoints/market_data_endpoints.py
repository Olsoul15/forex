"""
Market Data API Endpoints for Forex AI Trading System.

This module provides FastAPI endpoints for accessing market data.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from fastapi import (
    APIRouter,
    Query,
    Path,
    Body,
    HTTPException,
    status,
    Depends,
    BackgroundTasks,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
import time
import random
import httpx

from app.models.market_data_models import (
    TimeFrame,
    PriceType,
    InstrumentListRequest,
    PriceHistoryRequest,
    TechnicalIndicatorRequest,
    PatternDetectionRequest,
    MarketAnalysisRequest,
    PriceHistoryResponse,
    CurrentPriceResponse,
    InstrumentListResponse,
    InstrumentDetailResponse,
    TechnicalIndicatorResponse,
    PatternDetectionResponse,
    MarketAnalysisResponse,
    TechnicalLevelResponse,
    IndicatorType,
    ChartPatternType,
    StreamingEventType,
    StreamingPriceUpdate,
    StreamingHeartbeat,
    StreamingSessionChange,
)

from app.db.market_data_db import (
    get_instruments,
    get_instrument_details,
    get_current_prices,
    generate_price_history,
    generate_technical_levels,
    detect_chart_patterns,
    generate_market_analysis,
)

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/market-data", tags=["market-data"])

# Create a second router for market metrics with a different prefix
metrics_router = APIRouter(prefix="/market-metrics", tags=["market-metrics"])

# Endpoints


@router.get("/instruments", response_model=InstrumentListResponse)
async def get_instrument_list(
    type: Optional[str] = Query(None, description="Filter by instrument type"),
    tradeable_only: bool = Query(True, description="Only return tradeable instruments"),
):
    """
    Get all available tradable instruments.

    Returns a list of all instruments with optional filtering by type.
    """
    logger.info(
        f"Processing get instruments request with type={type}, tradeable_only={tradeable_only}"
    )

    # Get instruments
    instruments = get_instruments(type_filter=type, tradeable_only=tradeable_only)

    return InstrumentListResponse(
        instruments=instruments, count=len(instruments), timestamp=datetime.utcnow()
    )


@router.get("/instruments/{instrument}", response_model=InstrumentDetailResponse)
async def get_instrument_info(
    instrument: str = Path(..., description="The instrument to get details for")
):
    """
    Get detailed information about a specific instrument.

    Returns comprehensive details about the instrument including trading hours,
    margin requirements, and related instruments.
    """
    logger.info(f"Processing get instrument details request for {instrument}")

    # Get instrument details
    details = get_instrument_details(instrument)
    if not details:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Instrument {instrument} not found",
        )

    return InstrumentDetailResponse(
        instrument=details["instrument"],
        timestamp=datetime.utcnow(),
        trading_hours=details["trading_hours"],
        typical_spread=details["typical_spread"],
        margin_requirement=details["margin_requirement"],
        related_instruments=details["related_instruments"],
    )


@router.get("/prices/current", response_model=CurrentPriceResponse)
async def get_current_instrument_prices(
    instruments: Optional[List[str]] = Query(
        None, description="List of instruments to get prices for"
    )
):
    """
    Get current prices for specified instruments.

    Returns the current bid, ask, and calculated mid prices for the requested instruments.
    If no instruments are specified, returns prices for all available instruments.
    """
    logger.info(
        f"Processing get current prices request for {instruments if instruments else 'all instruments'}"
    )

    # Get current prices
    prices = get_current_prices(instruments)

    return CurrentPriceResponse(prices=prices, timestamp=datetime.utcnow())


@router.post("/prices/history", response_model=PriceHistoryResponse)
async def get_price_history(
    request: PriceHistoryRequest = Body(
        ..., description="Price history request parameters"
    )
):
    """
    Get historical price data for an instrument.

    Returns OHLC candles for the specified instrument, timeframe, and time range.
    """
    logger.info(
        f"Processing price history request for {request.instrument} on {request.timeframe}"
    )

    try:
        # Generate price history
        history = generate_price_history(
            instrument=request.instrument,
            timeframe=request.timeframe,
            from_time=request.from_time,
            to_time=request.to_time,
            count=request.count,
        )

        return PriceHistoryResponse(history=history, timestamp=datetime.utcnow())

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    except Exception as e:
        logger.error(f"Error generating price history: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error generating price history",
        )


@router.get("/ohlc/{instrument}")
async def get_historical_ohlc(
    instrument: str = Path(..., description="Trading instrument (e.g., EUR_USD)"),
    timeframe: str = Query("H1", description="Candle timeframe (M1, M5, H1, etc.)"),
    count: int = Query(100, description="Number of candles to return"),
    full_history: bool = Query(
        False, description="Whether to return all available historical data"
    ),
    max_candles: int = Query(
        5000,
        description="Maximum number of candles to return when full_history is true",
    ),
):
    """
    Get historical OHLC data for a specific instrument and timeframe.

    When full_history is true, returns as many candles as available in the system,
    limited only by the backend storage capacity or max_candles parameter.
    This is useful for initial chart loading.

    The max_candles parameter caps the number of candles returned to prevent
    overwhelming the client or network. OANDA typically limits to 5000 candles maximum.
    """
    logger.info(
        f"Processing historical OHLC request for {instrument} ({timeframe}), "
        f"count={count}, full_history={full_history}, max_candles={max_candles}"
    )

    try:
        # Format the instrument correctly
        instrument = instrument.replace("/", "_").upper()

        # Connect to TA service to get historical data
        ta_service_url = f"{TA_SERVICE_URL}/api/v1/analysis/candles/{instrument}"

        params = {
            "timeframe": timeframe,
            "full_history": "true" if full_history else "false",
        }

        if full_history:
            # Cap the maximum number of candles when requesting full history
            params["max_candles"] = min(
                max_candles, 5000
            )  # Never request more than 5000 (OANDA limit)
            logger.info(
                f"Requesting full history with max_candles={params['max_candles']}"
            )
        else:
            params["count"] = max(
                count, 100
            )  # Ensure we get at least 100 candles when not requesting full history

        async with httpx.AsyncClient(
            timeout=30.0
        ) as client:  # Increased timeout for large data requests
            response = await client.get(ta_service_url, params=params)

            if response.status_code != 200:
                logger.error(
                    f"TA service error: {response.status_code} - {response.text}"
                )
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Error fetching data from TA service: {response.text}",
                )

            data = response.json()

            # Log the number of candles received
            candles_count = len(data.get("candles", []))
            logger.info(
                f"Received {candles_count} candles from TA service for {instrument} ({timeframe})"
            )

            return {
                "instrument": instrument,
                "timeframe": timeframe,
                "candles": data.get("candles", []),
                "count": candles_count,
                "full_history": full_history,
                "max_candles": params.get("max_candles", params.get("count", 100)),
                "timestamp": datetime.utcnow().isoformat(),
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching historical OHLC data: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching historical OHLC data: {str(e)}",
        )


@router.get(
    "/technical/levels/{instrument}/{timeframe}", response_model=TechnicalLevelResponse
)
async def get_technical_levels(
    instrument: str = Path(..., description="The instrument to analyze"),
    timeframe: TimeFrame = Path(..., description="The timeframe to analyze"),
):
    """
    Get key technical levels for an instrument.

    Returns support, resistance, and pivot levels for the specified instrument and timeframe.
    """
    logger.info(f"Processing technical levels request for {instrument} on {timeframe}")

    try:
        # Generate technical levels
        levels = generate_technical_levels(instrument, timeframe)

        return TechnicalLevelResponse(
            instrument=instrument,
            timeframe=timeframe,
            levels=levels,
            timestamp=datetime.utcnow(),
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    except Exception as e:
        logger.error(f"Error generating technical levels: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error generating technical levels",
        )


@router.post("/technical/patterns", response_model=PatternDetectionResponse)
async def detect_patterns(
    request: PatternDetectionRequest = Body(
        ..., description="Pattern detection request parameters"
    )
):
    """
    Detect chart patterns in price data.

    Analyzes price data to identify common chart patterns like head and shoulders,
    double tops/bottoms, and more.
    """
    logger.info(
        f"Processing pattern detection request for {request.instrument} on {request.timeframe}"
    )

    try:
        # Detect patterns
        patterns = detect_chart_patterns(
            instrument=request.instrument,
            timeframe=request.timeframe,
            from_time=request.from_time,
            to_time=request.to_time,
            patterns=request.patterns,
            min_confidence=request.min_confidence,
        )

        return PatternDetectionResponse(
            instrument=request.instrument,
            timeframe=request.timeframe,
            patterns=patterns,
            timestamp=datetime.utcnow(),
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    except Exception as e:
        logger.error(f"Error detecting patterns: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error detecting patterns",
        )


@router.post("/analysis", response_model=MarketAnalysisResponse)
async def analyze_market(
    request: MarketAnalysisRequest = Body(
        ..., description="Market analysis request parameters"
    )
):
    """
    Get comprehensive market analysis for an instrument.

    Returns a detailed analysis including trend direction, volatility assessment,
    key levels, detected patterns, and trading signals.
    """
    logger.info(
        f"Processing market analysis request for {request.instrument} on {request.timeframe}"
    )

    try:
        # Generate market analysis
        analysis = generate_market_analysis(
            instrument=request.instrument, timeframe=request.timeframe
        )

        return MarketAnalysisResponse(analysis=analysis, timestamp=datetime.utcnow())

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    except Exception as e:
        logger.error(f"Error generating market analysis: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error generating market analysis",
        )


@router.post("/technical/indicators", response_model=TechnicalIndicatorResponse)
async def calculate_indicator(
    request: TechnicalIndicatorRequest = Body(
        ..., description="Technical indicator request parameters"
    )
):
    """
    Calculate a technical indicator for an instrument.

    Returns the values of the specified technical indicator calculated on the
    given instrument and timeframe.
    """
    logger.info(
        f"Processing indicator calculation request for {request.indicator} on {request.instrument}"
    )

    try:
        # Get price history
        history = generate_price_history(
            instrument=request.instrument,
            timeframe=request.timeframe,
            from_time=request.from_time,
            to_time=request.to_time,
            count=request.count or 100,
        )

        # Calculate indicator (mock implementation)
        values = []

        # For each candle, calculate a mock indicator value
        for i, candle in enumerate(history.candles):
            if request.indicator == IndicatorType.SMA:
                # Simple moving average - just use close price as mock
                period = request.parameters.get("period", 14)
                if i >= period - 1:
                    sma_value = (
                        sum(
                            history.candles[i - period + 1 : i + 1].close
                            for history.candles in history.candles
                        )
                        / period
                    )
                    values.append({"timestamp": candle.timestamp, "value": sma_value})
            elif request.indicator == IndicatorType.RSI:
                # Mock RSI calculation
                values.append(
                    {
                        "timestamp": candle.timestamp,
                        "value": 30
                        + (i % 70),  # Mock value that oscillates between 30 and 100
                    }
                )
            elif request.indicator == IndicatorType.BOLLINGER:
                # Mock Bollinger Bands
                middle = candle.close
                upper = middle * 1.02
                lower = middle * 0.98
                values.append(
                    {
                        "timestamp": candle.timestamp,
                        "middle": middle,
                        "upper": upper,
                        "lower": lower,
                    }
                )
            elif request.indicator == IndicatorType.MACD:
                # Mock MACD
                values.append(
                    {
                        "timestamp": candle.timestamp,
                        "macd": (i % 20) - 10,
                        "signal": (i % 15) - 7.5,
                        "histogram": ((i % 20) - 10) - ((i % 15) - 7.5),
                    }
                )
            else:
                # Generic mock indicator
                values.append({"timestamp": candle.timestamp, "value": candle.close})

        return TechnicalIndicatorResponse(
            instrument=request.instrument,
            indicator=request.indicator,
            timeframe=request.timeframe,
            from_time=history.from_time,
            to_time=history.to_time,
            values=values,
            timestamp=datetime.utcnow(),
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    except Exception as e:
        logger.error(f"Error calculating indicator: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error calculating indicator",
        )


@router.websocket("/stream/{instrument}")
async def websocket_endpoint(websocket: WebSocket, instrument: str):
    """
    WebSocket endpoint for streaming real-time price updates.

    Establishes a WebSocket connection to stream price updates for the specified instrument.
    """
    logger.info(f"WebSocket connection request for {instrument}")

    # Validate instrument
    valid_instruments = get_instruments()
    valid_instrument_names = [i.name for i in valid_instruments]

    if instrument not in valid_instrument_names:
        logger.warning(f"Invalid instrument requested: {instrument}")
        await websocket.close(code=1008)  # Policy violation
        return

    await websocket.accept()
    logger.info(f"WebSocket connection accepted for {instrument}")

    # Send connection status message
    await websocket.send_json(
        {
            "type": "CONNECTION_STATUS",
            "clientId": f"client_{int(time.time())}",
            "status": "connected",
            "instrument": instrument,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )

    try:
        # Send heartbeat every 5 seconds
        heartbeat_interval = 5
        last_heartbeat = datetime.utcnow()

        # Track current session
        current_session = "london"  # Default
        session_check_time = datetime.utcnow()

        while True:
            now = datetime.utcnow()

            # Check if we need to send a heartbeat
            if (now - last_heartbeat).total_seconds() >= heartbeat_interval:
                await websocket.send_json(
                    {"type": "HEARTBEAT", "timestamp": now.isoformat()}
                )
                last_heartbeat = now

            # Check if we need to update the session
            if (now - session_check_time).total_seconds() >= 60:  # Check every minute
                hour_utc = now.hour

                new_session = None
                if 22 <= hour_utc or hour_utc < 8:
                    new_session = "tokyo"
                elif 8 <= hour_utc < 16:
                    new_session = "london"
                elif 16 <= hour_utc < 22:
                    new_session = "new_york"

                if new_session and new_session != current_session:
                    await websocket.send_json(
                        {
                            "type": "SESSION_CHANGE",
                            "new_session": new_session,
                            "timestamp": now.isoformat(),
                        }
                    )
                    current_session = new_session

                session_check_time = now

            # Get current price for the instrument and send an update
            prices = get_current_prices([instrument])
            if prices:
                price = prices[0]
                await websocket.send_json(
                    {
                        "type": "PRICE",
                        "instrument": price.instrument,
                        "bid": price.bid,
                        "ask": price.ask,
                        "mid": price.mid,
                        "timestamp": price.timestamp.isoformat(),
                    }
                )

            # Add a small delay to prevent overwhelming the client
            import asyncio

            await asyncio.sleep(1)

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected from {instrument} stream")
    except Exception as e:
        logger.error(f"Error in WebSocket connection: {str(e)}", exc_info=True)
    finally:
        # Ensure the connection is closed
        try:
            await websocket.close()
        except Exception:
            pass


# Market Metrics endpoints


@metrics_router.get("/metrics/{instrument}")
async def get_market_metrics(
    instrument: str = Path(..., description="The instrument to get metrics for"),
    timeframe: str = Query("1H", description="Timeframe for the metrics"),
):
    """
    Get comprehensive market metrics for an instrument.

    Returns metrics including moving averages, momentum indicators, and trend strength.
    """
    logger.info(
        f"Processing market metrics request for {instrument} with timeframe {timeframe}"
    )

    # Generate random metrics for demonstration
    random_value = lambda min_val, max_val: round(random.uniform(min_val, max_val), 4)

    return {
        "instrument": instrument,
        "timeframe": timeframe,
        "timestamp": datetime.utcnow().isoformat(),
        "metrics": {
            "trend": {
                "direction": random.choice(["bullish", "bearish", "neutral"]),
                "strength": random_value(0, 100),
                "momentum": random_value(-1, 1),
            },
            "moving_averages": {
                "sma_20": random_value(1.0, 1.5),
                "sma_50": random_value(1.0, 1.5),
                "sma_200": random_value(1.0, 1.5),
                "ema_20": random_value(1.0, 1.5),
            },
            "oscillators": {
                "rsi_14": random_value(0, 100),
                "macd": random_value(-0.01, 0.01),
                "macd_signal": random_value(-0.01, 0.01),
                "stochastic_k": random_value(0, 100),
                "stochastic_d": random_value(0, 100),
            },
        },
    }


@metrics_router.get("/volatility/{instrument}")
async def get_volatility_metrics(
    instrument: str = Path(..., description="The instrument to get volatility for"),
    timeframe: str = Query("1H", description="Timeframe for volatility calculation"),
):
    """
    Get volatility metrics for an instrument.

    Returns historical volatility measures, ATR, and other volatility-related metrics.
    """
    logger.info(
        f"Processing volatility metrics request for {instrument} with timeframe {timeframe}"
    )

    # Generate random volatility data for demonstration
    random_value = lambda min_val, max_val: round(random.uniform(min_val, max_val), 6)

    return {
        "instrument": instrument,
        "timeframe": timeframe,
        "timestamp": datetime.utcnow().isoformat(),
        "volatility": {
            "daily_range_pips": round(random.uniform(30, 120)),
            "historical_volatility": {
                "daily": random_value(0.005, 0.02),
                "weekly": random_value(0.01, 0.03),
                "monthly": random_value(0.02, 0.05),
            },
            "atr": {"value": random_value(0.0005, 0.002), "period": 14},
            "bollinger_bands": {
                "upper": random_value(1.1, 1.2),
                "middle": random_value(1.05, 1.15),
                "lower": random_value(1.0, 1.1),
                "width": random_value(0.01, 0.05),
            },
            "volatility_index": random_value(0, 100),
        },
    }


@metrics_router.get("/volume/{instrument}")
async def get_volume_metrics(
    instrument: str = Path(..., description="The instrument to get volume data for"),
    timeframe: str = Query("1H", description="Timeframe for volume calculation"),
):
    """
    Get volume-related metrics for an instrument.

    Returns volume indicators, money flow index, and other volume-based metrics.
    """
    logger.info(
        f"Processing volume metrics request for {instrument} with timeframe {timeframe}"
    )

    # Generate random volume data for demonstration
    random_value = lambda min_val, max_val: round(random.uniform(min_val, max_val), 2)

    return {
        "instrument": instrument,
        "timeframe": timeframe,
        "timestamp": datetime.utcnow().isoformat(),
        "volume": {
            "current_session": round(random.uniform(10000, 100000)),
            "average_daily": round(random.uniform(50000, 500000)),
            "indicators": {
                "obv": round(random.uniform(-1000000, 1000000)),
                "mfi": random_value(0, 100),
                "volume_oscillator": random_value(-100, 100),
                "volume_ma_ratio": random_value(0.5, 1.5),
            },
            "trends": {
                "increasing_volume_bars": round(random.uniform(0, 10)),
                "volume_spike_detected": random.choice([True, False]),
                "volume_trend": random.choice(["increasing", "decreasing", "stable"]),
            },
        },
    }

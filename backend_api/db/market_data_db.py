"""
Mock Database for Market Data.

This module provides mock database functionality for market data access.
In a production environment, this would be replaced with a real database
and API calls to a broker or data provider.
"""

import logging
import random
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

from app.models.market_data_models import (
    TimeFrame,
    PriceType,
    InstrumentInfo,
    CandleData,
    PriceHistory,
    CurrentPrice,
    TechnicalLevel,
    ChartPattern,
    ChartPatternType,
    TrendType,
    VolatilityLevel,
    MarketAnalysis,
)

# Setup logging
logger = logging.getLogger(__name__)

# Mock database for instrument information
instruments_db: Dict[str, InstrumentInfo] = {}


# Initialize instruments
def initialize_instruments():
    """Initialize the instruments database with common forex pairs."""
    global instruments_db

    # Major pairs
    instruments_db["EUR_USD"] = InstrumentInfo(
        name="EUR_USD",
        display_name="EUR/USD",
        pip_location=-4,
        trade_units_precision=0,
        margin_rate=0.02,
        max_leverage=50.0,
        bid=1.09245,
        ask=1.09265,
        base_currency="EUR",
        quote_currency="USD",
    )

    instruments_db["GBP_USD"] = InstrumentInfo(
        name="GBP_USD",
        display_name="GBP/USD",
        pip_location=-4,
        trade_units_precision=0,
        margin_rate=0.02,
        max_leverage=50.0,
        bid=1.27345,
        ask=1.27375,
        base_currency="GBP",
        quote_currency="USD",
    )

    instruments_db["USD_JPY"] = InstrumentInfo(
        name="USD_JPY",
        display_name="USD/JPY",
        pip_location=-2,
        trade_units_precision=0,
        margin_rate=0.02,
        max_leverage=50.0,
        bid=148.725,
        ask=148.745,
        base_currency="USD",
        quote_currency="JPY",
    )

    # Minor pairs
    instruments_db["EUR_GBP"] = InstrumentInfo(
        name="EUR_GBP",
        display_name="EUR/GBP",
        pip_location=-4,
        trade_units_precision=0,
        margin_rate=0.02,
        max_leverage=50.0,
        bid=0.85715,
        ask=0.85735,
        base_currency="EUR",
        quote_currency="GBP",
    )

    instruments_db["AUD_USD"] = InstrumentInfo(
        name="AUD_USD",
        display_name="AUD/USD",
        pip_location=-4,
        trade_units_precision=0,
        margin_rate=0.02,
        max_leverage=50.0,
        bid=0.65815,
        ask=0.65835,
        base_currency="AUD",
        quote_currency="USD",
    )

    instruments_db["USD_CAD"] = InstrumentInfo(
        name="USD_CAD",
        display_name="USD/CAD",
        pip_location=-4,
        trade_units_precision=0,
        margin_rate=0.02,
        max_leverage=50.0,
        bid=1.37625,
        ask=1.37645,
        base_currency="USD",
        quote_currency="CAD",
    )

    # Exotic pairs
    instruments_db["USD_SGD"] = InstrumentInfo(
        name="USD_SGD",
        display_name="USD/SGD",
        pip_location=-4,
        trade_units_precision=0,
        margin_rate=0.03,
        max_leverage=33.33,
        bid=1.35515,
        ask=1.35555,
        base_currency="USD",
        quote_currency="SGD",
    )

    instruments_db["EUR_NOK"] = InstrumentInfo(
        name="EUR_NOK",
        display_name="EUR/NOK",
        pip_location=-4,
        trade_units_precision=0,
        margin_rate=0.03,
        max_leverage=33.33,
        bid=11.7655,
        ask=11.7715,
        base_currency="EUR",
        quote_currency="NOK",
    )

    # Commodity pairs
    instruments_db["XAU_USD"] = InstrumentInfo(
        name="XAU_USD",
        display_name="Gold",
        pip_location=-2,
        trade_units_precision=0,
        margin_rate=0.05,
        max_leverage=20.0,
        bid=2326.50,
        ask=2327.10,
        base_currency="XAU",
        quote_currency="USD",
        type="COMMODITY",
    )

    logger.info(f"Initialized {len(instruments_db)} instruments")


# Helper functions for price generation
def generate_price_walk(
    starting_price: float, volatility: float, trend: float, count: int
) -> List[float]:
    """
    Generate a random price walk with given parameters.

    Args:
        starting_price: Initial price
        volatility: Volatility factor (0.0-1.0)
        trend: Trend direction and strength (-1.0 to 1.0)
        count: Number of prices to generate

    Returns:
        List of generated prices
    """
    prices = [starting_price]
    for _ in range(count - 1):
        # Random component based on volatility
        random_change = random.gauss(0, 1) * volatility * starting_price * 0.01
        # Trend component
        trend_change = trend * starting_price * 0.001
        # New price
        new_price = max(0.00001, prices[-1] + random_change + trend_change)
        prices.append(new_price)

    return prices


def generate_ohlc_from_prices(
    prices: List[float], base_time: datetime, timeframe: TimeFrame
) -> List[CandleData]:
    """
    Generate OHLC candles from a list of prices.

    Args:
        prices: List of generated price points
        base_time: Starting time for the first candle
        timeframe: Timeframe for the candles

    Returns:
        List of CandleData objects
    """
    candles = []

    # Determine number of price points per candle based on timeframe
    # This is simplified - in reality we'd have varying numbers
    if timeframe == TimeFrame.M1:
        points_per_candle = 1
        time_delta = timedelta(minutes=1)
    elif timeframe == TimeFrame.M5:
        points_per_candle = 5
        time_delta = timedelta(minutes=5)
    elif timeframe == TimeFrame.M15:
        points_per_candle = 15
        time_delta = timedelta(minutes=15)
    elif timeframe == TimeFrame.M30:
        points_per_candle = 30
        time_delta = timedelta(minutes=30)
    elif timeframe == TimeFrame.H1:
        points_per_candle = 60
        time_delta = timedelta(hours=1)
    elif timeframe == TimeFrame.H4:
        points_per_candle = 240
        time_delta = timedelta(hours=4)
    elif timeframe == TimeFrame.D1:
        points_per_candle = 1440
        time_delta = timedelta(days=1)
    elif timeframe == TimeFrame.W1:
        points_per_candle = 10080
        time_delta = timedelta(weeks=1)
    else:  # MN
        points_per_candle = 40320
        time_delta = timedelta(days=30)  # Approximation

    # Create candles
    for i in range(0, len(prices), points_per_candle):
        if i + points_per_candle <= len(prices):
            candle_prices = prices[i : i + points_per_candle]

            # Generate random volume
            volume = random.uniform(10000, 100000)

            candle = CandleData(
                open=candle_prices[0],
                high=max(candle_prices),
                low=min(candle_prices),
                close=candle_prices[-1],
                volume=volume,
                timestamp=base_time + (i // points_per_candle) * time_delta,
                complete=True,
            )

            candles.append(candle)

    return candles


# Generate price history for an instrument
def generate_price_history(
    instrument: str,
    timeframe: TimeFrame,
    from_time: Optional[datetime] = None,
    to_time: Optional[datetime] = None,
    count: Optional[int] = None,
) -> PriceHistory:
    """
    Generate mock price history for an instrument.

    Args:
        instrument: Instrument name
        timeframe: Timeframe for the data
        from_time: Start time (optional)
        to_time: End time (optional)
        count: Number of candles (optional)

    Returns:
        PriceHistory object containing the generated data
    """
    # Get instrument info
    if instrument not in instruments_db:
        raise ValueError(f"Instrument {instrument} not found")

    instrument_info = instruments_db[instrument]

    # Determine time range and count
    now = datetime.utcnow()

    if to_time is None:
        to_time = now

    if from_time is None and count is None:
        # Default to 100 candles
        count = 100

    if count is not None:
        # Calculate from_time based on count and timeframe
        if timeframe == TimeFrame.M1:
            from_time = to_time - timedelta(minutes=count)
        elif timeframe == TimeFrame.M5:
            from_time = to_time - timedelta(minutes=5 * count)
        elif timeframe == TimeFrame.M15:
            from_time = to_time - timedelta(minutes=15 * count)
        elif timeframe == TimeFrame.M30:
            from_time = to_time - timedelta(minutes=30 * count)
        elif timeframe == TimeFrame.H1:
            from_time = to_time - timedelta(hours=count)
        elif timeframe == TimeFrame.H4:
            from_time = to_time - timedelta(hours=4 * count)
        elif timeframe == TimeFrame.D1:
            from_time = to_time - timedelta(days=count)
        elif timeframe == TimeFrame.W1:
            from_time = to_time - timedelta(weeks=count)
        else:  # MN
            from_time = to_time - timedelta(days=30 * count)
    else:
        # Calculate count based on from_time and to_time
        if timeframe == TimeFrame.M1:
            count = int((to_time - from_time).total_seconds() / 60)
        elif timeframe == TimeFrame.M5:
            count = int((to_time - from_time).total_seconds() / 300)
        elif timeframe == TimeFrame.M15:
            count = int((to_time - from_time).total_seconds() / 900)
        elif timeframe == TimeFrame.M30:
            count = int((to_time - from_time).total_seconds() / 1800)
        elif timeframe == TimeFrame.H1:
            count = int((to_time - from_time).total_seconds() / 3600)
        elif timeframe == TimeFrame.H4:
            count = int((to_time - from_time).total_seconds() / 14400)
        elif timeframe == TimeFrame.D1:
            count = int((to_time - from_time).total_seconds() / 86400)
        elif timeframe == TimeFrame.W1:
            count = int((to_time - from_time).total_seconds() / 604800)
        else:  # MN
            count = int((to_time - from_time).total_seconds() / 2592000)

    # Generate more price points than needed to create realistic OHLC data
    extra_points = 10  # Number of price points per candle

    # Determine trend and volatility based on instrument
    # In a real implementation, we'd use market analysis
    volatility = random.uniform(0.1, 0.5)  # Lower for majors, higher for exotics
    trend = random.uniform(-0.2, 0.2)  # Random trend

    # Generate the price walk
    mid_price = (instrument_info.bid + instrument_info.ask) / 2
    prices = generate_price_walk(mid_price, volatility, trend, count * extra_points)

    # Generate OHLC data
    candles = generate_ohlc_from_prices(prices, from_time, timeframe)

    # Create price history object
    history = PriceHistory(
        instrument=instrument,
        granularity=timeframe,
        candles=candles,
        complete=True,
        from_time=from_time,
        to_time=to_time,
    )

    return history


# Get current prices for instruments
def get_current_prices(instruments: Optional[List[str]] = None) -> List[CurrentPrice]:
    """
    Get current prices for the specified instruments.

    Args:
        instruments: List of instrument names (optional, defaults to all)

    Returns:
        List of CurrentPrice objects
    """
    if instruments is None:
        # Use all instruments
        instruments = list(instruments_db.keys())

    current_prices = []
    now = datetime.utcnow()

    for instrument_name in instruments:
        if instrument_name in instruments_db:
            instrument = instruments_db[instrument_name]

            # Add some randomness to the prices
            bid_change = random.uniform(-0.0005, 0.0005) * instrument.bid
            bid = instrument.bid + bid_change

            # Ensure ask is always higher than bid
            spread_factor = random.uniform(0.8, 1.2)  # Vary the spread a bit
            original_spread = instrument.ask - instrument.bid
            ask = bid + original_spread * spread_factor

            # Update the stored prices
            instrument.bid = bid
            instrument.ask = ask

            current_price = CurrentPrice(
                instrument=instrument_name, bid=bid, ask=ask, timestamp=now
            )

            current_prices.append(current_price)

    return current_prices


# Get list of instruments
def get_instruments(
    type_filter: Optional[str] = None, tradeable_only: bool = True
) -> List[InstrumentInfo]:
    """
    Get list of available instruments.

    Args:
        type_filter: Filter by instrument type (optional)
        tradeable_only: Only return tradeable instruments (default True)

    Returns:
        List of InstrumentInfo objects
    """
    filtered_instruments = []

    for instrument in instruments_db.values():
        # Apply filters
        if tradeable_only and not instrument.tradeable:
            continue

        if type_filter and instrument.type != type_filter:
            continue

        filtered_instruments.append(instrument)

    return filtered_instruments


# Get detailed information about an instrument
def get_instrument_details(instrument_name: str) -> Optional[Dict[str, Any]]:
    """
    Get detailed information about an instrument.

    Args:
        instrument_name: Instrument name

    Returns:
        Dictionary with instrument details or None if not found
    """
    if instrument_name not in instruments_db:
        return None

    instrument = instruments_db[instrument_name]

    # Additional details for the instrument
    details = {
        "instrument": instrument,
        "trading_hours": {
            "sunday": {"open": "21:00", "close": "23:59"},
            "monday": {"open": "00:00", "close": "23:59"},
            "tuesday": {"open": "00:00", "close": "23:59"},
            "wednesday": {"open": "00:00", "close": "23:59"},
            "thursday": {"open": "00:00", "close": "23:59"},
            "friday": {"open": "00:00", "close": "21:00"},
            "saturday": {"open": None, "close": None},
        },
        "typical_spread": round(instrument.ask - instrument.bid, 5),
        "margin_requirement": instrument.margin_rate * 100,  # Convert to percentage
        "related_instruments": [],
    }

    # Add related instruments
    if instrument.base_currency == "EUR":
        details["related_instruments"].append("EUR_GBP")
        details["related_instruments"].append("EUR_JPY")
    elif instrument.base_currency == "USD":
        details["related_instruments"].append("EUR_USD")
        details["related_instruments"].append("GBP_USD")

    return details


# Generate technical levels for an instrument
def generate_technical_levels(
    instrument: str, timeframe: TimeFrame
) -> List[TechnicalLevel]:
    """
    Generate mock technical levels for an instrument.

    Args:
        instrument: Instrument name
        timeframe: Timeframe

    Returns:
        List of TechnicalLevel objects
    """
    if instrument not in instruments_db:
        raise ValueError(f"Instrument {instrument} not found")

    instrument_info = instruments_db[instrument]

    # Get price history to generate realistic levels
    history = generate_price_history(
        instrument=instrument, timeframe=timeframe, count=100  # Use last 100 candles
    )

    levels = []

    # Get min and max prices
    all_prices = []
    for candle in history.candles:
        all_prices.extend([candle.high, candle.low])

    min_price = min(all_prices)
    max_price = max(all_prices)
    price_range = max_price - min_price

    # Generate support levels
    current_price = (instrument_info.bid + instrument_info.ask) / 2

    # Support levels (below current price)
    for i in range(1, 4):
        # Generate a level at a realistic distance
        level_price = current_price - (price_range * random.uniform(0.1, 0.4) * i)
        strength = random.uniform(0.6, 0.9)  # Random strength

        level = TechnicalLevel(
            level_type="support",
            price=round(level_price, 5),
            strength=round(strength, 2),
            description=f"Support level {i}",
        )

        levels.append(level)

    # Resistance levels (above current price)
    for i in range(1, 4):
        # Generate a level at a realistic distance
        level_price = current_price + (price_range * random.uniform(0.1, 0.4) * i)
        strength = random.uniform(0.6, 0.9)  # Random strength

        level = TechnicalLevel(
            level_type="resistance",
            price=round(level_price, 5),
            strength=round(strength, 2),
            description=f"Resistance level {i}",
        )

        levels.append(level)

    # Add a pivot point near current price
    pivot = TechnicalLevel(
        level_type="pivot",
        price=round(current_price, 5),
        strength=0.85,
        description="Daily pivot point",
    )

    levels.append(pivot)

    return levels


# Detect chart patterns in price data
def detect_chart_patterns(
    instrument: str,
    timeframe: TimeFrame,
    from_time: Optional[datetime] = None,
    to_time: Optional[datetime] = None,
    patterns: Optional[List[ChartPatternType]] = None,
    min_confidence: float = 0.7,
) -> List[ChartPattern]:
    """
    Detect mock chart patterns in price data.

    Args:
        instrument: Instrument name
        timeframe: Timeframe
        from_time: Start time (optional)
        to_time: End time (optional)
        patterns: Specific patterns to look for (optional)
        min_confidence: Minimum confidence level for patterns

    Returns:
        List of ChartPattern objects
    """
    # Get price history
    history = generate_price_history(
        instrument=instrument,
        timeframe=timeframe,
        from_time=from_time,
        to_time=to_time,
        count=100 if from_time is None and to_time is None else None,
    )

    detected_patterns = []

    # Filter patterns if specified
    available_patterns = list(ChartPatternType)
    if patterns:
        available_patterns = [p for p in available_patterns if p in patterns]

    # Number of patterns to detect (random)
    num_patterns = random.randint(0, 3)

    for _ in range(num_patterns):
        # Select a random pattern type
        pattern_type = random.choice(available_patterns)

        # Generate a random confidence above the minimum
        confidence = random.uniform(min_confidence, 1.0)

        # Select random start and end times from the history
        candle_indices = random.sample(range(len(history.candles)), 2)
        candle_indices.sort()

        start_time = history.candles[candle_indices[0]].timestamp
        end_time = history.candles[candle_indices[1]].timestamp

        # Generate target price and stop loss based on the pattern type
        last_close = history.candles[-1].close

        if pattern_type in [
            ChartPatternType.DOUBLE_BOTTOM,
            ChartPatternType.INV_HEAD_SHOULDERS,
        ]:
            # Bullish patterns
            target_price = last_close * (1 + random.uniform(0.01, 0.05))
            stop_loss = last_close * (1 - random.uniform(0.005, 0.02))
        elif pattern_type in [
            ChartPatternType.DOUBLE_TOP,
            ChartPatternType.HEAD_SHOULDERS,
        ]:
            # Bearish patterns
            target_price = last_close * (1 - random.uniform(0.01, 0.05))
            stop_loss = last_close * (1 + random.uniform(0.005, 0.02))
        else:
            # Neutral patterns
            direction = random.choice([-1, 1])
            target_price = last_close * (1 + direction * random.uniform(0.01, 0.04))
            stop_loss = last_close * (1 - direction * random.uniform(0.005, 0.02))

        pattern = ChartPattern(
            pattern_type=pattern_type,
            start_time=start_time,
            end_time=end_time,
            confidence=round(confidence, 2),
            target_price=round(target_price, 5),
            stop_loss=round(stop_loss, 5),
            description=f"{pattern_type.value.replace('_', ' ').title()} pattern detected",
        )

        detected_patterns.append(pattern)

    return detected_patterns


# Generate a complete market analysis
def generate_market_analysis(instrument: str, timeframe: TimeFrame) -> MarketAnalysis:
    """
    Generate a comprehensive market analysis for an instrument.

    Args:
        instrument: Instrument name
        timeframe: Timeframe

    Returns:
        MarketAnalysis object
    """
    if instrument not in instruments_db:
        raise ValueError(f"Instrument {instrument} not found")

    # Get technical levels and patterns
    levels = generate_technical_levels(instrument, timeframe)
    patterns = detect_chart_patterns(instrument, timeframe)

    # Determine trend and volatility
    trend_options = list(TrendType)
    volatility_options = list(VolatilityLevel)

    trend = random.choice(trend_options)
    volatility = random.choice(volatility_options)

    # Generate trading signals
    signals = []

    signal_types = ["buy", "sell", "neutral"]
    signal_strengths = ["strong", "moderate", "weak"]

    # Generate 1-3 signals
    num_signals = random.randint(1, 3)

    for _ in range(num_signals):
        signal_type = random.choice(signal_types)
        strength = random.choice(signal_strengths)

        # Generate an explanation
        if signal_type == "buy":
            explanation = random.choice(
                [
                    "Price bounced off support level",
                    "RSI indicates oversold conditions",
                    "Bullish engulfing pattern detected",
                    "MACD crossed above signal line",
                ]
            )
        elif signal_type == "sell":
            explanation = random.choice(
                [
                    "Price rejected at resistance level",
                    "RSI indicates overbought conditions",
                    "Bearish engulfing pattern detected",
                    "MACD crossed below signal line",
                ]
            )
        else:
            explanation = random.choice(
                [
                    "Price is in a consolidation phase",
                    "Mixed signals from indicators",
                    "Awaiting breakout confirmation",
                    "Low volatility environment",
                ]
            )

        signal = {
            "type": signal_type,
            "strength": strength,
            "explanation": explanation,
            "indicator": random.choice(
                ["RSI", "MACD", "Moving Averages", "Pattern Recognition"]
            ),
        }

        signals.append(signal)

    # Generate a summary
    summary_parts = []

    # Trend part
    if trend in [TrendType.STRONG_UPTREND, TrendType.UPTREND]:
        summary_parts.append(
            f"{instrument} is in an uptrend on the {timeframe} timeframe."
        )
    elif trend in [TrendType.STRONG_DOWNTREND, TrendType.DOWNTREND]:
        summary_parts.append(
            f"{instrument} is in a downtrend on the {timeframe} timeframe."
        )
    else:
        summary_parts.append(
            f"{instrument} is consolidating on the {timeframe} timeframe."
        )

    # Volatility part
    if volatility in [VolatilityLevel.HIGH, VolatilityLevel.VERY_HIGH]:
        summary_parts.append(
            "Volatility is elevated, suggesting potential for significant price movements."
        )
    elif volatility == VolatilityLevel.MODERATE:
        summary_parts.append(
            "Volatility is moderate, with typical price movements expected."
        )
    else:
        summary_parts.append("Volatility is low, suggesting a range-bound environment.")

    # Support/resistance part
    if levels:
        key_support = min(
            [l.price for l in levels if l.level_type == "support"], default=None
        )
        key_resistance = min(
            [l.price for l in levels if l.level_type == "resistance"], default=None
        )

        if key_support and key_resistance:
            summary_parts.append(
                f"Key support at {key_support} and resistance at {key_resistance}."
            )

    # Patterns part
    if patterns:
        pattern_names = [
            p.pattern_type.value.replace("_", " ").title() for p in patterns
        ]
        if len(pattern_names) == 1:
            summary_parts.append(f"A {pattern_names[0]} pattern has been detected.")
        else:
            summary_parts.append(
                f"Several patterns detected including {', '.join(pattern_names[:-1])} and {pattern_names[-1]}."
            )

    # Signals part
    buy_signals = [s for s in signals if s["type"] == "buy"]
    sell_signals = [s for s in signals if s["type"] == "sell"]

    if buy_signals and not sell_signals:
        summary_parts.append(
            "Technical indicators generally suggest bullish conditions."
        )
    elif sell_signals and not buy_signals:
        summary_parts.append(
            "Technical indicators generally suggest bearish conditions."
        )
    elif buy_signals and sell_signals:
        summary_parts.append("Mixed signals from technical indicators suggest caution.")

    summary = " ".join(summary_parts)

    # Create the analysis object
    analysis = MarketAnalysis(
        instrument=instrument,
        timestamp=datetime.utcnow(),
        timeframe=timeframe,
        trend=trend,
        volatility=volatility,
        support_levels=[l for l in levels if l.level_type == "support"],
        resistance_levels=[l for l in levels if l.level_type == "resistance"],
        patterns=patterns,
        trading_signals=signals,
        summary=summary,
    )

    return analysis


# Initialize the database
initialize_instruments()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AutoAgent Orchestrator for AI Forex Trading

This module provides the orchestration layer for the AutoAgent integration with
the AI Forex trading system. It manages market analysis workflows and coordinates
the flow of information between different components.
"""

import logging
import datetime
import asyncio
from typing import Dict, List, Any, Optional, Union

from forex_ai.integration.enhanced_memory_manager import EnhancedMemoryManager

logger = logging.getLogger(__name__)


class AutoAgentOrchestrator:
    """
    Orchestrates market analysis workflows using AutoAgent integration.

    This class coordinates multiple analysis components, manages context across
    analysis sessions, and generates trading signals based on comprehensive analysis.
    """

    def __init__(self, config: Dict[str, Any], memory_manager: Optional[EnhancedMemoryManager] = None):
        """
        Initialize the AutoAgent orchestrator.

        Args:
            config: Configuration dictionary with the following keys:
                - config: AnalysisConfig object with analysis parameters
                - data_fetcher: MarketDataFetcher instance for retrieving market data
            memory_manager: Optional instance of EnhancedMemoryManager.
        """
        self.config = config.get("config", {})
        self.data_fetcher = config.get("data_fetcher")
        self.confidence_threshold = config.get("confidence_threshold", 0.65)
        self.memory_manager = memory_manager or EnhancedMemoryManager()

        logger.info("AutoAgentOrchestrator initialized with config: %s", config)

    async def analyze_market(
        self, instrument: str, timeframe: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze market conditions for the specified instrument.

        Args:
            instrument: Trading pair to analyze (e.g., "EUR_USD")
            timeframe: Optional timeframe to analyze (e.g., "H1", "H4")
                       If None, will analyze all timeframes in config

        Returns:
            Dict containing analysis results with keys:
                - market_view: Overall market direction and confidence
                - insights: List of insights from the analysis
                - signals: Trading signals generated (if any)
                - support_resistance: Support and resistance levels
        """
        logger.info(f"Analyzing market for {instrument} on timeframe {timeframe}")

        timeframes = (
            [timeframe]
            if timeframe
            else getattr(self.config, "timeframes", ["H1", "H4"])
        )
        results = {
            "instrument": instrument,
            "timestamp": datetime.datetime.now().isoformat(),
            "market_view": {},
            "insights": [],
            "signals": [],
            "support_resistance": {},
        }

        # Process each timeframe concurrently
        timeframe_analyses = await asyncio.gather(
            *[self._analyze_timeframe(instrument, tf) for tf in timeframes]
        )

        for tf_results in timeframe_analyses:
            results["insights"].extend(tf_results.get("insights", []))
            # Add support and resistance levels
            if tf_results.get("support_resistance"):
                tf = tf_results.get("timeframe")
                if tf:
                    results["support_resistance"][tf] = tf_results["support_resistance"]

        # Generate overall market view by aggregating timeframe analyses
        results["market_view"] = self._generate_market_view(results["insights"])

        # Generate trading signals if confidence is high enough
        if results["market_view"].get("confidence", 0) >= self.confidence_threshold:
            signals = self._generate_signals(instrument, results)
            results["signals"] = signals

        # Store context for future reference
        await self._update_context(instrument, results)

        return results

    async def _analyze_timeframe(self, instrument: str, timeframe: str) -> Dict[str, Any]:
        """
        Analyze a specific timeframe for the given instrument.

        Args:
            instrument: Trading pair to analyze
            timeframe: Timeframe to analyze

        Returns:
            Dict with analysis results for the timeframe
        """
        logger.debug(f"Analyzing {instrument} on {timeframe} timeframe")

        # Fetch market data for the timeframe
        market_data = None
        if self.data_fetcher:
            market_data = await self.data_fetcher.get_candles(
                instrument=instrument,
                timeframe=timeframe,
                count=100,  # Use configuration or reasonable default
            )
        else:
            # Use mock data for testing if no data fetcher is provided
            market_data = self._generate_mock_data(instrument, timeframe)

        # Extract technical indicators
        indicators = self._extract_indicators(market_data)

        # Calculate support and resistance levels
        support_resistance = self._get_support_resistance(market_data)

        # Generate insights based on indicators and support/resistance
        insights = self._generate_insights(
            instrument, timeframe, indicators, support_resistance
        )

        return {
            "timeframe": timeframe,
            "indicators": indicators,
            "support_resistance": support_resistance,
            "insights": insights,
        }

    def _extract_indicators(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract technical indicators from market data.

        Args:
            market_data: Dictionary containing OHLCV data

        Returns:
            Dictionary containing calculated indicators
        """
        # Process market data to extract indicators
        if not market_data or "candles" not in market_data:
            logger.warning("No market data available for indicator extraction")
            return {}

        candles = market_data.get("candles", [])
        if not candles:
            logger.warning("Empty candles list in market data")
            return {}

        # Extract prices for calculation
        closes = [float(candle["mid"]["c"]) for candle in candles if "mid" in candle]
        highs = [float(candle["mid"]["h"]) for candle in candles if "mid" in candle]
        lows = [float(candle["mid"]["l"]) for candle in candles if "mid" in candle]

        if not closes:
            logger.warning("No close prices available in candles")
            return {}

        # Calculate RSI
        try:
            rsi_value = self._calculate_rsi(closes, period=14)
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            rsi_value = None

        # Calculate MACD
        try:
            macd, signal, hist = self._calculate_macd(closes)
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            macd, signal, hist = None, None, None

        # Calculate Bollinger Bands
        try:
            upper, middle, lower = self._calculate_bollinger_bands(closes)
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            upper, middle, lower = None, None, None

        # Calculate current price and price action metrics
        current_price = closes[-1] if closes else None

        # Handle None values safely
        if rsi_value is None:
            rsi_value = 50  # Neutral default

        if hist is None:
            hist = 0  # Neutral default

        return {
            "rsi": rsi_value,
            "macd": {"macd": macd, "signal": signal, "histogram": hist},
            "bollinger_bands": {"upper": upper, "middle": middle, "lower": lower},
            "current_price": current_price,
            "closes": closes,
            "highs": highs,
            "lows": lows,
        }

    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """
        Calculate the Relative Strength Index using a smoothed moving average.

        Args:
            prices: List of closing prices
            period: RSI period (default: 14)

        Returns:
            RSI value, or 50 if not enough data.
        """
        if len(prices) < period + 1:
            return 50.0

        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [delta if delta > 0 else 0 for delta in deltas]
        losses = [-delta if delta < 0 else 0 for delta in deltas]

        # Calculate initial average gain and loss
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        # Smooth subsequent gains and losses
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    def _calculate_macd(
        self, prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9
    ) -> tuple:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        Args:
            prices: List of closing prices
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period

        Returns:
            Tuple of the latest (MACD line, signal line, histogram) values.
        """
        if len(prices) < slow:
            return 0, 0, 0

        ema_fast = self._calculate_ema(prices, period=fast)
        ema_slow = self._calculate_ema(prices, period=slow)

        if not ema_fast or not ema_slow:
            return 0, 0, 0

        # Align the fast EMA with the slow EMA
        ema_fast_aligned = ema_fast[slow - fast:]

        macd_line = [f - s for f, s in zip(ema_fast_aligned, ema_slow)]

        if len(macd_line) < signal:
            return 0, 0, 0

        signal_line = self._calculate_ema(macd_line, signal)

        # Align MACD and signal lines
        hist_start_offset = len(macd_line) - len(signal_line)
        histogram = [m - s for m, s in zip(macd_line[hist_start_offset:], signal_line)]

        latest_macd = macd_line[-1] if macd_line else None
        latest_signal = signal_line[-1] if signal_line else None
        latest_hist = histogram[-1] if histogram else None

        logger.debug(f"MACD calculation: len(prices)={len(prices)}")
        logger.debug(f"MACD calculation: len(ema_fast)={len(ema_fast)}, len(ema_slow)={len(ema_slow)}")
        logger.debug(f"MACD calculation: len(macd_line)={len(macd_line)}")
        logger.debug(f"MACD calculation: len(signal_line)={len(signal_line)}")
        logger.debug(f"MACD calculation: len(histogram)={len(histogram)}")
        logger.debug(f"MACD calculation: latest_macd={latest_macd}, latest_signal={latest_signal}, latest_hist={latest_hist}")

        return latest_macd, latest_signal, latest_hist

    def _calculate_ema(self, prices: List[float], period: int) -> List[float]:
        """
        Calculate the Exponential Moving Average (EMA).

        Args:
            prices: List of closing prices
            period: EMA period

        Returns:
            A list of EMA values starting from the first possible period.
        """
        if len(prices) < period:
            return []
        
        ema_values = []
        multiplier = 2 / (period + 1)
        
        # Calculate the initial SMA
        sma = sum(prices[:period]) / period
        ema_values.append(sma)
        
        # Calculate subsequent EMA values
        for price in prices[period:]:
            ema = (price - ema_values[-1]) * multiplier + ema_values[-1]
            ema_values.append(ema)
            
        return ema_values

    def _calculate_bollinger_bands(
        self, prices: List[float], period: int = 20, deviation: int = 2
    ) -> tuple:
        """
        Calculate Bollinger Bands.

        Args:
            prices: List of closing prices
            period: Period for moving average (default: 20)
            deviation: Number of standard deviations (default: 2)

        Returns:
            Tuple of (upper band, middle band, lower band)
        """
        if len(prices) < period:
            return None, None, None  # Not enough data

        # Calculate middle band (SMA)
        middle = sum(prices[-period:]) / period

        # Calculate standard deviation
        variance = sum([(price - middle) ** 2 for price in prices[-period:]]) / period
        std_dev = variance**0.5

        # Calculate upper and lower bands
        upper = middle + (std_dev * deviation)
        lower = middle - (std_dev * deviation)

        return upper, middle, lower

    def _get_support_resistance(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate support and resistance levels.

        Args:
            market_data: Dictionary containing OHLCV data

        Returns:
            Dictionary containing support and resistance levels
        """
        if not market_data or "candles" not in market_data:
            logger.warning(
                "No market data available for support/resistance calculation"
            )
            return {}

        candles = market_data.get("candles", [])
        if not candles:
            return {}

        # Extract price data
        highs = [float(candle["mid"]["h"]) for candle in candles if "mid" in candle]
        lows = [float(candle["mid"]["l"]) for candle in candles if "mid" in candle]
        closes = [float(candle["mid"]["c"]) for candle in candles if "mid" in candle]

        if not highs or not lows or not closes:
            return {}

        # Use last N values for calculation (or all if fewer than 10)
        high_val = max(highs[-10:] if len(highs) >= 10 else highs)
        low_val = min(lows[-10:] if len(lows) >= 10 else lows)
        close_val = closes[-1] if closes else None

        # Simple pivot point calculation
        if high_val is not None and low_val is not None and close_val is not None:
            pivot = (high_val + low_val + close_val) / 3
            r1 = (2 * pivot) - low_val
            r2 = pivot + (high_val - low_val)
            s1 = (2 * pivot) - high_val
            s2 = pivot - (high_val - low_val)

            return {
                "pivot": pivot,
                "resistance_levels": [r1, r2],
                "support_levels": [s1, s2],
            }

        return {}

    def _generate_insights(
        self,
        instrument: str,
        timeframe: str,
        indicators: Dict[str, Any],
        support_resistance: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Generate insights based on technical indicators and support/resistance levels.

        Args:
            instrument: Trading pair
            timeframe: Timeframe being analyzed
            indicators: Dictionary of technical indicators
            support_resistance: Dictionary of support and resistance levels

        Returns:
            List of insight dictionaries
        """
        insights = []

        # Current price
        current_price = indicators.get("current_price")
        if current_price is None:
            logger.warning("No current price available for insight generation")
            return insights

        # RSI insights
        rsi = indicators.get("rsi")
        if rsi is not None:
            if rsi > 70:
                insights.append(
                    {
                        "timeframe": timeframe,
                        "instrument": instrument,
                        "indicator": "RSI",
                        "value": rsi,
                        "message": f"RSI is overbought at {rsi:.2f}",
                        "sentiment": "bearish",
                    }
                )
            elif rsi < 30:
                insights.append(
                    {
                        "timeframe": timeframe,
                        "instrument": instrument,
                        "indicator": "RSI",
                        "value": rsi,
                        "message": f"RSI is oversold at {rsi:.2f}",
                        "sentiment": "bullish",
                    }
                )
            else:
                insights.append(
                    {
                        "timeframe": timeframe,
                        "instrument": instrument,
                        "indicator": "RSI",
                        "value": rsi,
                        "message": f"RSI is neutral at {rsi:.2f}",
                        "sentiment": "neutral",
                    }
                )

        # MACD insights
        macd_data = indicators.get("macd", {})
        macd_hist = macd_data.get("histogram")
        if macd_hist is not None:
            if macd_hist > 0:
                insights.append(
                    {
                        "timeframe": timeframe,
                        "instrument": instrument,
                        "indicator": "MACD",
                        "value": macd_hist,
                        "message": f"MACD histogram is positive at {macd_hist:.6f}",
                        "sentiment": "bullish",
                    }
                )
            elif macd_hist < 0:
                insights.append(
                    {
                        "timeframe": timeframe,
                        "instrument": instrument,
                        "indicator": "MACD",
                        "value": macd_hist,
                        "message": f"MACD histogram is negative at {macd_hist:.6f}",
                        "sentiment": "bearish",
                    }
                )
            else:
                insights.append(
                    {
                        "timeframe": timeframe,
                        "instrument": instrument,
                        "indicator": "MACD",
                        "value": macd_hist,
                        "message": f"MACD histogram is at {macd_hist:.6f}",
                        "sentiment": "neutral",
                    }
                )

        # Bollinger Bands insights
        bb = indicators.get("bollinger_bands", {})
        upper = bb.get("upper")
        lower = bb.get("lower")
        if upper is not None and lower is not None and current_price is not None:
            if current_price > upper:
                insights.append(
                    {
                        "timeframe": timeframe,
                        "instrument": instrument,
                        "indicator": "Bollinger Bands",
                        "value": current_price,
                        "message": f"Price ({current_price:.5f}) is above upper Bollinger Band ({upper:.5f})",
                        "sentiment": "bearish",
                    }
                )
            elif current_price < lower:
                insights.append(
                    {
                        "timeframe": timeframe,
                        "instrument": instrument,
                        "indicator": "Bollinger Bands",
                        "value": current_price,
                        "message": f"Price ({current_price:.5f}) is below lower Bollinger Band ({lower:.5f})",
                        "sentiment": "bullish",
                    }
                )
            else:
                insights.append(
                    {
                        "timeframe": timeframe,
                        "instrument": instrument,
                        "indicator": "Bollinger Bands",
                        "value": current_price,
                        "message": f"Price ({current_price:.5f}) is within Bollinger Bands",
                        "sentiment": "neutral",
                    }
                )

        # Support and Resistance insights
        sr = support_resistance
        if sr and current_price:
            # Ensure support and resistance levels are lists
            support_levels = sr.get("support_levels", [])
            resistance_levels = sr.get("resistance_levels", [])

            if not isinstance(support_levels, list):
                support_levels = [support_levels] if support_levels is not None else []

            if not isinstance(resistance_levels, list):
                resistance_levels = (
                    [resistance_levels] if resistance_levels is not None else []
                )

            # Filter out None values
            support_levels = [level for level in support_levels if level is not None]
            resistance_levels = [
                level for level in resistance_levels if level is not None
            ]

            # Find closest support and resistance
            if support_levels:
                closest_support = max(
                    [s for s in support_levels if s < current_price], default=None
                )
                if closest_support:
                    insights.append(
                        {
                            "timeframe": timeframe,
                            "instrument": instrument,
                            "indicator": "Support",
                            "value": closest_support,
                            "message": f"Closest support at {closest_support:.5f}",
                            "sentiment": "neutral",
                        }
                    )

            if resistance_levels:
                closest_resistance = min(
                    [r for r in resistance_levels if r > current_price], default=None
                )
                if closest_resistance:
                    insights.append(
                        {
                            "timeframe": timeframe,
                            "instrument": instrument,
                            "indicator": "Resistance",
                            "value": closest_resistance,
                            "message": f"Closest resistance at {closest_resistance:.5f}",
                            "sentiment": "neutral",
                        }
                    )

        return insights

    def _generate_market_view(self, insights: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate overall market view from aggregated insights.

        Args:
            insights: List of insight dictionaries

        Returns:
            Dictionary with overall market direction and confidence
        """
        if not insights:
            return {"overall_direction": "neutral", "confidence": 0.5}

        # Count sentiments
        bullish_count = sum(
            1 for insight in insights if insight.get("sentiment") == "bullish"
        )
        bearish_count = sum(
            1 for insight in insights if insight.get("sentiment") == "bearish"
        )
        neutral_count = sum(
            1 for insight in insights if insight.get("sentiment") == "neutral"
        )

        total_count = bullish_count + bearish_count + neutral_count

        if total_count == 0:
            return {"overall_direction": "neutral", "confidence": 0.5}

        # Determine direction
        if bullish_count > bearish_count:
            direction = "bullish"
            raw_confidence = (bullish_count - bearish_count) / total_count
        elif bearish_count > bullish_count:
            direction = "bearish"
            raw_confidence = (bearish_count - bullish_count) / total_count
        else:
            direction = "neutral"
            raw_confidence = neutral_count / total_count

        # Adjust confidence based on strength of indicators
        rsi_insights = [i for i in insights if i.get("indicator") == "RSI"]
        macd_insights = [i for i in insights if i.get("indicator") == "MACD"]

        # Get the RSI and MACD values (if available)
        rsi_value = rsi_insights[0].get("value") if rsi_insights else None
        macd_value = macd_insights[0].get("value") if macd_insights else None

        # Adjust confidence based on RSI extremes
        confidence_modifier = 0
        if rsi_value is not None:
            if rsi_value > 80 or rsi_value < 20:
                confidence_modifier += 0.15  # Strong signal
            elif rsi_value > 70 or rsi_value < 30:
                confidence_modifier += 0.1  # Moderate signal

        # Adjust confidence based on MACD strength
        if macd_value is not None:
            if abs(macd_value) > 0.001:  # Significant MACD movement
                confidence_modifier += 0.1

        # Apply modifications and ensure confidence is between 0.1 and 0.9
        confidence = min(0.9, max(0.1, raw_confidence + confidence_modifier))

        return {"overall_direction": direction, "confidence": confidence}

    def _generate_signals(
        self, instrument: str, analysis_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate trading signals based on analysis results.

        Args:
            instrument: Trading pair
            analysis_results: Results from market analysis

        Returns:
            List of signal dictionaries
        """
        signals = []

        # Extract market view
        market_view = analysis_results.get("market_view", {})
        direction = market_view.get("overall_direction", "neutral")
        confidence = market_view.get("confidence", 0)

        # Get current price
        current_price = None
        for tf_data in analysis_results.get("insights", []):
            if "current_price" in tf_data:
                current_price = tf_data["current_price"]
                break

        if current_price is None:
            logger.warning("No current price available for signal generation")
            return signals

        # Only generate signals if confidence is above threshold
        if confidence < self.confidence_threshold:
            logger.info(
                f"Confidence {confidence} below threshold {self.confidence_threshold}, no signal generated"
            )
            return signals

        # Determine signal type based on direction
        signal_type = None
        signal_direction = None

        if direction == "bullish":
            signal_type = "ENTRY"
            signal_direction = "BUY"
        elif direction == "bearish":
            signal_type = "ENTRY"
            signal_direction = "SELL"
        else:
            return signals  # No signal for neutral direction

        # Extract support and resistance levels
        sr_data = {}
        for tf, sr in analysis_results.get("support_resistance", {}).items():
            if isinstance(sr, dict):
                support_levels = sr.get("support_levels", [])
                resistance_levels = sr.get("resistance_levels", [])

                # Ensure they're lists
                if not isinstance(support_levels, list):
                    support_levels = (
                        [support_levels] if support_levels is not None else []
                    )

                if not isinstance(resistance_levels, list):
                    resistance_levels = (
                        [resistance_levels] if resistance_levels is not None else []
                    )

                # Filter out None values
                support_levels = [
                    level for level in support_levels if level is not None
                ]
                resistance_levels = [
                    level for level in resistance_levels if level is not None
                ]

                sr_data[tf] = {
                    "support_levels": support_levels,
                    "resistance_levels": resistance_levels,
                }

        # Find stop loss and take profit levels based on support/resistance
        stop_loss = None
        take_profit = None

        # Use H1 timeframe for stop loss and take profit if available
        h1_sr = sr_data.get(
            "H1", sr_data.get(list(sr_data.keys())[0] if sr_data else {})
        )

        if signal_direction == "BUY":
            # For buy signals, stop loss below support, take profit at resistance
            supports = h1_sr.get("support_levels", [])
            resistances = h1_sr.get("resistance_levels", [])

            if supports:
                # Stop loss at closest support below current price
                valid_supports = [s for s in supports if s < current_price]
                if valid_supports:
                    stop_loss = max(valid_supports) * 0.999  # Slightly below support

            if resistances:
                # Take profit at closest resistance above current price
                valid_resistances = [r for r in resistances if r > current_price]
                if valid_resistances:
                    take_profit = (
                        min(valid_resistances) * 1.001
                    )  # Slightly above resistance

        elif signal_direction == "SELL":
            # For sell signals, stop loss above resistance, take profit at support
            supports = h1_sr.get("support_levels", [])
            resistances = h1_sr.get("resistance_levels", [])

            if resistances:
                # Stop loss at closest resistance above current price
                valid_resistances = [r for r in resistances if r > current_price]
                if valid_resistances:
                    stop_loss = (
                        min(valid_resistances) * 1.001
                    )  # Slightly above resistance

            if supports:
                # Take profit at closest support below current price
                valid_supports = [s for s in supports if s < current_price]
                if valid_supports:
                    take_profit = max(valid_supports) * 0.999  # Slightly below support

        # If no S/R levels found, use percentage-based levels
        if stop_loss is None:
            if signal_direction == "BUY":
                stop_loss = current_price * 0.99  # 1% below entry
            else:
                stop_loss = current_price * 1.01  # 1% above entry

        if take_profit is None:
            if signal_direction == "BUY":
                take_profit = current_price * 1.02  # 2% above entry
            else:
                take_profit = current_price * 0.98  # 2% below entry

        # Generate signal
        signal = {
            "instrument": instrument,
            "type": signal_type,
            "direction": signal_direction,
            "price": current_price,
            "confidence": confidence,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "timestamp": datetime.datetime.now().isoformat(),
            "timeframe": list(sr_data.keys()),
        }

        signals.append(signal)
        logger.info(
            f"Signal generated: {signal_direction} {instrument} at {current_price}"
        )

        return signals

    async def _update_context(self, instrument: str, results: Dict[str, Any]) -> None:
        """
        Update the context memory with the latest analysis results.

        Args:
            instrument: Trading pair analyzed
            results: Dictionary with analysis results
        """
        try:
            context_id = await self.memory_manager.store_analysis_result(
                results, "comprehensive"
            )
            logger.debug(f"Stored analysis for {instrument} with context ID: {context_id}")
        except Exception as e:
            logger.error(f"Failed to store analysis context for {instrument}: {e}")

    def _generate_mock_data(self, instrument: str, timeframe: str) -> Dict[str, Any]:
        """
        Generate mock market data for testing.

        Args:
            instrument: Trading pair
            timeframe: Timeframe

        Returns:
            Dictionary with mock market data
        """
        candles = []
        base_price = 1.1000  # Base price for EUR/USD

        # Generate 100 candles
        for i in range(100):
            price_change = (i % 10 - 5) * 0.0010  # Create some wave pattern
            close_price = base_price + price_change

            # Add some randomness
            import random

            random_factor = random.random() * 0.0020 - 0.0010

            high_price = close_price + abs(random_factor)
            low_price = close_price - abs(random_factor)
            open_price = close_price - random_factor

            candle = {
                "complete": True,
                "volume": 100 + i,
                "time": (
                    datetime.datetime.now() - datetime.timedelta(hours=100 - i)
                ).isoformat(),
                "mid": {
                    "o": str(round(open_price, 5)),
                    "h": str(round(high_price, 5)),
                    "l": str(round(low_price, 5)),
                    "c": str(round(close_price, 5)),
                },
            }

            candles.append(candle)

        return {"instrument": instrument, "granularity": timeframe, "candles": candles}

    async def analyze_candlestick_patterns(
        self,
        instrument: str,
        timeframe: Optional[str] = None,
        pattern_categories: Optional[List[str]] = None,
        confidence_threshold: float = 0.6,
    ) -> Dict[str, Any]:
        """
        Analyze candlestick patterns for the specified instrument.

        Args:
            instrument: Trading pair to analyze (e.g., "EUR_USD")
            timeframe: Optional timeframe to analyze (e.g., "H1", "H4")
                       If None, will analyze all timeframes in config
            pattern_categories: Categories of patterns to focus on
                               (e.g., ["reversal", "continuation"])
            confidence_threshold: Minimum confidence threshold for patterns

        Returns:
            Dict containing pattern analysis results
        """
        logger.info(
            f"Analyzing candlestick patterns for {instrument} on timeframe {timeframe}"
        )

        timeframes = (
            [timeframe] if timeframe else self.config.get("timeframes", ["H1", "H4"])
        )
        results = {
            "instrument": instrument,
            "timestamp": datetime.datetime.now().isoformat(),
            "patterns": [],
            "strongest_patterns": {},
            "market_bias": {},
            "timeframes_analyzed": timeframes,
        }

        # Import from App's implementation
        from AutoAgent.app_auto_agent.forex_ai.features.candlestick_patterns import (
            get_pattern_recognizer,
        )

        pattern_recognizer = get_pattern_recognizer()

        # Process each timeframe
        for tf in timeframes:
            # Fetch market data for the timeframe
            if self.data_fetcher:
                market_data = await self.data_fetcher.get_candles(
                    instrument=instrument,
                    timeframe=tf,
                    count=100,  # Get enough data for pattern detection
                )
            else:
                # Use mock data for testing if no data fetcher is provided
                market_data = self._generate_mock_data(instrument, tf)

            # Detect patterns
            tf_patterns = await pattern_recognizer.recognize_patterns(
                market_data,
                patterns=None,  # Use all patterns
                confidence_threshold=confidence_threshold,
            )

            # Filter by categories if specified
            if pattern_categories and tf_patterns.get("patterns"):
                tf_patterns["patterns"] = [
                    p
                    for p in tf_patterns["patterns"]
                    if p.get("category") in pattern_categories
                ]
                tf_patterns["detected_count"] = len(tf_patterns["patterns"])

            # Add timeframe to each pattern
            for pattern in tf_patterns.get("patterns", []):
                pattern["timeframe"] = tf

            # Add to results
            results["patterns"].extend(tf_patterns.get("patterns", []))

            # Track strongest pattern per timeframe
            if tf_patterns.get("strongest_pattern"):
                results["strongest_patterns"][tf] = tf_patterns["strongest_pattern"]

            # Track market bias per timeframe
            if tf_patterns.get("overall_bias"):
                results["market_bias"][tf] = tf_patterns["overall_bias"]

        # Generate overall market view
        if results["market_bias"]:
            # Aggregate bias across timeframes with higher weights for larger timeframes
            bullish_score = 0
            bearish_score = 0

            timeframe_weights = {
                "M1": 0.5,
                "M5": 0.6,
                "M15": 0.7,
                "M30": 0.8,
                "H1": 1.0,
                "H4": 1.5,
                "D1": 2.0,
                "W1": 3.0,
                "MN": 4.0,
            }

            for tf, bias in results["market_bias"].items():
                direction = bias.get("direction")
                strength = bias.get("strength", 0.5)
                weight = timeframe_weights.get(tf, 1.0)

                if direction == "bullish":
                    bullish_score += strength * weight
                else:
                    bearish_score += strength * weight

            # Determine overall bias
            if bullish_score > bearish_score:
                results["overall_bias"] = {
                    "direction": "bullish",
                    "strength": min(
                        1.0, bullish_score / (bullish_score + bearish_score + 0.01)
                    ),
                    "confidence": min(0.95, bullish_score / 5.0),  # Scale confidence
                }
            else:
                results["overall_bias"] = {
                    "direction": "bearish",
                    "strength": min(
                        1.0, bearish_score / (bullish_score + bearish_score + 0.01)
                    ),
                    "confidence": min(0.95, bearish_score / 5.0),  # Scale confidence
                }

        # Add pattern count
        results["pattern_count"] = len(results["patterns"])

        return results

    async def get_historical_analysis(
        self, instrument: str, from_time: datetime, to_time: datetime
    ) -> List[Dict[str, Any]]:
        """
        Retrieves historical analysis data for a given instrument and time range.

        Args:
            instrument: The trading instrument (e.g., 'EUR_USD').
            from_time: The start of the time range.
            to_time: The end of the time range.

        Returns:
            A list of historical analysis records.
        """
        if not self.memory_manager:
            logger.warning("Memory manager is not available.")
            return []
        
        context_id = f"analysis:{instrument}"
        return await self.memory_manager.get_historical_analyses(
            context_id, from_time, to_time
        )

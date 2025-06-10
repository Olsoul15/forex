"""
OANDA Data Handler for the Forex AI Trading System.

This module handles real-time and historical data fetching from OANDA,
including WebSocket connections for live price updates.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable

import pandas as pd
from oandapyV20 import API
from oandapyV20.exceptions import V20Error
from oandapyV20.endpoints.pricing import PricingStream
from oandapyV20.endpoints.instruments import InstrumentsCandles

logger = logging.getLogger(__name__)

class OandaDataHandler:
    """Handles data fetching and streaming from OANDA."""
    
    def __init__(self, access_token: str, account_id: str, practice: bool = True):
        """
        Initialize OANDA data handler.
        
        Args:
            access_token: OANDA API access token
            account_id: OANDA account ID
            practice: Whether to use practice account (default: True)
        """
        self.access_token = access_token
        self.account_id = account_id
        self.environment = "practice" if practice else "live"
        self.api = API(access_token=access_token, environment=self.environment)
        self.streams = {}  # Active price streams
        self.callbacks = {}  # Callback functions for data updates
        
    def fetch_historical_data(
        self,
        pair: str,
        timeframe: str = "M1",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        count: int = 5000
    ) -> pd.DataFrame:
        """
        Fetch historical candle data from OANDA.
        
        Args:
            pair: Currency pair (e.g., "EUR_USD")
            timeframe: Candle timeframe (e.g., "M1", "M5", "H1", "D")
            start: Start datetime (or None for count-based fetch)
            end: End datetime (or None for now)
            count: Number of candles to fetch (max 5000)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            params = {
                "granularity": timeframe,
                "price": "M"  # Midpoint candles
            }
            
            if start and end:
                params["from"] = start.strftime("%Y-%m-%dT%H:%M:%SZ")
                params["to"] = end.strftime("%Y-%m-%dT%H:%M:%SZ")
            else:
                params["count"] = min(count, 5000)
            
            request = InstrumentsCandles(instrument=pair, params=params)
            response = self.api.request(request)
            
            # Parse candle data
            candles = []
            for candle in response["candles"]:
                if candle["complete"]:
                    candles.append({
                        "timestamp": pd.to_datetime(candle["time"]),
                        "open": float(candle["mid"]["o"]),
                        "high": float(candle["mid"]["h"]),
                        "low": float(candle["mid"]["l"]),
                        "close": float(candle["mid"]["c"]),
                        "volume": int(candle["volume"])
                    })
            
            # Create DataFrame
            df = pd.DataFrame(candles)
            if not df.empty:
                df.set_index("timestamp", inplace=True)
                df.sort_index(inplace=True)
            
            return df
            
        except V20Error as e:
            logger.error(f"OANDA API error: {str(e)}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            return pd.DataFrame()
    
    def start_price_stream(
        self,
        pairs: List[str],
        on_data: Callable[[Dict], None],
        on_error: Optional[Callable[[Exception], None]] = None
    ) -> bool:
        """
        Start streaming price data for specified pairs.
        
        Args:
            pairs: List of currency pairs to stream
            on_data: Callback function for price updates
            on_error: Optional callback function for errors
            
        Returns:
            bool: Whether stream was started successfully
        """
        try:
            # Format pairs for OANDA (EUR/USD -> EUR_USD)
            formatted_pairs = [p.replace("/", "_") for p in pairs]
            
            # Create pricing stream
            stream = PricingStream(
                accountID=self.account_id,
                params={"instruments": ",".join(formatted_pairs)}
            )
            
            # Store callbacks
            stream_id = ",".join(formatted_pairs)
            self.callbacks[stream_id] = {
                "on_data": on_data,
                "on_error": on_error
            }
            
            # Start stream in a separate thread
            import threading
            thread = threading.Thread(
                target=self._run_price_stream,
                args=(stream, stream_id),
                daemon=True
            )
            thread.start()
            
            # Store stream reference
            self.streams[stream_id] = {
                "stream": stream,
                "thread": thread,
                "active": True
            }
            
            logger.info(f"Started price stream for {formatted_pairs}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting price stream: {str(e)}")
            if on_error:
                on_error(e)
            return False
    
    def _run_price_stream(self, stream: PricingStream, stream_id: str):
        """Run price stream and handle incoming data."""
        try:
            for msg in self.api.request(stream):
                if not self.streams[stream_id]["active"]:
                    break
                    
                if msg["type"] == "PRICE":
                    # Format price data
                    data = {
                        "type": "price",
                        "pair": msg["instrument"],
                        "timestamp": pd.to_datetime(msg["time"]),
                        "bid": float(msg["bids"][0]["price"]),
                        "ask": float(msg["asks"][0]["price"]),
                        "mid": (float(msg["bids"][0]["price"]) + float(msg["asks"][0]["price"])) / 2
                    }
                    
                    # Call data callback
                    if stream_id in self.callbacks:
                        self.callbacks[stream_id]["on_data"](data)
                        
        except Exception as e:
            logger.error(f"Error in price stream: {str(e)}")
            if stream_id in self.callbacks and self.callbacks[stream_id]["on_error"]:
                self.callbacks[stream_id]["on_error"](e)
    
    def stop_price_stream(self, pairs: List[str]):
        """Stop price stream for specified pairs."""
        stream_id = ",".join([p.replace("/", "_") for p in pairs])
        if stream_id in self.streams:
            self.streams[stream_id]["active"] = False
            del self.streams[stream_id]
            del self.callbacks[stream_id]
            logger.info(f"Stopped price stream for {pairs}")
    
    def stop_all_streams(self):
        """Stop all active price streams."""
        for stream_id in list(self.streams.keys()):
            self.streams[stream_id]["active"] = False
        self.streams.clear()
        self.callbacks.clear()
        logger.info("Stopped all price streams") 
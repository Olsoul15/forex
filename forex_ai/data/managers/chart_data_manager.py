"""
Chart Data Manager for the Forex AI Trading System.

This module manages the data flow between data sources and the chart system,
handling data buffering, decimation, and updates.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from forex_ai.connectors.oanda_handler import OandaDataHandler
from forex_ai.ui.dashboard.components.trading_view import TradingView

logger = logging.getLogger(__name__)

class ChartDataManager:
    """Manages data flow between data sources and charts."""
    
    def __init__(
        self,
        trading_view: TradingView,
        oanda_token: str,
        oanda_account: str,
        practice: bool = True,
        buffer_size: int = 1000,
        decimation_target: int = 500,
        decimation_method: str = "lttb"
    ):
        """
        Initialize chart data manager.
        
        Args:
            trading_view: TradingView instance for chart updates
            oanda_token: OANDA API access token
            oanda_account: OANDA account ID
            practice: Whether to use practice account
            buffer_size: Maximum size of data buffer
            decimation_target: Target number of points after decimation
            decimation_method: Decimation method (lttb, minmax, simple)
        """
        self.trading_view = trading_view
        self.oanda = OandaDataHandler(oanda_token, oanda_account, practice)
        
        self.buffer_size = buffer_size
        self.decimation_target = decimation_target
        self.decimation_method = decimation_method
        
        self.data_buffers = {}  # Pair -> DataFrame
        self.charts = {}  # Pair -> Chart reference
        self.active_timeframes = {}  # Pair -> List[timeframe]
        self.active_indicators = {}  # Pair -> Dict[indicator_id, params]
    
    def initialize_chart(
        self,
        pair: str,
        timeframe: str = "M1",
        chart_type: str = "candlestick",
        lookback_days: int = 30
    ) -> bool:
        """Initialize a new chart without indicators."""
        try:
            # Format pair for OANDA
            oanda_pair = pair.replace("/", "_")
            
            # Calculate date range
            end = datetime.utcnow()
            start = end - timedelta(days=lookback_days)
            
            # Fetch historical data
            df = self.oanda.fetch_historical_data(
                pair=oanda_pair,
                timeframe=timeframe,
                start=start,
                end=end
            )
            
            if df.empty:
                logger.error(f"No historical data received for {pair}")
                return False
            
            # Store in buffer
            self.data_buffers[pair] = df
            
            # Create chart without indicators
            chart = self.trading_view.create_chart(
                data=df,
                chart_type=chart_type,
                pair_name=pair
            )
            
            # Store chart reference
            self.charts[pair] = chart
            
            # Initialize indicator tracking
            self.active_indicators[pair] = {}
            
            # Track active timeframe
            if pair not in self.active_timeframes:
                self.active_timeframes[pair] = []
            self.active_timeframes[pair].append(timeframe)
            
            # Start price stream
            self._start_streaming(pair)
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing chart for {pair}: {str(e)}")
            return False
    
    def _start_streaming(self, pair: str):
        """Start price streaming for a pair."""
        oanda_pair = pair.replace("/", "_")
        
        def on_price_update(data):
            try:
                # Update data buffer
                timestamp = data["timestamp"]
                new_data = pd.DataFrame({
                    "open": [data["mid"]],
                    "high": [data["mid"]],
                    "low": [data["mid"]],
                    "close": [data["mid"]],
                    "volume": [0]
                }, index=[timestamp])
                
                self.data_buffers[pair] = pd.concat([
                    self.data_buffers[pair],
                    new_data
                ]).tail(self.buffer_size)
                
                # Decimate if needed
                if len(self.data_buffers[pair]) > self.buffer_size:
                    self.data_buffers[pair] = self._decimate_data(
                        self.data_buffers[pair],
                        self.decimation_target
                    )
                
                # Update chart
                if pair in self.charts:
                    self.trading_view.update_chart(
                        self.charts[pair],
                        self.data_buffers[pair],
                        use_webgl=True
                    )
                    
            except Exception as e:
                logger.error(f"Error handling price update for {pair}: {str(e)}")
        
        def on_error(error):
            logger.error(f"Stream error for {pair}: {str(error)}")
            # Attempt to restart stream after delay
            import time
            time.sleep(5)
            self._start_streaming(pair)
        
        self.oanda.start_price_stream([oanda_pair], on_price_update, on_error)
    
    def _decimate_data(
        self,
        data: pd.DataFrame,
        target_points: int,
        method: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Decimate data to reduce points while preserving visual appearance.
        
        Args:
            data: DataFrame to decimate
            target_points: Target number of points
            method: Decimation method (or None for default)
            
        Returns:
            Decimated DataFrame
        """
        method = method or self.decimation_method
        
        if len(data) <= target_points:
            return data
        
        if method == "lttb":
            # Largest Triangle Three Buckets
            return self._apply_lttb(data, target_points)
            
        elif method == "minmax":
            # Preserve min/max values
            bucket_size = len(data) // target_points
            result = []
            
            for i in range(0, len(data), bucket_size):
                bucket = data.iloc[i:i + bucket_size]
                if len(bucket) > 0:
                    result.extend([
                        bucket.iloc[0],  # First point
                        bucket.loc[bucket["high"].idxmax()],  # Highest point
                        bucket.loc[bucket["low"].idxmin()],  # Lowest point
                        bucket.iloc[-1]  # Last point
                    ])
            
            return pd.DataFrame(result).drop_duplicates()
            
        else:
            # Simple downsampling
            return data.iloc[::len(data)//target_points]
    
    def _apply_lttb(self, data: pd.DataFrame, target_points: int) -> pd.DataFrame:
        """Apply Largest Triangle Three Buckets algorithm."""
        if len(data) <= target_points:
            return data
            
        # Convert to numpy for faster computation
        times = data.index.astype(np.int64)
        values = data["close"].values
        
        # Initialize result with first point
        result_indices = [0]
        
        # Process all but the last bucket
        bucket_size = (len(data) - 2) / (target_points - 2)
        
        a = 0  # Initial point
        
        for i in range(target_points - 2):
            # Calculate next a
            next_a = int((i + 1) * bucket_size) + 1
            
            # Calculate area of triangles formed by points
            areas = []
            for j in range(int(a + 1), next_a):
                area = abs(
                    (times[a] - times[next_a]) * (values[j] - values[a]) -
                    (times[a] - times[j]) * (values[next_a] - values[a])
                ) * 0.5
                areas.append((j, area))
            
            # Select point that creates largest triangle
            max_area = -1
            max_idx = a + 1
            
            for idx, area in areas:
                if area > max_area:
                    max_area = area
                    max_idx = idx
            
            result_indices.append(max_idx)
            a = next_a
        
        # Add last point
        result_indices.append(len(data) - 1)
        
        return data.iloc[result_indices]
    
    def add_indicator(
        self,
        pair: str,
        indicator_type: str,
        params: Dict,
        indicator_id: Optional[str] = None
    ) -> str:
        """
        Add a technical indicator to a chart.
        
        Args:
            pair: Currency pair
            indicator_type: Type of indicator
            params: Indicator parameters
            indicator_id: Optional unique ID for the indicator
            
        Returns:
            Indicator ID if successful, None if failed
        """
        if pair not in self.charts:
            logger.error(f"No chart found for {pair}")
            return None
            
        try:
            # Generate unique ID if not provided
            if indicator_id is None:
                indicator_id = f"{indicator_type}_{len(self.active_indicators[pair])}"
            
            # Add indicator to chart
            success = self.trading_view.add_technical_indicator(
                self.charts[pair],
                self.data_buffers[pair],
                indicator_type,
                params,
                indicator_id=indicator_id
            )
            
            if success:
                # Track active indicator
                self.active_indicators[pair][indicator_id] = {
                    "type": indicator_type,
                    "params": params
                }
                return indicator_id
            
            return None
            
        except Exception as e:
            logger.error(f"Error adding indicator to {pair}: {str(e)}")
            return None
    
    def remove_indicator(self, pair: str, indicator_id: str) -> bool:
        """
        Remove a technical indicator from a chart.
        
        Args:
            pair: Currency pair
            indicator_id: Unique ID of the indicator to remove
            
        Returns:
            bool: Whether removal was successful
        """
        if pair not in self.charts:
            logger.error(f"No chart found for {pair}")
            return False
            
        if indicator_id not in self.active_indicators[pair]:
            logger.error(f"No indicator found with ID {indicator_id}")
            return False
            
        try:
            # Remove indicator from chart
            success = self.trading_view.remove_technical_indicator(
                self.charts[pair],
                indicator_id
            )
            
            if success:
                # Remove from tracking
                del self.active_indicators[pair][indicator_id]
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error removing indicator from {pair}: {str(e)}")
            return False
    
    def update_indicator(
        self,
        pair: str,
        indicator_id: str,
        new_params: Dict
    ) -> bool:
        """
        Update an existing indicator's parameters.
        
        Args:
            pair: Currency pair
            indicator_id: Unique ID of the indicator to update
            new_params: New parameters for the indicator
            
        Returns:
            bool: Whether update was successful
        """
        if pair not in self.active_indicators:
            logger.error(f"No indicators found for {pair}")
            return False
            
        if indicator_id not in self.active_indicators[pair]:
            logger.error(f"No indicator found with ID {indicator_id}")
            return False
            
        try:
            indicator = self.active_indicators[pair][indicator_id]
            
            # Remove existing indicator
            self.remove_indicator(pair, indicator_id)
            
            # Add new indicator with updated params
            success = self.add_indicator(
                pair,
                indicator["type"],
                new_params,
                indicator_id
            )
            
            return success is not None
            
        except Exception as e:
            logger.error(f"Error updating indicator: {str(e)}")
            return False
    
    def get_active_indicators(self, pair: str) -> Dict:
        """Get all active indicators for a pair."""
        return self.active_indicators.get(pair, {})
    
    def cleanup(self):
        """Clean up resources and stop data streams."""
        self.oanda.stop_all_streams()
        self.data_buffers.clear()
        self.charts.clear()
        self.active_timeframes.clear()
        self.active_indicators.clear() 
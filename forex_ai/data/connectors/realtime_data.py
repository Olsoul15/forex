"""
Real-time data connector for the Forex AI Trading System.

This module provides functionality to fetch real-time forex data and update charts.
"""

import logging
import json
import threading
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import plotly.graph_objects as go

from forex_ai.config.settings import get_settings
from forex_ai.exceptions import DataConnectorError
from forex_ai.data.connectors.base import DataConnector
from forex_ai.custom_types import MarketDataPoint, CurrencyPair, TimeFrame

logger = logging.getLogger(__name__)

class RealtimeDataConnector:
    """
    Connector for real-time data updates and WebSocket connections.
    
    This connector provides methods to fetch real-time forex market data
    and update charts with live data streams.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the real-time data connector.
        
        Args:
            api_key: API key for data providers
        """
        self.api_key = api_key
        self.active_connections = {}  # Store active WebSocket connections
        self.settings = get_settings()
        
    def initialize_websocket(
        self,
        currency_pairs: List[str],
        api_key: Optional[str] = None,
        provider: str = "default",
        on_message_callback: Optional[Callable] = None,
        on_error_callback: Optional[Callable] = None,
        on_close_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Initialize a WebSocket connection for real-time forex data.
        
        Args:
            currency_pairs: List of currency pairs to subscribe to
            api_key: API key for the data provider (if not using the default)
            provider: Data provider to use ("default", "oanda", "fxcm", etc.)
            on_message_callback: Callback function for received messages
            on_error_callback: Callback function for errors
            on_close_callback: Callback function for connection close
            
        Returns:
            Connection info dictionary with WebSocket instance and metadata
        """
        try:
            import websocket
        except ImportError:
            logger.error("websocket-client package not installed. Please install it with 'pip install websocket-client'")
            return {"status": "error", "message": "Missing websocket-client package"}
        
        # Use provided API key or fall back to default
        api_key = api_key or self.api_key or self.settings.FOREX_API_KEY
        
        # Define provider-specific connection details
        providers = {
            "default": {
                "url": "wss://stream.example.com/forex",  # Replace with actual provider URL
                "auth_required": True,
                "subscription_message": lambda pairs: json.dumps({"action": "subscribe", "pairs": pairs})
            },
            "oanda": {
                "url": "wss://stream-fxtrade.oanda.com/v3/",
                "auth_required": True,
                "subscription_message": lambda pairs: json.dumps({"type": "price", "instruments": ",".join(pairs)})
            },
            "fxcm": {
                "url": "wss://api.fxcm.com/socket.io/?transport=websocket",
                "auth_required": True,
                "subscription_message": lambda pairs: json.dumps({"method": "subscribe", "params": {"symbols": pairs}})
            }
        }
        
        # Get provider configuration
        if provider not in providers:
            logger.warning(f"Unknown provider: {provider}, using default")
            provider = "default"
        
        provider_config = providers[provider]
        ws_url = provider_config["url"]
        
        # Add authentication if required
        if provider_config["auth_required"] and api_key:
            if "?" in ws_url:
                ws_url += f"&api_key={api_key}"
            else:
                ws_url += f"?api_key={api_key}"
        
        # Define callback handlers
        def on_message(ws, message):
            logger.debug(f"WebSocket message received: {message[:100]}...")
            if on_message_callback:
                on_message_callback(message)
        
        def on_error(ws, error):
            logger.error(f"WebSocket error: {error}")
            if on_error_callback:
                on_error_callback(error)
        
        def on_close(ws, close_status_code, close_msg):
            logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")
            if on_close_callback:
                on_close_callback(close_status_code, close_msg)
        
        def on_open(ws):
            logger.info(f"WebSocket connection established with {provider}")
            # Subscribe to currency pairs
            subscription_msg = provider_config["subscription_message"](currency_pairs)
            logger.debug(f"Sending subscription: {subscription_msg}")
            ws.send(subscription_msg)
        
        # Create WebSocket connection
        try:
            ws = websocket.WebSocketApp(
                ws_url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open
            )
            
            # Create connection info
            connection_id = f"{provider}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            connection_info = {
                "id": connection_id,
                "status": "initializing",
                "provider": provider,
                "url": ws_url,
                "currency_pairs": currency_pairs,
                "websocket": ws,
                "created_at": datetime.now().isoformat()
            }
            
            # Store in active connections
            self.active_connections[connection_id] = connection_info
            
            return connection_info
            
        except Exception as e:
            logger.error(f"WebSocket initialization error: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def start_websocket(
        self,
        connection_info: Dict[str, Any],
        run_in_thread: bool = True
    ) -> Dict[str, Any]:
        """
        Start a WebSocket connection for real-time data.
        
        Args:
            connection_info: Connection info from initialize_websocket
            run_in_thread: Whether to run in a separate thread
            
        Returns:
            Updated connection info dictionary
        """
        if "websocket" not in connection_info:
            logger.error("Invalid connection info - missing websocket")
            return {"status": "error", "message": "Invalid connection info"}
        
        ws = connection_info["websocket"]
        
        try:
            if run_in_thread:
                wst = threading.Thread(target=ws.run_forever)
                wst.daemon = True  # Daemon thread will terminate when main thread exits
                wst.start()
                
                # Update connection info
                connection_info["status"] = "running"
                connection_info["thread"] = wst
                connection_info["start_time"] = datetime.now().isoformat()
                
                logger.info(f"WebSocket started in thread for {connection_info.get('provider', 'default')}")
            else:
                # Run in the main thread (will block until closed)
                logger.info("Starting WebSocket in main thread (blocking)")
                connection_info["status"] = "running"
                connection_info["start_time"] = datetime.now().isoformat()
                ws.run_forever()
            
            # Update stored connection info
            if "id" in connection_info and connection_info["id"] in self.active_connections:
                self.active_connections[connection_info["id"]] = connection_info
            
            return connection_info
            
        except Exception as e:
            logger.error(f"WebSocket start error: {str(e)}")
            connection_info["status"] = "error"
            connection_info["error"] = str(e)
            return connection_info
    
    def stop_websocket(
        self,
        connection_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Stop a WebSocket connection.
        
        Args:
            connection_info: Connection info dictionary
            
        Returns:
            Updated connection info dictionary
        """
        if "websocket" not in connection_info:
            logger.error("Invalid connection info - missing websocket")
            return {"status": "error", "message": "Invalid connection info"}
        
        ws = connection_info["websocket"]
        
        try:
            ws.close()
            connection_info["status"] = "closed"
            connection_info["stop_time"] = datetime.now().isoformat()
            logger.info("WebSocket connection closed")
            
            # Update stored connection info
            if "id" in connection_info and connection_info["id"] in self.active_connections:
                self.active_connections[connection_info["id"]] = connection_info
            
            return connection_info
            
        except Exception as e:
            logger.error(f"WebSocket close error: {str(e)}")
            connection_info["status"] = "error"
            connection_info["error"] = str(e)
            return connection_info
    
    def update_chart_with_realtime_data(
        self,
        chart: go.Figure,
        new_data: pd.DataFrame,
        max_points: int = None,
        update_indicators: bool = True
    ) -> go.Figure:
        """
        Update a chart with new real-time data.
        
        Args:
            chart: Plotly Figure object to update
            new_data: New OHLCV data to add to the chart
            max_points: Maximum number of data points to keep (None for unlimited)
            update_indicators: Whether to update technical indicators
            
        Returns:
            Updated Plotly Figure
        """
        if chart is None or new_data.empty:
            logger.warning("Cannot update chart: empty chart or data")
            return chart
        
        # Find the main price trace (candlestick, ohlc, or scatter)
        main_trace_idx = None
        for i, trace in enumerate(chart.data):
            if hasattr(trace, "type") and trace.type in ["candlestick", "ohlc"]:
                main_trace_idx = i
                break
        
        # If no OHLC/candlestick trace found, look for a scatter trace named "Price"
        if main_trace_idx is None:
            for i, trace in enumerate(chart.data):
                if hasattr(trace, "name") and "price" in trace.name.lower():
                    main_trace_idx = i
                    break
        
        # If still no trace found, use the first trace
        if main_trace_idx is None and len(chart.data) > 0:
            main_trace_idx = 0
        
        # If no trace found, return unchanged
        if main_trace_idx is None:
            logger.warning("Cannot update chart: no suitable trace found")
            return chart
        
        # Get the current trace data
        current_trace = chart.data[main_trace_idx]
        
        # Update the trace based on its type
        try:
            trace_type = current_trace.type if hasattr(current_trace, "type") else "scatter"
            
            if trace_type == "candlestick":
                # Get current data
                current_x = list(current_trace.x)
                current_open = list(current_trace.open)
                current_high = list(current_trace.high)
                current_low = list(current_trace.low)
                current_close = list(current_trace.close)
                
                # Add new data
                updated_x = current_x + list(new_data.index)
                updated_open = current_open + list(new_data["open"])
                updated_high = current_high + list(new_data["high"])
                updated_low = current_low + list(new_data["low"])
                updated_close = current_close + list(new_data["close"])
                
                # Limit number of points if specified
                if max_points and len(updated_x) > max_points:
                    updated_x = updated_x[-max_points:]
                    updated_open = updated_open[-max_points:]
                    updated_high = updated_high[-max_points:]
                    updated_low = updated_low[-max_points:]
                    updated_close = updated_close[-max_points:]
                
                # Update the trace
                chart.data[main_trace_idx].x = updated_x
                chart.data[main_trace_idx].open = updated_open
                chart.data[main_trace_idx].high = updated_high
                chart.data[main_trace_idx].low = updated_low
                chart.data[main_trace_idx].close = updated_close
                
            elif trace_type == "ohlc":
                # Similar to candlestick but with OHLC type
                current_x = list(current_trace.x)
                current_open = list(current_trace.open)
                current_high = list(current_trace.high)
                current_low = list(current_trace.low)
                current_close = list(current_trace.close)
                
                updated_x = current_x + list(new_data.index)
                updated_open = current_open + list(new_data["open"])
                updated_high = current_high + list(new_data["high"])
                updated_low = current_low + list(new_data["low"])
                updated_close = current_close + list(new_data["close"])
                
                if max_points and len(updated_x) > max_points:
                    updated_x = updated_x[-max_points:]
                    updated_open = updated_open[-max_points:]
                    updated_high = updated_high[-max_points:]
                    updated_low = updated_low[-max_points:]
                    updated_close = updated_close[-max_points:]
                
                chart.data[main_trace_idx].x = updated_x
                chart.data[main_trace_idx].open = updated_open
                chart.data[main_trace_idx].high = updated_high
                chart.data[main_trace_idx].low = updated_low
                chart.data[main_trace_idx].close = updated_close
                
            else:  # scatter, line, etc.
                # Get current data
                current_x = list(current_trace.x)
                current_y = list(current_trace.y)
                
                # Add new data (assume y is close price if not specified)
                updated_x = current_x + list(new_data.index)
                if "close" in new_data.columns:
                    updated_y = current_y + list(new_data["close"])
                else:
                    updated_y = current_y + list(new_data.iloc[:, 0])  # Use first column if no close
                
                if max_points and len(updated_x) > max_points:
                    updated_x = updated_x[-max_points:]
                    updated_y = updated_y[-max_points:]
                
                chart.data[main_trace_idx].x = updated_x
                chart.data[main_trace_idx].y = updated_y
            
            # Update volume trace if it exists
            if "volume" in new_data.columns:
                for i, trace in enumerate(chart.data):
                    if hasattr(trace, "name") and trace.name and "volume" in trace.name.lower():
                        current_x = list(trace.x)
                        current_y = list(trace.y)
                        
                        updated_x = current_x + list(new_data.index)
                        updated_y = current_y + list(new_data["volume"])
                        
                        if max_points and len(updated_x) > max_points:
                            updated_x = updated_x[-max_points:]
                            updated_y = updated_y[-max_points:]
                        
                        chart.data[i].x = updated_x
                        chart.data[i].y = updated_y
                        
                        # Update colors for volume bars
                        if hasattr(trace, "marker") and hasattr(trace.marker, "color") and isinstance(trace.marker.color, list):
                            # Generate colors for new volume bars
                            colors = []
                            for j in range(len(new_data)):
                                if j > 0 or len(current_y) == 0:
                                    if j > 0 and new_data["close"].iloc[j] > new_data["close"].iloc[j-1]:
                                        colors.append("rgba(76, 175, 80, 0.5)")  # Green
                                    else:
                                        colors.append("rgba(244, 67, 54, 0.5)")  # Red
                                else:
                                    # Compare with the last candle in current data
                                    if new_data["close"].iloc[0] > current_close[-1]:
                                        colors.append("rgba(76, 175, 80, 0.5)")  # Green
                                    else:
                                        colors.append("rgba(244, 67, 54, 0.5)")  # Red
                            
                            updated_colors = list(trace.marker.color) + colors
                            if max_points and len(updated_colors) > max_points:
                                updated_colors = updated_colors[-max_points:]
                            
                            chart.data[i].marker.color = updated_colors
            
            # Update the x-axis range to show the latest data
            if len(updated_x) > 0:
                # Calculate a reasonable range - show the latest data
                visible_points = min(60, len(updated_x))  # Show last 60 points or all if fewer
                chart.update_layout(
                    xaxis_range=[updated_x[-visible_points], updated_x[-1]]
                )
            
            return chart
            
        except Exception as e:
            logger.error(f"Error updating chart with real-time data: {e}")
            return chart
    
    def create_realtime_alerts(
        self,
        chart: go.Figure,
        alert_conditions: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Set up real-time price alerts on a chart.
        
        Args:
            chart: Plotly Figure to add alerts to
            alert_conditions: List of alert condition dictionaries
            
        Returns:
            Alerts configuration dictionary
        """
        if chart is None:
            logger.error("Cannot set up alerts: chart is None")
            return {"status": "error", "message": "Invalid chart"}
        
        if not alert_conditions:
            # Create some default alert conditions
            alert_conditions = [
                {
                    "type": "price",
                    "field": "close",
                    "comparison": "above",
                    "value": None,  # Will be set later
                    "message": "Price moved above {value}"
                },
                {
                    "type": "price",
                    "field": "close",
                    "comparison": "below",
                    "value": None,  # Will be set later
                    "message": "Price moved below {value}"
                }
            ]
        
        # Try to set price-based alert values from current chart data
        for condition in alert_conditions:
            if condition["type"] == "price" and condition["value"] is None:
                # Find the main price trace
                for trace in chart.data:
                    if hasattr(trace, "type") and trace.type in ["candlestick", "ohlc"]:
                        if condition["field"] == "close" and hasattr(trace, "close") and len(trace.close) > 0:
                            last_close = trace.close[-1]
                            # Set alert slightly above or below current price
                            if condition["comparison"] == "above":
                                condition["value"] = round(last_close * 1.01, 4)  # 1% above
                            elif condition["comparison"] == "below":
                                condition["value"] = round(last_close * 0.99, 4)  # 1% below
                            break
                    elif hasattr(trace, "y") and len(trace.y) > 0:
                        last_value = trace.y[-1]
                        # Set alert slightly above or below current price
                        if condition["comparison"] == "above":
                            condition["value"] = round(last_value * 1.01, 4)  # 1% above
                        elif condition["comparison"] == "below":
                            condition["value"] = round(last_value * 0.99, 4)  # 1% below
                        break
        
        # Add alert indicators to the chart
        for i, condition in enumerate(alert_conditions):
            if condition["type"] == "price" and condition["value"] is not None:
                # Add a horizontal line for price alerts
                chart.add_shape(
                    type="line",
                    x0=0,
                    x1=1,
                    y0=condition["value"],
                    y1=condition["value"],
                    xref="paper",
                    line=dict(
                        color="rgba(255, 165, 0, 0.5)",
                        width=2,
                        dash="dash"
                    ),
                    name=f"Alert: {condition['comparison']} {condition['value']}"
                )
                
                # Add a label
                comparison_symbol = ">" if condition["comparison"] == "above" else "<"
                chart.add_annotation(
                    x=0.02,
                    y=condition["value"],
                    text=f"{condition['field'].capitalize()} {comparison_symbol} {condition['value']}",
                    showarrow=False,
                    xref="paper",
                    font=dict(size=10, color="orange")
                )
        
        # Create alerts configuration
        alerts_config = {
            "chart": chart,
            "conditions": alert_conditions,
            "triggered_alerts": [],
            "last_check": datetime.now(),
            "status": "configured"
        }
        
        return alerts_config
    
    def check_alerts(
        self,
        alerts_config: Dict[str, Any],
        current_data: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        Check if any alerts have been triggered by current data.
        
        Args:
            alerts_config: Alerts configuration dictionary
            current_data: Current OHLCV data
            
        Returns:
            List of triggered alerts
        """
        if not alerts_config or current_data.empty:
            return []
        
        conditions = alerts_config.get("conditions", [])
        triggered_alerts = []
        
        for condition in conditions:
            try:
                if condition["type"] == "price":
                    field = condition["field"]
                    value = condition["value"]
                    comparison = condition["comparison"]
                    
                    if field in current_data.columns and value is not None:
                        current_value = current_data[field].iloc[-1]
                        
                        # Check if the condition is triggered
                        if comparison == "above" and current_value > value:
                            triggered_alerts.append({
                                "type": "price",
                                "message": condition.get("message", "").format(value=value),
                                "timestamp": datetime.now().isoformat(),
                                "current_value": current_value,
                                "threshold": value
                            })
                        elif comparison == "below" and current_value < value:
                            triggered_alerts.append({
                                "type": "price",
                                "message": condition.get("message", "").format(value=value),
                                "timestamp": datetime.now().isoformat(),
                                "current_value": current_value,
                                "threshold": value
                            })
            
            except Exception as e:
                logger.error(f"Error checking alert condition: {e}")
        
        return triggered_alerts
    
    def setup_auto_refresh(
        self, 
        chart: go.Figure,
        refresh_interval: int = 5,  # seconds
        data_source_callback: Callable = None,
        max_points: int = 1000
    ) -> Dict[str, Any]:
        """
        Set up auto-refresh functionality for a chart.
        
        Args:
            chart: Plotly Figure object to refresh
            refresh_interval: Refresh interval in seconds
            data_source_callback: Callback function that returns new data when called
            max_points: Maximum number of data points to keep
            
        Returns:
            Configuration dictionary for auto-refresh
        """
        if chart is None:
            logger.error("Cannot set up auto-refresh: chart is None")
            return {"status": "error", "message": "Invalid chart"}
        
        if data_source_callback is None:
            logger.error("Cannot set up auto-refresh: no data source callback provided")
            return {"status": "error", "message": "No data source callback"}
        
        # Add auto-refresh indicator to the chart
        chart.add_annotation(
            x=1.0,
            y=1.05,
            xref="paper",
            yref="paper",
            text=f"Auto-refresh: {refresh_interval}s",
            showarrow=False,
            font=dict(size=10, color="#888888"),
            xanchor="right"
        )
        
        # Create refresh configuration
        refresh_config = {
            "chart": chart,
            "interval": refresh_interval,
            "data_callback": data_source_callback,
            "max_points": max_points,
            "last_refresh": datetime.now(),
            "status": "configured"
        }
        
        return refresh_config
    
    def parse_realtime_message(
        self,
        message: str,
        provider: str = "default"
    ) -> pd.DataFrame:
        """
        Parse a real-time WebSocket message into a DataFrame.
        
        Args:
            message: WebSocket message (usually JSON)
            provider: Provider type to determine parsing logic
            
        Returns:
            DataFrame with parsed data
        """
        try:
            # Parse JSON message
            data = json.loads(message)
            
            # Different providers have different message formats
            if provider == "oanda":
                # Example OANDA format
                if "type" in data and data["type"] == "price":
                    instrument = data.get("instrument", "")
                    time = pd.to_datetime(data.get("time", ""))
                    ask = float(data.get("asks", [{}])[0].get("price", 0))
                    bid = float(data.get("bids", [{}])[0].get("price", 0))
                    mid = (ask + bid) / 2
                    
                    # Create a single-row DataFrame
                    df = pd.DataFrame({
                        "open": [mid],
                        "high": [mid],
                        "low": [mid],
                        "close": [mid],
                        "bid": [bid],
                        "ask": [ask]
                    }, index=[time])
                    
                    return df
            
            elif provider == "fxcm":
                # Example FXCM format
                if "data" in data:
                    tick_data = data["data"]
                    symbol = tick_data.get("symbol", "")
                    time = pd.to_datetime(tick_data.get("updated", ""))
                    bid = float(tick_data.get("bid", 0))
                    ask = float(tick_data.get("ask", 0))
                    mid = (ask + bid) / 2
                    
                    df = pd.DataFrame({
                        "open": [mid],
                        "high": [mid],
                        "low": [mid],
                        "close": [mid],
                        "bid": [bid],
                        "ask": [ask]
                    }, index=[time])
                    
                    return df
            
            else:
                # Generic format (assumes a simple tick data format)
                if "symbol" in data and "price" in data:
                    symbol = data.get("symbol", "")
                    time = pd.to_datetime(data.get("timestamp", datetime.now()))
                    price = float(data.get("price", 0))
                    
                    df = pd.DataFrame({
                        "open": [price],
                        "high": [price],
                        "low": [price],
                        "close": [price]
                    }, index=[time])
                    
                    return df
            
            # If we couldn't parse, return empty DataFrame
            logger.warning(f"Could not parse message format for provider {provider}")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error parsing real-time message: {e}")
            return pd.DataFrame()
    
    def aggregate_tick_data(
        self,
        tick_data: pd.DataFrame,
        timeframe: str = "1m",
        update_current: bool = True
    ) -> pd.DataFrame:
        """
        Aggregate tick data into OHLC candles for a specified timeframe.
        
        Args:
            tick_data: DataFrame with tick data
            timeframe: Timeframe for aggregation (1m, 5m, 15m, 1h, etc.)
            update_current: Whether to update the current candle or create a new one
            
        Returns:
            DataFrame with aggregated OHLC data
        """
        if tick_data.empty:
            return pd.DataFrame()
        
        try:
            # Determine the resampling rule based on timeframe
            resample_rule = ""
            if timeframe.endswith("m"):
                resample_rule = f"{timeframe[:-1]}T"  # Convert 5m to 5T (pandas format)
            elif timeframe.endswith("h"):
                resample_rule = f"{timeframe[:-1]}H"
            elif timeframe.endswith("d"):
                resample_rule = f"{timeframe[:-1]}D"
            else:
                logger.warning(f"Unknown timeframe format: {timeframe}")
                return pd.DataFrame()
            
            # Resample tick data to OHLC
            ohlc_data = tick_data.resample(resample_rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last'
            })
            
            # If bid/ask are available, include them
            if 'bid' in tick_data.columns and 'ask' in tick_data.columns:
                bid_ask = tick_data.resample(resample_rule).agg({
                    'bid': 'last',
                    'ask': 'last'
                })
                ohlc_data = pd.concat([ohlc_data, bid_ask], axis=1)
            
            return ohlc_data.dropna()
            
        except Exception as e:
            logger.error(f"Error aggregating tick data: {e}")
            return pd.DataFrame() 
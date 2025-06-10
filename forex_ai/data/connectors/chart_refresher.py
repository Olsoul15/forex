"""
Chart refresher utility for automatically updating charts with real-time data.

This module provides a utility for automatically refreshing charts with real-time data
using a periodic update mechanism that can be used with web frameworks like Dash or Flask.
"""

import logging
import threading
import time
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go

from forex_ai.data.connectors.realtime_data import RealtimeDataConnector

logger = logging.getLogger(__name__)

class ChartRefresher:
    """
    Utility for automatically refreshing charts with real-time data.
    
    This class provides a way to set up automatic chart refreshing with a 
    configurable refresh interval and data source.
    """
    
    def __init__(self, realtime_connector: RealtimeDataConnector):
        """
        Initialize the chart refresher.
        
        Args:
            realtime_connector: RealtimeDataConnector instance for data processing
        """
        self.realtime_connector = realtime_connector
        self.active_charts = {}  # Dictionary of charts being refreshed
        self._stop_event = threading.Event()
        self._refresh_thread = None
    
    def start_refreshing(
        self,
        chart_id: str,
        chart: go.Figure,
        data_source: Callable,
        refresh_interval: int = 5,  # seconds
        max_points: int = 500,
        update_indicators: bool = True,
        on_update_callback: Optional[Callable] = None,
        alert_conditions: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Start automatically refreshing a chart.
        
        Args:
            chart_id: Unique identifier for the chart
            chart: Plotly Figure to refresh
            data_source: Callable that returns new data (pd.DataFrame)
            refresh_interval: Refresh interval in seconds
            max_points: Maximum number of data points to keep
            update_indicators: Whether to update technical indicators
            on_update_callback: Callback function called after each update
            alert_conditions: Optional alert conditions to check
            
        Returns:
            Chart configuration dictionary
        """
        if chart is None or data_source is None:
            logger.error("Invalid chart or data source provided")
            return {"status": "error", "message": "Invalid chart or data source"}
        
        # Set up refresh configuration
        refresh_config = {
            "chart_id": chart_id,
            "chart": chart,
            "data_source": data_source,
            "interval": refresh_interval,
            "max_points": max_points,
            "update_indicators": update_indicators,
            "last_refresh": datetime.now(),
            "on_update_callback": on_update_callback,
            "alert_conditions": alert_conditions,
            "triggered_alerts": [],
            "status": "running"
        }
        
        # Add refresh indicator to the chart
        if not any(a.text == f"Auto-refresh: {refresh_interval}s" for a in chart.layout.annotations):
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
        
        # Add to active charts
        self.active_charts[chart_id] = refresh_config
        
        # Start the refresh thread if not already running
        if self._refresh_thread is None or not self._refresh_thread.is_alive():
            self._stop_event.clear()
            self._refresh_thread = threading.Thread(target=self._refresh_loop)
            self._refresh_thread.daemon = True
            self._refresh_thread.start()
        
        return refresh_config
    
    def stop_refreshing(self, chart_id: str) -> Dict[str, Any]:
        """
        Stop refreshing a specific chart.
        
        Args:
            chart_id: Unique identifier of the chart to stop refreshing
            
        Returns:
            Status dictionary
        """
        if chart_id not in self.active_charts:
            return {"status": "error", "message": f"Chart {chart_id} not found"}
        
        # Update status
        self.active_charts[chart_id]["status"] = "stopped"
        
        # Get the configuration before removing
        config = self.active_charts[chart_id]
        
        # Remove from active charts
        del self.active_charts[chart_id]
        
        return {"status": "stopped", "chart_id": chart_id, "config": config}
    
    def stop_all(self) -> Dict[str, Any]:
        """
        Stop all chart refreshing.
        
        Returns:
            Status dictionary
        """
        # Signal the thread to stop
        self._stop_event.set()
        
        # Update status for all charts
        for chart_id in list(self.active_charts.keys()):
            self.active_charts[chart_id]["status"] = "stopped"
        
        # Clear the active charts
        stopped_charts = list(self.active_charts.keys())
        self.active_charts.clear()
        
        return {"status": "stopped", "charts": stopped_charts}
    
    def _refresh_loop(self):
        """Internal method that runs the refresh loop in a thread."""
        logger.info("Starting chart refresh loop")
        
        while not self._stop_event.is_set() and self.active_charts:
            current_time = datetime.now()
            
            # Process each active chart
            for chart_id, config in list(self.active_charts.items()):
                if config["status"] != "running":
                    continue
                
                # Check if it's time to refresh
                time_since_last_refresh = (current_time - config["last_refresh"]).total_seconds()
                if time_since_last_refresh >= config["interval"]:
                    try:
                        # Get new data
                        new_data = config["data_source"]()
                        
                        if new_data is not None and not new_data.empty:
                            # Update the chart
                            updated_chart = self.realtime_connector.update_chart_with_realtime_data(
                                chart=config["chart"],
                                new_data=new_data,
                                max_points=config["max_points"],
                                update_indicators=config["update_indicators"]
                            )
                            
                            # Check for alerts if configured
                            triggered_alerts = []
                            if config["alert_conditions"]:
                                alerts_config = {
                                    "chart": updated_chart,
                                    "conditions": config["alert_conditions"]
                                }
                                triggered_alerts = self.realtime_connector.check_alerts(alerts_config, new_data)
                                config["triggered_alerts"].extend(triggered_alerts)
                            
                            # Update configuration
                            config["chart"] = updated_chart
                            config["last_refresh"] = current_time
                            
                            # Call the update callback if provided
                            if config["on_update_callback"]:
                                config["on_update_callback"](updated_chart, triggered_alerts)
                            
                    except Exception as e:
                        logger.error(f"Error refreshing chart {chart_id}: {e}")
            
            # Sleep briefly to avoid high CPU usage
            time.sleep(0.2)
        
        logger.info("Chart refresh loop stopped")
    
    def get_chart(self, chart_id: str) -> Optional[go.Figure]:
        """
        Get the current state of a chart being refreshed.
        
        Args:
            chart_id: Unique identifier of the chart
            
        Returns:
            The current Plotly Figure or None if not found
        """
        if chart_id in self.active_charts:
            return self.active_charts[chart_id]["chart"]
        return None
    
    def get_triggered_alerts(self, chart_id: str, clear: bool = False) -> List[Dict[str, Any]]:
        """
        Get triggered alerts for a chart.
        
        Args:
            chart_id: Unique identifier of the chart
            clear: Whether to clear the alerts after retrieving
            
        Returns:
            List of triggered alerts
        """
        if chart_id not in self.active_charts:
            return []
        
        alerts = self.active_charts[chart_id]["triggered_alerts"]
        
        if clear:
            self.active_charts[chart_id]["triggered_alerts"] = []
        
        return alerts
    
    def update_refresh_interval(self, chart_id: str, interval: int) -> Dict[str, Any]:
        """
        Update the refresh interval for a chart.
        
        Args:
            chart_id: Unique identifier of the chart
            interval: New refresh interval in seconds
            
        Returns:
            Updated configuration dictionary
        """
        if chart_id not in self.active_charts:
            return {"status": "error", "message": f"Chart {chart_id} not found"}
        
        config = self.active_charts[chart_id]
        config["interval"] = interval
        
        # Update the refresh indicator on the chart
        for i, annotation in enumerate(config["chart"].layout.annotations):
            if "Auto-refresh" in annotation.text:
                config["chart"].layout.annotations[i].text = f"Auto-refresh: {interval}s"
                break
        
        return config 
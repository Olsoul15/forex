"""
Example demonstrating real-time chart updates with WebSocket data.

This script shows how to:
1. Create a TradingView chart with Plotly
2. Set up a WebSocket connection for real-time data
3. Update the chart with incoming data in real-time
4. Set up alerts and auto-refresh functionality
"""

import time
import logging
import json
import threading
import random
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from forex_ai.data.connectors.trading_view import TradingViewConnector
from forex_ai.data.connectors.realtime_data import RealtimeDataConnector

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_mock_data(pair="EURUSD", timeframe="1h", num_candles=100):
    """Generate mock OHLCV data for testing."""
    end_date = datetime.now()

    # Determine time interval based on timeframe
    if timeframe.endswith("m"):
        interval = timedelta(minutes=int(timeframe[:-1]))
    elif timeframe.endswith("h"):
        interval = timedelta(hours=int(timeframe[:-1]))
    else:
        interval = timedelta(hours=1)  # Default to 1h

    start_date = end_date - (num_candles * interval)

    # Create date range for the historical data
    dates = [start_date + (i * interval) for i in range(num_candles)]

    # Generate random price data
    base_price = 1.1000  # Starting price for EURUSD
    volatility = 0.0005  # Price movement per candle

    data = []
    current_price = base_price

    for i in range(num_candles):
        # Random walk with drift
        price_change = (random.random() - 0.5) * volatility
        if random.random() < 0.52:  # Slight upward bias
            price_change += 0.0001

        # Generate OHLC data
        open_price = current_price
        close_price = open_price + price_change
        high_price = max(open_price, close_price) + (random.random() * volatility * 0.5)
        low_price = min(open_price, close_price) - (random.random() * volatility * 0.5)
        volume = int(random.random() * 1000) + 500

        data.append(
            {
                "date": dates[i],
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume,
            }
        )

        current_price = close_price

    # Create DataFrame
    df = pd.DataFrame(data)
    df.set_index("date", inplace=True)

    return df


def mock_tick_updates(ws_instance, pair="EURUSD"):
    """Simulate tick data updates from a WebSocket connection."""
    base_price = 1.1200  # Current price for EURUSD
    volatility = 0.0002  # Tick-by-tick volatility

    # Dummy message handler function
    def on_message_callback(message):
        logger.info(f"Received tick: {message}")

    # Add mock data to the connection info
    ws_instance["on_message_callback"] = on_message_callback

    # Function to simulate ticks
    def generate_ticks():
        current_price = base_price
        while ws_instance["status"] == "running":
            # Small random price movement
            price_change = (random.random() - 0.5) * volatility

            # Simulate bid/ask spread
            spread = 0.00010  # 1 pip for EURUSD
            bid_price = current_price - (spread / 2)
            ask_price = current_price + (spread / 2)

            # Create tick message
            tick = {
                "type": "price",
                "instrument": pair,
                "time": datetime.now().isoformat(),
                "bids": [{"price": round(bid_price, 5), "liquidity": 1000000}],
                "asks": [{"price": round(ask_price, 5), "liquidity": 1000000}],
            }

            # Send the tick to the callback
            on_message_callback(json.dumps(tick))

            # Update current price
            current_price += price_change

            # Wait before next tick (simulate 2-5 ticks per second)
            time.sleep(random.uniform(0.2, 0.5))

    # Start generating ticks in a thread
    tick_thread = threading.Thread(target=generate_ticks)
    tick_thread.daemon = True
    tick_thread.start()

    return ws_instance


def main():
    """Main example function demonstrating real-time charts."""
    # Initialize connectors
    tv_connector = TradingViewConnector()
    realtime_connector = RealtimeDataConnector(api_key="mock_api_key")

    # Create initial historical data
    historical_data = generate_mock_data(pair="EURUSD", timeframe="1m", num_candles=120)

    # Create the initial chart
    chart = tv_connector.create_chart(
        data=historical_data,
        chart_type="candlestick",
        title="EURUSD Live Chart (1m)",
        pair_name="EUR/USD",
        height=600,
        width=1000,
        show_volume=True,
        dark_mode=True,
    )

    # Add some technical indicators
    chart = tv_connector.add_technical_indicator(
        chart=chart,
        data=historical_data,
        indicator_type="sma",
        params={"period": 20},
        color="#1E88E5",  # Blue
    )

    chart = tv_connector.add_technical_indicator(
        chart=chart,
        data=historical_data,
        indicator_type="sma",
        params={"period": 50},
        color="#FF5722",  # Orange
    )

    chart = tv_connector.add_technical_indicator(
        chart=chart,
        data=historical_data,
        indicator_type="rsi",
        params={"period": 14},
        color="#7CB342",  # Green
    )

    # 1. Set up real-time alerts
    alert_conditions = [
        {
            "type": "price",
            "field": "close",
            "comparison": "above",
            "value": historical_data["close"].iloc[-1] * 1.001,  # 0.1% above last price
            "message": "Price moved above {value}",
        },
        {
            "type": "price",
            "field": "close",
            "comparison": "below",
            "value": historical_data["close"].iloc[-1] * 0.999,  # 0.1% below last price
            "message": "Price moved below {value}",
        },
    ]

    alerts_config = realtime_connector.create_realtime_alerts(chart, alert_conditions)

    # 2. Initialize a WebSocket connection (using mock data for example)
    connection_info = realtime_connector.initialize_websocket(
        currency_pairs=["EURUSD"],
        provider="oanda",  # This will use mock data in our example
    )

    # 3. Start the WebSocket with mock data
    connection_info["status"] = "running"
    connection_info = mock_tick_updates(connection_info)

    # Function to process incoming data
    tick_buffer = []
    last_update_time = datetime.now()

    def process_realtime_data():
        nonlocal tick_buffer, last_update_time

        # Check if we have any new data to process
        if not tick_buffer:
            return chart, []

        # Process accumulated ticks
        current_time = datetime.now()
        if (
            current_time - last_update_time
        ).total_seconds() >= 1:  # Update chart every second
            # Parse the tick data
            tick_dfs = []
            for tick_msg in tick_buffer:
                try:
                    data = json.loads(tick_msg)
                    # Parse into DataFrame
                    if "type" in data and data["type"] == "price":
                        time = pd.to_datetime(
                            data.get("time", datetime.now().isoformat())
                        )
                        ask = float(data.get("asks", [{}])[0].get("price", 0))
                        bid = float(data.get("bids", [{}])[0].get("price", 0))
                        mid = (ask + bid) / 2

                        df = pd.DataFrame(
                            {
                                "open": [mid],
                                "high": [mid],
                                "low": [mid],
                                "close": [mid],
                                "bid": [bid],
                                "ask": [ask],
                            },
                            index=[time],
                        )

                        tick_dfs.append(df)
                except Exception as e:
                    logger.error(f"Error parsing tick: {e}")

            # Combine ticks into a single DataFrame
            if tick_dfs:
                combined_ticks = pd.concat(tick_dfs)

                # Aggregate into 1-minute candles
                new_data = realtime_connector.aggregate_tick_data(
                    combined_ticks, timeframe="1m"
                )

                if not new_data.empty:
                    # Update the chart with new data
                    updated_chart = realtime_connector.update_chart_with_realtime_data(
                        chart=chart,
                        new_data=new_data,
                        max_points=200,  # Keep only the latest 200 candles
                        update_indicators=True,
                    )

                    # Check for triggered alerts
                    triggered_alerts = realtime_connector.check_alerts(
                        alerts_config, new_data
                    )

                    # Clear the buffer
                    tick_buffer.clear()
                    last_update_time = current_time

                    return updated_chart, triggered_alerts

        return chart, []

    # Mock message callback to update the chart
    def on_message(message):
        tick_buffer.append(message)
        updated_chart, alerts = process_realtime_data()

        # Display triggered alerts
        for alert in alerts:
            logger.info(f"ALERT: {alert['message']}")

    # Set the message callback
    connection_info["on_message_callback"] = on_message

    # Show initial chart
    chart.show()

    # In a real application, you would keep the script running
    # For this example, we'll just run for a short time
    logger.info("Starting real-time updates simulation...")

    try:
        # Run for 60 seconds
        end_time = datetime.now() + timedelta(seconds=60)
        while datetime.now() < end_time:
            time.sleep(1)
            logger.info("Waiting for updates...")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    finally:
        # Clean up
        logger.info("Stopping WebSocket connection...")
        connection_info["status"] = "closed"

    logger.info("Example completed")

    # For a real application, you would keep the chart updated and display it
    # in a web interface using Dash, Flask, or a similar framework


if __name__ == "__main__":
    main()

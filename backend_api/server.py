from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import json
import os
import time
import logging
from logging.handlers import RotatingFileHandler
import threading
import websockets
import asyncio
from werkzeug.serving import make_server

from services.talib.ohlc_service import OHLCService
from services.talib.indicator_service import IndicatorService

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
if not os.path.exists("logs"):
    os.makedirs("logs")

file_handler = RotatingFileHandler(
    "logs/talib_server.log", maxBytes=10485760, backupCount=10
)
file_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]"
    )
)
file_handler.setLevel(logging.INFO)
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)
app.logger.info("TA-Lib Server startup")

# Initialize services
ohlc_service = OHLCService()
indicator_service = IndicatorService()

# WebSocket clients
websocket_clients = {}
lock = threading.Lock()


# WebSocket subscription handler
async def websocket_handler(websocket, path):
    """Handle WebSocket connections for streaming candle and price data"""
    app.logger.info(f"New WebSocket connection: {path}")

    # Parse the path to determine what to stream
    # Format: /ws/candles/<instrument>/<timeframe> or /ws/prices/<instrument>
    path_parts = path.strip("/").split("/")

    if len(path_parts) < 3:
        app.logger.warning(f"Invalid WebSocket path: {path}")
        await websocket.close(1008, "Invalid path")
        return

    stream_type = path_parts[1]  # 'candles' or 'prices'
    instrument = path_parts[2]

    client_id = id(websocket)
    subscription_key = None

    try:
        if stream_type == "candles" and len(path_parts) >= 4:
            timeframe = path_parts[3]
            subscription_key = f"{instrument}/{timeframe}"
            app.logger.info(
                f"Client {client_id} subscribing to candles for {subscription_key}"
            )

            # Register this client for candle updates
            with lock:
                if subscription_key not in websocket_clients:
                    websocket_clients[subscription_key] = set()
                websocket_clients[subscription_key].add(websocket)

            # Send welcome message
            await websocket.send(
                json.dumps(
                    {
                        "type": "CONNECTED",
                        "message": f"Connected to candle stream for {subscription_key}",
                        "time": time.time(),
                    }
                )
            )

            # Send initial candles
            candles = ohlc_service.get_candles(instrument, timeframe, 100)
            if candles:
                await websocket.send(
                    json.dumps(
                        {
                            "type": "CANDLES",
                            "instrument": instrument,
                            "timeframe": timeframe,
                            "candles": candles,
                        }
                    )
                )

        elif stream_type == "prices":
            subscription_key = f"{instrument}/PRICE"
            app.logger.info(
                f"Client {client_id} subscribing to prices for {instrument}"
            )

            # Register this client for price updates
            with lock:
                if subscription_key not in websocket_clients:
                    websocket_clients[subscription_key] = set()
                websocket_clients[subscription_key].add(websocket)

            # Send welcome message
            await websocket.send(
                json.dumps(
                    {
                        "type": "CONNECTED",
                        "message": f"Connected to price stream for {instrument}",
                        "time": time.time(),
                    }
                )
            )

        else:
            app.logger.warning(f"Unsupported WebSocket stream type: {stream_type}")
            await websocket.close(1008, "Unsupported stream type")
            return

        # Keep the connection alive
        while True:
            try:
                message = await websocket.recv()
                data = json.loads(message)

                # Handle ping messages
                if data.get("type") == "PING":
                    await websocket.send(
                        json.dumps({"type": "PONG", "time": time.time()})
                    )
            except websockets.exceptions.ConnectionClosed:
                break

    except Exception as e:
        app.logger.error(f"WebSocket error: {str(e)}")

    finally:
        # Clean up when the client disconnects
        app.logger.info(f"Client {client_id} disconnected from {subscription_key}")

        if subscription_key and subscription_key in websocket_clients:
            with lock:
                websocket_clients[subscription_key].discard(websocket)

                # Remove the set if it's empty
                if not websocket_clients[subscription_key]:
                    del websocket_clients[subscription_key]


# Function to send updates to WebSocket clients
async def broadcast_update(key, data):
    """Broadcast an update to all clients subscribed to the given key"""
    if key not in websocket_clients:
        return

    # Copy the set to avoid modification during iteration
    clients = set()
    with lock:
        clients = websocket_clients[key].copy()

    # Convert data to JSON
    message = json.dumps(data)

    # Send to all clients
    for websocket in clients:
        try:
            await websocket.send(message)
        except Exception as e:
            app.logger.error(f"Error sending to WebSocket client: {str(e)}")
            # Don't remove here, let the handler do it when the connection is closed


# Register the broadcast function with the OHLC service
ohlc_service.register_update_callback(broadcast_update)


# Start WebSocket server
async def start_websocket_server():
    """Start the WebSocket server"""
    host = os.environ.get("WS_HOST", "0.0.0.0")
    port = int(os.environ.get("WS_PORT", 9006))

    app.logger.info(f"Starting WebSocket server on {host}:{port}")

    server = await websockets.serve(websocket_handler, host, port)

    app.logger.info("WebSocket server started")

    return server


# Route to check server status
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "timestamp": time.time()})


@app.route("/price/<instrument>", methods=["POST"])
def receive_price_data(instrument):
    """Endpoint to receive price data directly from the proxy server"""
    try:
        price_data = request.get_json()

        if not price_data:
            return jsonify({"error": "No price data provided"}), 400

        # Pass to OHLC service
        result = ohlc_service.receive_price_data(instrument, price_data)

        if result:
            return (
                jsonify(
                    {
                        "status": "success",
                        "message": f"Price data for {instrument} processed",
                    }
                ),
                200,
            )
        else:
            return jsonify({"error": "Failed to process price data"}), 500
    except Exception as e:
        app.logger.error(f"Error processing price data for {instrument}: {str(e)}")
        return jsonify({"error": str(e)}), 500


# Add the WebSocket server to the event loop
websocket_server = None


def run_websocket_server():
    """Run the WebSocket server in a separate thread"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    global websocket_server
    websocket_server = loop.run_until_complete(start_websocket_server())

    loop.run_forever()


# Start the WebSocket server in a separate thread
websocket_thread = threading.Thread(target=run_websocket_server, daemon=True)
websocket_thread.start()

# Run the Flask app
if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 9005))

    app.logger.info(f"Starting Flask server on {host}:{port}")

    # Use Werkzeug server
    server = make_server(host, port, app)
    server.serve_forever()

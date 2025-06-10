# API Module

This module provides the REST API interfaces for the Forex AI Trading System, enabling programmatic access to all system functionality.

## Overview

The API module implements a FastAPI-based REST interface that allows users to interact with the Forex AI Trading System programmatically. The API provides endpoints for data retrieval, analysis execution, strategy management, and trading operations.

## Key Components

- **main.py**: The main API application entry point
  - Configures FastAPI application
  - Sets up middleware, error handling, and authentication
  - Mounts all route modules

- **auth.py**: Authentication and authorization handlers
  - Implements JWT-based authentication
  - Handles user roles and permissions

- **health_routes.py**: Health check and system status endpoints
  - System component health monitoring
  - Performance metrics

- **strategy_endpoints.py**: Trading strategy management
  - Strategy creation and configuration
  - Backtest execution
  - Strategy performance metrics

- **template_filters.py**: Custom template rendering filters for the API docs

## API Endpoints

### Authentication

- `POST /api/auth/login`: Authenticate user and get JWT token
- `POST /api/auth/refresh`: Refresh JWT token
- `GET /api/auth/user`: Get current user information

### Market Data

- `GET /api/market-data/{symbol}/{timeframe}`: Get historical market data
- `GET /api/market-data/latest/{symbol}`: Get latest price
- `GET /api/market-data/indicators/{symbol}/{indicator}`: Get technical indicator values

### Analysis

- `POST /api/analysis/technical`: Perform technical analysis
- `POST /api/analysis/fundamental`: Perform fundamental analysis
- `POST /api/analysis/sentiment`: Perform sentiment analysis
- `POST /api/analysis/market-state`: Get market state analysis

### Trading

- `GET /api/trading/accounts`: Get trading accounts
- `GET /api/trading/positions`: Get open positions
- `POST /api/trading/orders`: Place new order
- `PUT /api/trading/orders/{order_id}`: Modify existing order
- `DELETE /api/trading/orders/{order_id}`: Cancel order

### Strategies

- `GET /api/strategies`: List available strategies
- `GET /api/strategies/{id}`: Get strategy details
- `POST /api/strategies`: Create new strategy
- `PUT /api/strategies/{id}`: Update existing strategy
- `DELETE /api/strategies/{id}`: Delete strategy
- `POST /api/strategies/{id}/backtest`: Run backtest for strategy
- `GET /api/strategies/{id}/performance`: Get strategy performance

## Usage Examples

### Running the API

```python
# Start the API server
from forex_ai.api.main import app
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Making API Requests

```python
import requests
import json

# Authentication
response = requests.post(
    "http://localhost:8000/api/auth/login",
    json={"username": "user", "password": "password"}
)
token = response.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

# Get market data
response = requests.get(
    "http://localhost:8000/api/market-data/EUR_USD/H1",
    headers=headers,
    params={"start": "2023-01-01", "end": "2023-01-31"}
)
data = response.json()

# Run technical analysis
response = requests.post(
    "http://localhost:8000/api/analysis/technical",
    headers=headers,
    json={
        "symbol": "EUR_USD",
        "timeframe": "H1",
        "indicators": ["RSI", "MACD"],
        "start": "2023-01-01",
        "end": "2023-01-31"
    }
)
analysis = response.json()
```

## Dependencies

- **FastAPI**: Web framework for building the API
- **Pydantic**: Data validation and settings management
- **Uvicorn**: ASGI server for running the FastAPI application
- **JWT**: For authentication token handling
- **SQLAlchemy**: For database interactions
- **Core Module**: For core system functionality 
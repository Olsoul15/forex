# Data Module

This module is responsible for managing all data operations in the Forex AI Trading System, including market data retrieval, storage, and processing.

## Overview

The data module provides a unified interface for accessing various data sources and storing data in a standardized format. It handles the following:

- Historical and real-time market data retrieval
- Data storage and caching
- Data transformation and normalization
- Data pipelines for processing market data

## Key Components

### Database Connectivity

- **db_connector.py**: Central connector that provides unified database access using Supabase as the primary storage client.
  - Provides the `get_db_client()` function to retrieve a cached database client instance
  - Acts as a facade that delegates to the Supabase client for all database operations

### Market Data Services

- **market_data.py**: Service layer for market data operations.
  - Provides the `get_market_data_service()` function to obtain a singleton instance
  - Offers methods for fetching, importing, exporting, and converting market data
  - Abstracts the details of specific data sources and pipelines

### Connectors

The `connectors/` directory contains implementations for various external data sources:

- **oanda_connector.py**: Connector for the OANDA trading platform API
- **alpha_vantage.py**: Connector for Alpha Vantage financial data API
- **news_api.py**: Connector for news APIs to retrieve market-relevant news
- **youtube_connector.py**: Connector for retrieving financial content from YouTube

### Storage

The `storage/` directory contains implementations for various storage backends:

- **supabase_client.py**: Client for interacting with Supabase (primary storage)
- **postgres_client.py**: Direct PostgreSQL client for low-level operations
- **redis_client.py**: Redis client for caching and quick-access storage

### Pipelines

The `pipelines/` directory contains data processing pipelines:

- **market_data.py**: Functions for processing market data, including conversions and transformations

## Usage Examples

### Accessing the Database

```python
from forex_ai.data.db_connector import get_db_client

# Get the database client
db_client = get_db_client()

# Use the client to perform database operations
result = db_client.from_table("market_data").select("*").execute()
```

### Using the Market Data Service

```python
from forex_ai.data.market_data import get_market_data_service

# Get the market data service
market_data = get_market_data_service()

# Fetch data for a specific symbol and timeframe
data = market_data.fetch_data(
    symbol="EUR/USD",
    timeframe="1h",
    start_date="2023-01-01",
    end_date="2023-01-31"
)

# Export to CSV
market_data.export_csv(data, "eurusd_1h_january.csv")
```

## Dependencies

- **Supabase**: Primary database client for persistent storage
- **Pandas**: Used for data manipulation and transformation
- **NumPy**: Used for numerical operations 
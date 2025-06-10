# Services Module

This module provides external service integrations and internal service infrastructure for the Forex AI Trading System.

## Overview

The services module serves as the integration layer for external services and provides internal service functionality for the Forex AI Trading System. It includes interfaces to external APIs, communication protocols, authentication mechanisms, and service management infrastructure.

## Key Components

### Broker Services

- **broker_service.py**: Broker API integration
  - Implements broker API clients (OANDA, Interactive Brokers, etc.)
  - Manages authentication and session handling
  - Provides account information access
  - Handles order submission and management

### Market Data Services

- **market_data_service.py**: Market data integration
  - Implements market data provider clients
  - Manages streaming and historical data access
  - Provides instrument information
  - Handles data transformation and normalization

### News Services

- **news_service.py**: Financial news integration
  - Implements news provider clients
  - Manages news feed access and filtering
  - Provides sentiment analysis
  - Handles event detection and classification

### Database Services

- **database_service.py**: Database integration
  - Implements database access layer
  - Manages connection pooling and transactions
  - Provides query building and execution
  - Handles data migration and schema management

### Authentication Services

- **auth_service.py**: Authentication and authorization
  - Implements authentication providers
  - Manages user identity and credentials
  - Provides permission and role management
  - Handles token generation and validation

### Notification Services

- **notification_service.py**: Notification delivery
  - Implements notification providers (email, SMS, push, etc.)
  - Manages notification templates and rendering
  - Provides delivery status tracking
  - Handles notification preferences

## Service Infrastructure

The services module implements several core infrastructure components:

### Service Registry

- **service_registry.py**: Service management
  - Implements service registration and discovery
  - Manages service lifecycle and dependencies
  - Provides service configuration and metadata
  - Handles service health monitoring

### Service Client

- **service_client.py**: Service client infrastructure
  - Implements HTTP/REST client infrastructure
  - Manages request/response handling
  - Provides retry and circuit breaker patterns
  - Handles rate limiting and throttling

### API Gateway

- **api_gateway.py**: API gateway functionality
  - Implements API routing and aggregation
  - Manages API versioning and documentation
  - Provides request validation and transformation
  - Handles response caching and compression

## Usage Examples

### Broker Service

```python
from forex_ai.services.broker_service import BrokerService

# Initialize the broker service
broker = BrokerService(provider="oanda")

# Get account information
account_info = broker.get_account_info()
print(f"Account Balance: {account_info.balance}")
print(f"Margin Available: {account_info.margin_available}")

# Get open positions
positions = broker.get_open_positions()
for position in positions:
    print(f"Instrument: {position.instrument}")
    print(f"Units: {position.units}")
    print(f"Average Price: {position.average_price}")
    print(f"Unrealized P/L: {position.unrealized_pl}")

# Place a market order
order_result = broker.place_market_order(
    instrument="EUR_USD",
    units=10000,  # Buy 10,000 units (0.1 lot)
    stop_loss_pips=30,
    take_profit_pips=50
)
print(f"Order ID: {order_result.order_id}")
print(f"Fill Price: {order_result.fill_price}")
```

### Market Data Service

```python
from forex_ai.services.market_data_service import MarketDataService
from datetime import datetime, timedelta

# Initialize the market data service
market_data = MarketDataService(provider="oanda")

# Get current prices
current_price = market_data.get_current_price("EUR_USD")
print(f"EUR/USD Bid: {current_price.bid}, Ask: {current_price.ask}")

# Get historical candles
end_time = datetime.now()
start_time = end_time - timedelta(days=7)

candles = market_data.get_candles(
    instrument="EUR_USD",
    granularity="H1",
    start=start_time,
    end=end_time
)

for candle in candles[:5]:  # Print first 5 candles
    print(f"Time: {candle.time}, Open: {candle.open}, Close: {candle.close}")

# Subscribe to price updates
def price_callback(price):
    print(f"New price - Instrument: {price.instrument}, Bid: {price.bid}, Ask: {price.ask}")

subscription = market_data.subscribe_to_prices(
    instruments=["EUR_USD", "USD_JPY", "GBP_USD"],
    callback=price_callback
)

# Later, unsubscribe
# subscription.unsubscribe()
```

### News Service

```python
from forex_ai.services.news_service import NewsService
from datetime import datetime, timedelta

# Initialize the news service
news = NewsService(provider="forexlive")

# Get recent news
recent_news = news.get_recent_news(
    categories=["central_banks", "economic_data"],
    instruments=["EUR_USD"],
    max_items=10
)

for item in recent_news:
    print(f"Title: {item.title}")
    print(f"Time: {item.published_at}")
    print(f"Sentiment: {item.sentiment}")
    print(f"URL: {item.url}")

# Get economic calendar
start_date = datetime.now()
end_date = start_date + timedelta(days=7)

calendar = news.get_economic_calendar(
    start_date=start_date,
    end_date=end_date,
    importance=["high", "medium"]
)

for event in calendar:
    print(f"Event: {event.title}")
    print(f"Time: {event.time}")
    print(f"Currency: {event.currency}")
    print(f"Importance: {event.importance}")
    print(f"Forecast: {event.forecast}")
    print(f"Previous: {event.previous}")
```

### Database Service

```python
from forex_ai.services.database_service import DatabaseService

# Initialize the database service
db = DatabaseService(provider="postgres")

# Execute a query
result = db.execute_query(
    "SELECT * FROM trades WHERE instrument = %s AND created_at > %s",
    params=["EUR_USD", "2023-01-01"]
)

for row in result:
    print(f"Trade ID: {row['trade_id']}")
    print(f"Direction: {row['direction']}")
    print(f"Units: {row['units']}")
    print(f"Price: {row['price']}")

# Execute a transaction
with db.transaction():
    db.execute_query(
        "INSERT INTO trades (instrument, direction, units, price) VALUES (%s, %s, %s, %s)",
        params=["EUR_USD", "BUY", 10000, 1.1050]
    )
    
    db.execute_query(
        "UPDATE account SET balance = balance - %s",
        params=[100.25]
    )
```

## Dependencies

- **Core Module**: For system infrastructure and data models
- **Config Module**: For service configuration
- **Health Module**: For health monitoring integration
- **Requests**: For HTTP client functionality
- **SQLAlchemy**: For database services
- **WebSockets**: For streaming data connections
- **JWT**: For authentication token handling 
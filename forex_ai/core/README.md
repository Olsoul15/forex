# Core Module

This module provides the core functionality and infrastructure for the Forex AI Trading System, serving as the foundation for all other modules.

## Overview

The core module implements essential services, data routing, and infrastructure components that are used throughout the Forex AI Trading System. It provides a reliable foundation upon which all other modules are built.

## Key Components

### Data Router

- **data_router.py**: Central data distribution system
  - Routes data between system components
  - Prioritizes and manages data flows
  - Ensures efficient data delivery to consuming components

### Exceptions

- **exceptions.py**: Core exception definitions
  - Specialized exceptions for different error scenarios
  - Consistent error handling patterns

## Data Routing System

The Data Router is a central component that manages the flow of information between different parts of the system. It ensures that:

1. **Data is routed appropriately**: Data is sent to the correct consumers based on data type and routing rules
2. **Prioritization is maintained**: Critical data (like trade executions) gets priority over less time-sensitive data
3. **Backpressure is handled**: Prevents fast producers from overwhelming slow consumers
4. **Delivery is guaranteed**: Ensures data is delivered reliably, even in the face of temporary failures

### Data Types

The router handles various data types:

- **Market Data**: Price data, indicator values, etc.
- **System Events**: System status changes, errors, etc.
- **Analysis Results**: Results from analysis operations
- **Trading Signals**: Buy/sell signals from strategies
- **Execution Events**: Order execution confirmations, position updates, etc.

### Priority Levels

The router supports different priority levels:

- **Critical**: Immediate processing required (order executions, error events)
- **High**: Near-real-time processing (price updates, trading signals)
- **Normal**: Standard processing (analysis results, non-critical updates)
- **Low**: Background processing (historical data, logs)

## Usage Examples

### Using the Data Router

```python
from forex_ai.core.data_router import DataRouter, DataType, Priority

# Create a router instance
router = DataRouter()

# Register a consumer
def price_consumer(data):
    print(f"Received price: {data}")

router.register_consumer(
    consumer=price_consumer,
    data_types=[DataType.PRICE],
    priority=Priority.HIGH
)

# Publish data
router.publish(
    data={"symbol": "EUR/USD", "bid": 1.1234, "ask": 1.1236},
    data_type=DataType.PRICE,
    priority=Priority.HIGH
)
```

## Dependencies

- **Utils Module**: For logging and common utilities
- **Config Module**: For configuration settings
- **Asyncio**: For asynchronous operations
- **Redis** (optional): For distributed routing scenarios 
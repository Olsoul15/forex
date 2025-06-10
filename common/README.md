# Common Module

This module provides shared utilities, data models, and common functionality for the Forex AI Trading System.

## Overview

The common module serves as a repository for code that is used across multiple components of the Forex AI Trading System. It includes data models, utilities, constants, and helper functions that standardize common operations and promote code reuse throughout the system.

## Key Components

### Data Models

- **models.py**: Core data structures and models
  - Defines market data structures (OHLC, Tick, etc.)
  - Implements order and position models
  - Provides strategy configuration models
  - Defines system state models

### Utilities

- **utils.py**: General utility functions
  - Implements time and date handling utilities
  - Provides data conversion and formatting functions
  - Includes mathematical and statistical helpers
  - Offers general-purpose utility functions

### Constants

- **constants.py**: System-wide constants and enumerations
  - Defines market state constants
  - Implements order type enumerations
  - Provides timeframe definitions
  - Includes system configuration constants

### Error Handling

- **exceptions.py**: Custom exception classes
  - Implements domain-specific exceptions
  - Provides error classification and hierarchies
  - Includes exception handling utilities
  - Defines error codes and messages

### Configuration

- **config_utils.py**: Configuration utilities
  - Implements configuration loading and validation
  - Provides environment-specific configuration
  - Includes default configuration values
  - Offers configuration management utilities

## Data Types

The common module defines several core data types used throughout the system:

### Market Data Types

- **OHLC**: Open, High, Low, Close data with volume and time
- **Tick**: Individual price tick with bid/ask and volume
- **OrderBook**: Market depth representation
- **MarketState**: Current market conditions and state

### Order and Position Types

- **Order**: Trading order with parameters and status
- **Position**: Open trading position with metadata
- **Trade**: Executed trade information
- **Strategy**: Trading strategy configuration

## Usage Examples

### Working with Market Data

```python
from forex_ai.common.models import OHLC, Timeframe
from datetime import datetime

# Create an OHLC candle
candle = OHLC(
    instrument="EUR_USD",
    timeframe=Timeframe.H1,
    timestamp=datetime.now(),
    open=1.1050,
    high=1.1080,
    low=1.1030,
    close=1.1060,
    volume=1250,
    complete=True
)

# Access candle properties
print(f"Instrument: {candle.instrument}")
print(f"Timeframe: {candle.timeframe}")
print(f"Range: {candle.high - candle.low}")
print(f"Body: {abs(candle.close - candle.open)}")
print(f"Is bullish: {candle.is_bullish()}")
```

### Using Utility Functions

```python
from forex_ai.common.utils import (
    convert_timeframe,
    calculate_pip_value,
    format_price,
    timestamp_to_string
)

# Convert between timeframes
minutes = convert_timeframe("H4", to_minutes=True)  # 240
print(f"H4 is {minutes} minutes")

# Calculate pip value
pip_value = calculate_pip_value(
    instrument="USD_JPY",
    units=10000,
    price=133.50
)
print(f"Pip value: ${pip_value:.2f}")

# Format price with appropriate precision
formatted_price = format_price("EUR_USD", 1.10523)  # "1.10523"
print(f"Formatted price: {formatted_price}")

# Convert timestamp to string
time_str = timestamp_to_string(datetime.now(), "%Y-%m-%d %H:%M")
print(f"Formatted time: {time_str}")
```

### Working with Exceptions

```python
from forex_ai.common.exceptions import (
    OrderExecutionError,
    InsufficientMarginError,
    InvalidParameterError
)

# Handle specific error types
try:
    # Some operation that might fail
    pass
except InsufficientMarginError as e:
    print(f"Margin issue: {e}")
    # Handle margin error
except OrderExecutionError as e:
    print(f"Execution error: {e}")
    # Handle execution error
except InvalidParameterError as e:
    print(f"Parameter error: {e}")
    # Handle parameter error
```

### Working with Configuration

```python
from forex_ai.common.config_utils import (
    load_config,
    get_environment,
    set_config_value
)

# Load configuration
config = load_config()

# Get configuration values with defaults
api_url = config.get("api", {}).get("url", "https://default-api.example.com")
log_level = config.get("logging", {}).get("level", "INFO")

# Get current environment
env = get_environment()  # "development", "production", etc.
is_production = env == "production"

# Set configuration value
set_config_value("trading.max_positions", 10)
```

## Dependencies

- **Core Module**: For system infrastructure
- **Pydantic**: For data validation and modeling
- **Python-dateutil**: For date and time manipulation
- **Typing Extensions**: For enhanced type hints 
# Execution Module

This module handles the execution of trading strategies and order management for the Forex AI Trading System.

## Overview

The execution module is responsible for translating trading signals and decisions into actual market orders. It provides the infrastructure to execute, monitor, and manage trades across different brokers and trading platforms.

## Key Components

### Order Management

- **order_manager.py**: Core order management and execution 
  - Handles order creation, submission, modification, and cancellation
  - Implements order types (market, limit, stop, etc.)
  - Tracks order status and history
  - Manages order validation and risk checks

### Broker Integration

- **broker_controller.py**: Unified broker interface
  - Abstracts broker-specific API differences
  - Manages connections to trading platforms
  - Handles authentication and session management
  - Implements failover between brokers

### Position Management

- **position_manager.py**: Position tracking and management
  - Monitors open positions and exposure
  - Implements position sizing algorithms
  - Manages stop-loss and take-profit modifications
  - Handles partial closes and position adjustments

## Execution Process

The execution workflow typically follows these steps:

1. **Signal Reception**: Receive trading signals from strategy components
2. **Order Creation**: Convert signals into executable orders
3. **Risk Validation**: Verify compliance with risk parameters
4. **Order Execution**: Submit orders to the broker
5. **Order Monitoring**: Track order status until filled or canceled
6. **Position Tracking**: Monitor and manage the resulting positions
7. **Execution Reporting**: Log execution details for analysis

## Supported Brokers

The module supports integration with various brokers through adapter implementations:

- **OANDA**: FX and CFD trading
- **Interactive Brokers**: Multi-asset trading
- **MetaTrader**: MT4/MT5 integration
- **Alpaca**: Equity trading

## Usage Examples

### Basic Order Execution

```python
from forex_ai.execution.order_manager import OrderManager
from forex_ai.common.models import OrderRequest, OrderType, Side

# Initialize the order manager
order_manager = OrderManager()

# Create an order request
order_request = OrderRequest(
    instrument="EUR_USD",
    units=10000,  # 0.1 standard lot
    order_type=OrderType.MARKET,
    side=Side.BUY,
    take_profit_pips=50,
    stop_loss_pips=30
)

# Execute the order
order_result = order_manager.execute_order(order_request)

print(f"Order ID: {order_result.order_id}")
print(f"Execution Status: {order_result.status}")
print(f"Fill Price: {order_result.fill_price}")
```

### Position Management

```python
from forex_ai.execution.position_manager import PositionManager

# Initialize the position manager
position_manager = PositionManager()

# Get all open positions
positions = position_manager.get_open_positions()

for position in positions:
    print(f"Instrument: {position.instrument}")
    print(f"Direction: {position.direction}")
    print(f"Size: {position.units}")
    print(f"Entry Price: {position.avg_price}")
    print(f"Current P/L: {position.unrealized_pl}")
    
# Modify a position's stop loss
position_manager.modify_stop_loss(
    instrument="EUR_USD",
    new_stop_loss_price=1.0850
)

# Close a position partially
position_manager.partial_close(
    instrument="EUR_USD",
    units_to_close=5000  # Close 0.05 standard lots
)
```

## Dependencies

- **Core Module**: For system infrastructure and data models
- **Config Module**: For execution configuration settings
- **Broker APIs**: For broker-specific integrations
- **Services Module**: For authentication and rate limiting
- **Health Module**: For health monitoring integration 
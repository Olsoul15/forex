"""
Mock Database for Account Management.

This module provides mock database functionality for account management.
In a production environment, this would be replaced with real database or broker API calls.
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union

from app.models.account_models import (
    AccountType,
    OrderType,
    OrderStatus,
    OrderDirection,
    PositionStatus,
    AccountSummary,
    AccountMetrics,
    Order,
    Position,
    Trade,
    Transaction,
)

# Setup logging
logger = logging.getLogger(__name__)

# Mock databases
accounts_db: Dict[str, AccountSummary] = {}
metrics_db: Dict[str, AccountMetrics] = {}
orders_db: Dict[str, Order] = {}
positions_db: Dict[str, Position] = {}
trades_db: Dict[str, Trade] = {}
transactions_db: Dict[str, Transaction] = {}
balance_history_db: Dict[str, List[Dict[str, Any]]] = {}


def initialize_db():
    """Initialize the mock database with example accounts."""
    # Create a demo account
    demo_id = f"demo-{uuid.uuid4().hex[:8]}"
    accounts_db[demo_id] = AccountSummary(
        account_id=demo_id,
        name="Demo Account",
        currency="USD",
        balance=10000.0,
        equity=10000.0,
        margin_used=0.0,
        margin_available=10000.0,
        unrealized_pl=0.0,
        realized_pl=0.0,
        open_position_count=0,
        pending_order_count=0,
        account_type=AccountType.DEMO,
        leverage=100.0,
        margin_rate=0.01,
        created_at=datetime.now() - timedelta(days=30),
        provider="OANDA",
    )

    # Add metrics for the demo account
    metrics_db[demo_id] = AccountMetrics(
        win_rate=0.0,
        profit_factor=0.0,
        expectancy=0.0,
        avg_win=0.0,
        avg_loss=0.0,
        max_drawdown=0.0,
        sharpe_ratio=0.0,
        sortino_ratio=0.0,
        total_trades=0,
        winning_trades=0,
        losing_trades=0,
        time_period="all",
        timestamp=datetime.now(),
    )

    # Create a live account
    live_id = f"live-{uuid.uuid4().hex[:8]}"
    accounts_db[live_id] = AccountSummary(
        account_id=live_id,
        name="Live Account",
        currency="USD",
        balance=5000.0,
        equity=5000.0,
        margin_used=0.0,
        margin_available=5000.0,
        unrealized_pl=0.0,
        realized_pl=0.0,
        open_position_count=0,
        pending_order_count=0,
        account_type=AccountType.LIVE,
        leverage=50.0,
        margin_rate=0.02,
        created_at=datetime.now() - timedelta(days=60),
        provider="OANDA",
    )

    # Add metrics for the live account
    metrics_db[live_id] = AccountMetrics(
        win_rate=0.0,
        profit_factor=0.0,
        expectancy=0.0,
        avg_win=0.0,
        avg_loss=0.0,
        max_drawdown=0.0,
        sharpe_ratio=0.0,
        sortino_ratio=0.0,
        total_trades=0,
        winning_trades=0,
        losing_trades=0,
        time_period="all",
        timestamp=datetime.now(),
    )

    # Create a backtest account
    backtest_id = f"backtest-{uuid.uuid4().hex[:8]}"
    accounts_db[backtest_id] = AccountSummary(
        account_id=backtest_id,
        name="Backtest Account",
        currency="USD",
        balance=100000.0,
        equity=100000.0,
        margin_used=0.0,
        margin_available=100000.0,
        unrealized_pl=0.0,
        realized_pl=0.0,
        open_position_count=0,
        pending_order_count=0,
        account_type=AccountType.BACKTEST,
        leverage=200.0,
        margin_rate=0.005,
        created_at=datetime.now() - timedelta(days=10),
        provider="OANDA",
    )

    # Add metrics for the backtest account
    metrics_db[backtest_id] = AccountMetrics(
        win_rate=0.0,
        profit_factor=0.0,
        expectancy=0.0,
        avg_win=0.0,
        avg_loss=0.0,
        max_drawdown=0.0,
        sharpe_ratio=0.0,
        sortino_ratio=0.0,
        total_trades=0,
        winning_trades=0,
        losing_trades=0,
        time_period="all",
        timestamp=datetime.now(),
    )

    logger.info(f"Initialized account database with {len(accounts_db)} accounts")

    return {"demo_id": demo_id, "live_id": live_id, "backtest_id": backtest_id}


# Account CRUD operations
def get_accounts() -> List[AccountSummary]:
    """Get list of all accounts."""
    return list(accounts_db.values())


def get_account_by_id(account_id: str) -> Optional[AccountSummary]:
    """Get account by ID."""
    return accounts_db.get(account_id)


def get_accounts_by_user(user_id: str) -> List[AccountSummary]:
    """Get accounts belonging to a specific user."""
    return [account for account in accounts_db.values() if account.user_id == user_id]


def create_account(account_data: Dict[str, Any]) -> AccountSummary:
    """Create a new account."""
    account_id = account_data.get("account_id", f"account-{uuid.uuid4().hex[:8]}")

    account = AccountSummary(
        account_id=account_id,
        name=account_data.get("name", f"Account {account_id}"),
        currency=account_data.get("currency", "USD"),
        balance=account_data.get("balance", 10000.0),
        equity=account_data.get("balance", 10000.0),
        margin_used=0.0,
        margin_available=account_data.get("balance", 10000.0),
        unrealized_pl=0.0,
        realized_pl=0.0,
        open_position_count=0,
        pending_order_count=0,
        account_type=account_data.get("account_type", AccountType.DEMO),
        leverage=account_data.get("leverage", 100.0),
        margin_rate=account_data.get("margin_rate", 0.01),
        created_at=datetime.now(),
        provider=account_data.get("provider", "OANDA"),
    )

    accounts_db[account_id] = account

    # Initialize metrics
    metrics_db[account_id] = AccountMetrics(
        win_rate=0.0,
        profit_factor=0.0,
        expectancy=0.0,
        avg_win=0.0,
        avg_loss=0.0,
        max_drawdown=0.0,
        sharpe_ratio=0.0,
        sortino_ratio=0.0,
        total_trades=0,
        winning_trades=0,
        losing_trades=0,
        time_period="all",
        timestamp=datetime.now(),
    )

    # Initialize balance history
    balance_history_db[account_id] = [
        {
            "timestamp": datetime.now(),
            "balance": account.balance,
            "equity": account.balance,
            "reason": "ACCOUNT_CREATE",
        }
    ]

    return account


def update_account(
    account_id: str, update_data: Dict[str, Any]
) -> Optional[AccountSummary]:
    """Update an existing account."""
    account = accounts_db.get(account_id)

    if not account:
        return None

    # Update fields
    for key, value in update_data.items():
        if hasattr(account, key):
            setattr(account, key, value)

    account.updated_at = datetime.now()

    return account


def delete_account(account_id: str) -> bool:
    """Delete an account."""
    if account_id in accounts_db:
        # Check for open positions or orders
        for position in positions_db.values():
            if (
                position.account_id == account_id
                and position.status == PositionStatus.OPEN
            ):
                logger.warning(
                    f"Cannot delete account {account_id} with open positions"
                )
                return False

        for order in orders_db.values():
            if order.account_id == account_id and order.status in [
                OrderStatus.PENDING,
                OrderStatus.OPEN,
            ]:
                logger.warning(f"Cannot delete account {account_id} with open orders")
                return False

        # Delete the account and related data
        del accounts_db[account_id]

        if account_id in metrics_db:
            del metrics_db[account_id]

        if account_id in balance_history_db:
            del balance_history_db[account_id]

        # Delete positions, orders, trades, transactions
        positions_to_delete = [
            pos_id
            for pos_id, pos in positions_db.items()
            if pos.account_id == account_id
        ]
        for pos_id in positions_to_delete:
            del positions_db[pos_id]

        orders_to_delete = [
            order_id
            for order_id, order in orders_db.items()
            if order.account_id == account_id
        ]
        for order_id in orders_to_delete:
            del orders_db[order_id]

        trades_to_delete = [
            trade_id
            for trade_id, trade in trades_db.items()
            if trade.account_id == account_id
        ]
        for trade_id in trades_to_delete:
            del trades_db[trade_id]

        transactions_to_delete = [
            tx_id
            for tx_id, tx in transactions_db.items()
            if tx.account_id == account_id
        ]
        for tx_id in transactions_to_delete:
            del transactions_db[tx_id]

        return True

    return False


# Order CRUD operations
def get_orders(account_id: str) -> List[Order]:
    """Get all orders for an account."""
    return [order for order in orders_db.values() if order.account_id == account_id]


def get_order_by_id(order_id: str) -> Optional[Order]:
    """Get order by ID."""
    return orders_db.get(order_id)


def create_order(account_id: str, order_data: Dict[str, Any]) -> Optional[Order]:
    """Create a new order."""
    account = accounts_db.get(account_id)

    if not account:
        logger.error(f"Account {account_id} not found")
        return None

    order_id = order_data.get("id", f"order-{uuid.uuid4().hex[:8]}")
    order_type = order_data.get("type", OrderType.MARKET)
    units = order_data.get("units", 0.0)

    # Determine direction from units
    direction = OrderDirection.BUY if units > 0 else OrderDirection.SELL

    order = Order(
        id=order_id,
        account_id=account_id,
        instrument=order_data.get("instrument", "EUR_USD"),
        units=units,
        type=order_type,
        direction=direction,
        price=order_data.get("price"),  # Limit/Stop price
        status=OrderStatus.PENDING,
        time_in_force=order_data.get("time_in_force", "GTC"),
        stop_loss=order_data.get("stop_loss"),
        take_profit=order_data.get("take_profit"),
        created_at=datetime.now(),
        updated_at=datetime.now(),
        filled_at=None,
        canceled_at=None,
        execution_price=None,
        client_request_id=order_data.get("client_request_id"),
        strategy_id=order_data.get("strategy_id"),
    )

    orders_db[order_id] = order

    # Update account order count
    account.open_order_count += 1

    # Create a transaction for the order creation
    transaction_id = f"tx-{uuid.uuid4().hex[:8]}"
    transactions_db[transaction_id] = Transaction(
        id=transaction_id,
        account_id=account_id,
        type="ORDER_CREATE",
        instrument=order.instrument,
        units=order.units,
        price=order.price,
        timestamp=datetime.now(),
        details={
            "order_id": order_id,
            "order_type": order_type.value,
            "time_in_force": order.time_in_force,
            "client_request_id": order.client_request_id,
            "strategy_id": order.strategy_id,
        },
    )

    return order


def update_order(order_id: str, update_data: Dict[str, Any]) -> Optional[Order]:
    """Update an existing order."""
    order = orders_db.get(order_id)

    if not order:
        return None

    # Can't update filled or canceled orders
    if order.status in [OrderStatus.FILLED, OrderStatus.CANCELED]:
        logger.warning(f"Cannot update order {order_id} with status {order.status}")
        return order

    # Update fields
    for key, value in update_data.items():
        if hasattr(order, key):
            setattr(order, key, value)

    order.updated_at = datetime.now()

    # Create a transaction for the order update
    transaction_id = f"tx-{uuid.uuid4().hex[:8]}"
    transactions_db[transaction_id] = Transaction(
        id=transaction_id,
        account_id=order.account_id,
        type="ORDER_UPDATE",
        instrument=order.instrument,
        units=order.units,
        price=order.price,
        timestamp=datetime.now(),
        details={"order_id": order_id, "updated_fields": list(update_data.keys())},
    )

    return order


def cancel_order(order_id: str) -> Optional[Order]:
    """Cancel an open order."""
    order = orders_db.get(order_id)

    if not order:
        return None

    # Can't cancel filled or already canceled orders
    if order.status in [OrderStatus.FILLED, OrderStatus.CANCELED]:
        logger.warning(f"Cannot cancel order {order_id} with status {order.status}")
        return order

    # Update order status
    order.status = OrderStatus.CANCELED
    order.canceled_at = datetime.now()
    order.updated_at = datetime.now()

    # Update account order count
    account = accounts_db.get(order.account_id)
    if account:
        account.open_order_count -= 1

    # Create a transaction for the order cancellation
    transaction_id = f"tx-{uuid.uuid4().hex[:8]}"
    transactions_db[transaction_id] = Transaction(
        id=transaction_id,
        account_id=order.account_id,
        type="ORDER_CANCEL",
        instrument=order.instrument,
        units=order.units,
        price=order.price,
        timestamp=datetime.now(),
        details={"order_id": order_id, "reason": "USER_REQUEST"},
    )

    return order


# Position CRUD operations
def get_positions(account_id: str) -> List[Position]:
    """Get all positions for an account."""
    return [
        position
        for position in positions_db.values()
        if position.account_id == account_id
    ]


def get_open_positions(account_id: str) -> List[Position]:
    """Get open positions for an account."""
    return [
        position
        for position in positions_db.values()
        if position.account_id == account_id and position.status == PositionStatus.OPEN
    ]


def get_position_by_id(position_id: str) -> Optional[Position]:
    """Get position by ID."""
    return positions_db.get(position_id)


def close_position(
    position_id: str, units: Optional[float] = None
) -> Optional[Dict[str, Any]]:
    """Close a position or part of it."""
    position = positions_db.get(position_id)

    if not position:
        return None

    # Can't close already closed positions
    if position.status != PositionStatus.OPEN:
        logger.warning(
            f"Cannot close position {position_id} with status {position.status}"
        )
        return None

    account = accounts_db.get(position.account_id)
    if not account:
        logger.error(f"Account {position.account_id} not found")
        return None

    # Units to close (all by default)
    closing_units = units if units is not None else position.units

    # Validate units
    if closing_units <= 0 or closing_units > position.units:
        logger.error(f"Invalid units {closing_units} for position {position_id}")
        return None

    # Get current market price (simplified mock)
    current_price = 1.1  # This would be retrieved from market data service

    # Calculate profit/loss
    direction_multiplier = 1.0 if position.direction == OrderDirection.BUY else -1.0
    price_diff = (current_price - position.avg_price) * direction_multiplier
    unit_ratio = closing_units / position.units

    # PL in account currency
    realized_pl = price_diff * closing_units

    # Update account balance and metrics
    account.balance += realized_pl
    account.equity += realized_pl
    account.margin_available += (position.margin_used * unit_ratio) + realized_pl
    account.margin_used -= position.margin_used * unit_ratio
    account.realized_pl += realized_pl
    account.unrealized_pl -= position.unrealized_pl * unit_ratio

    # Create a transaction for the position close
    transaction_id = f"tx-{uuid.uuid4().hex[:8]}"
    transactions_db[transaction_id] = Transaction(
        id=transaction_id,
        account_id=position.account_id,
        type="POSITION_CLOSE",
        instrument=position.instrument,
        units=closing_units,
        price=current_price,
        timestamp=datetime.now(),
        details={
            "position_id": position_id,
            "realized_pl": realized_pl,
            "avg_open_price": position.avg_price,
            "close_price": current_price,
        },
    )

    # Update position or close it completely
    partially_closed = closing_units < position.units

    if partially_closed:
        # Update position with remaining units
        position.units -= closing_units
        position.margin_used -= position.margin_used * unit_ratio
        position.unrealized_pl -= position.unrealized_pl * unit_ratio
    else:
        # Close position completely
        position.status = PositionStatus.CLOSED
        position.close_time = datetime.now()
        position.close_price = current_price
        position.realized_pl += realized_pl
        position.unrealized_pl = 0.0
        position.margin_used = 0.0

        # Update account position count
        account.open_position_count -= 1

    # Update account metrics
    update_account_metrics(account.id, realized_pl)

    # Add to balance history
    if account.id in balance_history_db:
        balance_history_db[account.id].append(
            {
                "timestamp": datetime.now(),
                "balance": account.balance,
                "equity": account.equity,
                "reason": "POSITION_CLOSE",
                "details": {
                    "position_id": position_id,
                    "instrument": position.instrument,
                    "units": closing_units,
                    "pl": realized_pl,
                },
            }
        )

    return {
        "position_id": position_id,
        "transaction_id": transaction_id,
        "realized_pl": realized_pl,
        "remaining_units": position.units if partially_closed else 0,
        "position_status": position.status.value,
        "close_price": current_price,
    }


# Trade operations
def get_trades(account_id: str) -> List[Trade]:
    """Get all trades for an account."""
    return [trade for trade in trades_db.values() if trade.account_id == account_id]


def get_trade_by_id(trade_id: str) -> Optional[Trade]:
    """Get trade by ID."""
    return trades_db.get(trade_id)


# Transaction operations
def get_transactions(
    account_id: str, limit: int = 50, offset: int = 0
) -> List[Transaction]:
    """Get transactions for an account."""
    account_transactions = [
        tx for tx in transactions_db.values() if tx.account_id == account_id
    ]
    # Sort by timestamp descending
    account_transactions.sort(key=lambda tx: tx.timestamp, reverse=True)

    return account_transactions[offset : offset + limit]


def get_transaction_by_id(transaction_id: str) -> Optional[Transaction]:
    """Get transaction by ID."""
    return transactions_db.get(transaction_id)


# Balance history
def get_balance_history(account_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    """Get balance history for an account."""
    if account_id not in balance_history_db:
        return []

    history = balance_history_db[account_id]
    # Sort by timestamp ascending (oldest first)
    history.sort(key=lambda entry: entry["timestamp"])

    return history[-limit:]


# Account metrics operations
def get_account_metrics(account_id: str) -> Optional[AccountMetrics]:
    """Get metrics for an account."""
    return metrics_db.get(account_id)


def update_account_metrics(account_id: str, trade_pl: float = 0.0):
    """Update account metrics after a trade."""
    metrics = metrics_db.get(account_id)

    if not metrics:
        return

    # Update total trades
    metrics.total_trades += 1

    # Update profitable/losing trades
    if trade_pl > 0:
        metrics.profitable_trades += 1
        metrics.avg_win = (
            (metrics.avg_win * (metrics.profitable_trades - 1)) + trade_pl
        ) / metrics.profitable_trades
        metrics.largest_win = max(metrics.largest_win, trade_pl)
    elif trade_pl < 0:
        metrics.losing_trades += 1
        metrics.avg_loss = (
            (metrics.avg_loss * (metrics.losing_trades - 1)) + trade_pl
        ) / metrics.losing_trades
        metrics.largest_loss = min(metrics.largest_loss, trade_pl)

    # Update win rate
    if metrics.total_trades > 0:
        metrics.win_rate = (metrics.profitable_trades / metrics.total_trades) * 100

    # Update profit factor
    total_profit = metrics.avg_win * metrics.profitable_trades
    total_loss = abs(metrics.avg_loss * metrics.losing_trades)

    if total_loss > 0:
        metrics.profit_factor = total_profit / total_loss

    # Update timestamp
    metrics.timestamp = datetime.now()

    # For simplicity, we're not calculating other metrics like Sharpe ratio or max drawdown
    # Those would require more historical data and complex calculations

    return metrics


# Initialize the database with example accounts
if not accounts_db:
    initialize_db()

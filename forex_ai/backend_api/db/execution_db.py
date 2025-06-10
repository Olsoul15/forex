"""
Mock Database for Order Execution.

This module provides mock database functionality for order execution and day trading.
In a production environment, this would be replaced with real broker API calls.
"""

import logging
import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

from app.models.account_models import (
    OrderType,
    OrderDirection,
    OrderStatus,
    PositionStatus,
    Order,
    Position,
    Trade,
    Transaction,
)
from app.models.execution_models import (
    ExecutionMode,
    PositionSizeType,
    TimeInForce,
    RiskLevel,
    PositionSizing,
    ExecutionResult,
    RiskAnalysis,
    PositionSizeCalculation,
    OrderTriggerType,
)
from app.db import account_db

# Setup logging
logger = logging.getLogger(__name__)

# Mock databases for automated orders and order triggers
automated_orders_db: Dict[str, Dict[str, Any]] = {}
order_triggers_db: Dict[str, Dict[str, Any]] = {}
execution_history_db: Dict[str, List[Dict[str, Any]]] = {}
user_preferences_db: Dict[str, Dict[str, Any]] = {}

# Exchange rate mock database (for pip value calculations)
exchange_rates = {
    "EUR_USD": 1.0950,
    "GBP_USD": 1.2750,
    "USD_JPY": 148.50,
    "AUD_USD": 0.6700,
    "USD_CAD": 1.3300,
    "NZD_USD": 0.6100,
    "EUR_GBP": 0.8550,
}


# Initialize with default preferences
def initialize_preferences():
    """Initialize default execution preferences for accounts."""
    for account_id in account_db.accounts_db.keys():
        user_preferences_db[account_id] = {
            "default_risk_percent": 1.0,
            "default_position_size_type": PositionSizeType.RISK_PERCENT,
            "default_stop_loss_pips": 50,
            "default_take_profit_pips": 100,
            "default_time_in_force": TimeInForce.GTC,
            "trailing_stop_enabled": False,
            "trailing_stop_distance": None,
            "max_slippage_pips": 3,
            "max_spread_entry": None,
            "prevent_weekend_holdings": True,
        }

    logger.info(
        f"Initialized execution preferences for {len(user_preferences_db)} accounts"
    )


def get_user_preferences(account_id: str) -> Dict[str, Any]:
    """Get execution preferences for an account."""
    if account_id not in user_preferences_db:
        # Create default preferences if not exists
        user_preferences_db[account_id] = {
            "default_risk_percent": 1.0,
            "default_position_size_type": PositionSizeType.RISK_PERCENT,
            "default_stop_loss_pips": 50,
            "default_take_profit_pips": 100,
            "default_time_in_force": TimeInForce.GTC,
            "trailing_stop_enabled": False,
            "trailing_stop_distance": None,
            "max_slippage_pips": 3,
            "max_spread_entry": None,
            "prevent_weekend_holdings": True,
        }

    return user_preferences_db[account_id]


def update_user_preferences(
    account_id: str, preferences: Dict[str, Any]
) -> Dict[str, Any]:
    """Update execution preferences for an account."""
    if account_id not in user_preferences_db:
        # Create default first
        user_preferences_db[account_id] = {
            "default_risk_percent": 1.0,
            "default_position_size_type": PositionSizeType.RISK_PERCENT,
            "default_stop_loss_pips": 50,
            "default_take_profit_pips": 100,
            "default_time_in_force": TimeInForce.GTC,
            "trailing_stop_enabled": False,
            "trailing_stop_distance": None,
            "max_slippage_pips": 3,
            "max_spread_entry": None,
            "prevent_weekend_holdings": True,
        }

    # Update preferences
    for key, value in preferences.items():
        if key in user_preferences_db[account_id]:
            user_preferences_db[account_id][key] = value

    return user_preferences_db[account_id]


# Function to get the pip value for a currency pair
def get_pip_value(instrument: str, units: float = 1.0) -> float:
    """Calculate the value of a pip for the given instrument and units."""
    base_currency, quote_currency = instrument.split("_")

    # For simplicity, assuming standard pip values
    if quote_currency == "JPY":
        pip_value = 0.01  # For JPY pairs, a pip is 0.01
    else:
        pip_value = 0.0001  # For most pairs, a pip is 0.0001

    current_price = exchange_rates.get(instrument, 1.0)

    # Calculate pip value in quote currency
    pip_value_quote = units * pip_value

    # For pairs where USD is the quote currency, the pip value is already in USD
    if quote_currency == "USD":
        return pip_value_quote

    # For pairs where USD is the base currency, convert to USD
    if base_currency == "USD":
        return pip_value_quote / current_price

    # For cross pairs, need to convert to USD (simplified for mock)
    return pip_value_quote * 1.0  # Simplified conversion


def analyze_risk(
    account_id: str,
    instrument: str,
    direction: OrderDirection,
    units: float,
    entry_price: float,
    stop_loss_price: float,
    take_profit_price: Optional[float] = None,
) -> RiskAnalysis:
    """Analyze risk for a potential trade."""
    account = account_db.get_account_by_id(account_id)

    if not account:
        raise ValueError(f"Account {account_id} not found")

    # Calculate risk amount
    pip_size = 0.01 if instrument.endswith("JPY") else 0.0001
    pip_value = get_pip_value(instrument, units)

    if direction == OrderDirection.BUY:
        risk_pips = (entry_price - stop_loss_price) / pip_size
        if take_profit_price:
            reward_pips = (take_profit_price - entry_price) / pip_size
    else:  # SELL
        risk_pips = (stop_loss_price - entry_price) / pip_size
        if take_profit_price:
            reward_pips = (entry_price - take_profit_price) / pip_size

    risk_amount = risk_pips * pip_value
    risk_percent = (risk_amount / account.balance) * 100

    # Calculate reward if take profit is provided
    reward_amount = None
    reward_percent = None
    risk_reward_ratio = None

    if take_profit_price:
        reward_amount = reward_pips * pip_value
        reward_percent = (reward_amount / account.balance) * 100
        risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0

    # Determine risk level
    warnings = []

    if risk_percent > 3.0:
        recommendation = "danger"
        warnings.append("Risk exceeds 3% of account balance")
    elif risk_percent > 2.0:
        recommendation = "warning"
        warnings.append("Risk exceeds 2% of account balance")
    else:
        recommendation = "safe"

    if risk_reward_ratio is not None and risk_reward_ratio < 1.5:
        warnings.append("Risk-reward ratio below 1.5:1")
        if recommendation == "safe":
            recommendation = "warning"

    return RiskAnalysis(
        account_id=account_id,
        instrument=instrument,
        direction=direction,
        units=units,
        entry_price=entry_price,
        stop_loss_price=stop_loss_price,
        take_profit_price=take_profit_price,
        risk_amount=risk_amount,
        risk_percent=risk_percent,
        reward_amount=reward_amount,
        reward_percent=reward_percent,
        risk_reward_ratio=risk_reward_ratio,
        max_loss_on_account=risk_amount,
        max_loss_percent=risk_percent,
        recommendation=recommendation,
        warnings=warnings,
        timestamp=datetime.now(),
    )


def calculate_position_size(
    account_id: str,
    instrument: str,
    entry_price: float,
    stop_loss_price: float,
    risk_amount: Optional[float] = None,
    risk_percent: Optional[float] = None,
) -> PositionSizeCalculation:
    """Calculate position size options for a trade."""
    account = account_db.get_account_by_id(account_id)

    if not account:
        raise ValueError(f"Account {account_id} not found")

    # Get pip value for the instrument
    pip_size = 0.01 if instrument.endswith("JPY") else 0.0001
    pip_distance = abs(entry_price - stop_loss_price) / pip_size

    # Calculate for fixed lot sizes
    standard_lot = 100000
    mini_lot = 10000
    micro_lot = 1000

    fixed_lots_units = micro_lot  # Default to micro lot

    # Calculate for risk percentage
    risk_percents = {"0.5%": 0.5, "1%": 1.0, "2%": 2.0, "3%": 3.0}
    risk_percent_units = {}

    for label, percent in risk_percents.items():
        risk_amount_calc = account.balance * (percent / 100)
        units = risk_amount_calc / (pip_distance * get_pip_value(instrument, 1.0))
        # Round to nearest 1000
        units = round(units / 1000) * 1000
        risk_percent_units[label] = max(units, micro_lot)

    # Calculate for risk amount
    risk_amounts = {"$50": 50, "$100": 100, "$200": 200, "$500": 500}
    risk_amount_units = {}

    for label, amount in risk_amounts.items():
        units = amount / (pip_distance * get_pip_value(instrument, 1.0))
        # Round to nearest 1000
        units = round(units / 1000) * 1000
        risk_amount_units[label] = max(units, micro_lot)

    # Calculate for position percentage
    position_percents = {"5%": 5.0, "10%": 10.0, "15%": 15.0, "20%": 20.0}
    position_percent_units = {}

    for label, percent in position_percents.items():
        units = (account.balance * (percent / 100)) / entry_price
        # Convert to standard forex units
        units = round(units / 1000) * 1000
        position_percent_units[label] = max(units, micro_lot)

    # Calculate recommended position size
    preferences = get_user_preferences(account_id)
    preferred_type = preferences["default_position_size_type"]

    if preferred_type == PositionSizeType.FIXED:
        recommended_units = fixed_lots_units
        recommended_risk_amount = pip_distance * get_pip_value(
            instrument, recommended_units
        )
        recommended_risk_percent = (recommended_risk_amount / account.balance) * 100
    elif preferred_type == PositionSizeType.RISK_PERCENT:
        pref_risk_percent = preferences["default_risk_percent"]
        label = f"{pref_risk_percent}%"
        closest_key = min(
            risk_percents.keys(),
            key=lambda k: abs(float(k.strip("%")) - pref_risk_percent),
        )
        recommended_units = risk_percent_units[closest_key]
        recommended_risk_amount = pip_distance * get_pip_value(
            instrument, recommended_units
        )
        recommended_risk_percent = (recommended_risk_amount / account.balance) * 100
    else:
        # Default to 1% risk
        recommended_units = risk_percent_units["1%"]
        recommended_risk_amount = pip_distance * get_pip_value(
            instrument, recommended_units
        )
        recommended_risk_percent = (recommended_risk_amount / account.balance) * 100

    return PositionSizeCalculation(
        account_id=account_id,
        instrument=instrument,
        entry_price=entry_price,
        stop_loss_price=stop_loss_price,
        fixed_lots_units=fixed_lots_units,
        risk_percent_units=risk_percent_units,
        risk_amount_units=risk_amount_units,
        position_percent_units=position_percent_units,
        kelly_criterion_units=None,  # Kelly requires win rate and edge data
        recommended_size_type=preferred_type,
        recommended_units=recommended_units,
        recommended_risk_amount=recommended_risk_amount,
        recommended_risk_percent=recommended_risk_percent,
        timestamp=datetime.now(),
    )


def _calculate_sl_tp_prices(
    instrument: str,
    direction: OrderDirection,
    entry_price: float,
    stop_loss_pips: Optional[int],
    take_profit_pips: Optional[int],
) -> Tuple[Optional[float], Optional[float]]:
    """Calculate stop loss and take profit prices from pip distances."""
    pip_size = 0.01 if instrument.endswith("JPY") else 0.0001

    stop_loss_price = None
    take_profit_price = None

    if stop_loss_pips is not None:
        if direction == OrderDirection.BUY:
            stop_loss_price = entry_price - (stop_loss_pips * pip_size)
        else:  # SELL
            stop_loss_price = entry_price + (stop_loss_pips * pip_size)

    if take_profit_pips is not None:
        if direction == OrderDirection.BUY:
            take_profit_price = entry_price + (take_profit_pips * pip_size)
        else:  # SELL
            take_profit_price = entry_price - (take_profit_pips * pip_size)

    return (stop_loss_price, take_profit_price)


def execute_market_order(
    account_id: str,
    instrument: str,
    direction: OrderDirection,
    position_sizing: Dict[str, Any],
    stop_loss_pips: Optional[int] = None,
    take_profit_pips: Optional[int] = None,
    time_in_force: TimeInForce = TimeInForce.FOK,
    client_request_id: Optional[str] = None,
    strategy_id: Optional[str] = None,
) -> ExecutionResult:
    """Execute a market order."""
    # Get account
    account = account_db.get_account_by_id(account_id)
    if not account:
        return ExecutionResult(
            success=False,
            message=f"Account {account_id} not found",
            timestamp=datetime.now(),
        )

    # Get current market price (simplified mock)
    current_price = exchange_rates.get(instrument, 1.0)

    # Determine position size
    units = 0.0
    sizing_type = position_sizing.get("type", PositionSizeType.FIXED)

    if sizing_type == PositionSizeType.FIXED:
        units = position_sizing.get("value", 1000.0)
    elif sizing_type == PositionSizeType.RISK_PERCENT:
        # Calculate units based on risk percent
        risk_percent = position_sizing.get("value", 1.0)
        if stop_loss_pips:
            pip_value = get_pip_value(instrument, 1.0)
            risk_amount = account.balance * (risk_percent / 100)
            units = risk_amount / (stop_loss_pips * pip_value)
            units = round(units / 1000) * 1000  # Round to nearest 1000
        else:
            units = 1000.0  # Default to micro lot if no stop loss

    # Apply direction
    actual_units = units if direction == OrderDirection.BUY else -units

    # Calculate stop loss and take profit prices
    stop_loss_price, take_profit_price = _calculate_sl_tp_prices(
        instrument, direction, current_price, stop_loss_pips, take_profit_pips
    )

    # Generate order ID
    order_id = f"order-{uuid.uuid4().hex[:8]}"

    # Create order (using account_db to maintain consistency)
    order_data = {
        "instrument": instrument,
        "units": actual_units,
        "type": OrderType.MARKET,
        "price": None,  # Market order doesn't have a price
        "stop_loss": stop_loss_price,
        "take_profit": take_profit_price,
        "time_in_force": time_in_force.value,
    }

    order = account_db.create_order(account_id, order_data)

    if not order:
        return ExecutionResult(
            success=False, message="Failed to create order", timestamp=datetime.now()
        )

    # Simulate execution (in the real world, would be asynchronous)
    # For mock purposes, we'll create a position right away
    position_id = None
    trade_id = None

    # Create a transaction for the execution
    transaction_id = f"transaction-{uuid.uuid4().hex[:8]}"
    account_db.transactions_db[transaction_id] = Transaction(
        id=transaction_id,
        account_id=account_id,
        type="ORDER_FILL",
        instrument=instrument,
        units=actual_units,
        price=current_price,
        timestamp=datetime.now(),
        details={
            "reason": "MARKET_ORDER",
            "order_id": order_id,
            "client_request_id": client_request_id,
            "strategy_id": strategy_id,
        },
    )

    # Add execution to history
    if account_id not in execution_history_db:
        execution_history_db[account_id] = []

    execution_history_db[account_id].append(
        {
            "type": "MARKET_ORDER",
            "instrument": instrument,
            "direction": direction,
            "units": units,
            "price": current_price,
            "timestamp": datetime.now(),
            "order_id": order_id,
            "transaction_id": transaction_id,
            "status": "FILLED",
        }
    )

    # Get existing position or create new one
    existing_positions = account_db.get_positions(account_id)
    matching_position = None

    for pos in existing_positions:
        if pos.instrument == instrument and pos.direction == direction:
            matching_position = pos
            break

    if matching_position:
        # Update existing position
        position_id = matching_position.id
        matching_position.units += units
        matching_position.avg_price = (matching_position.avg_price + current_price) / 2
        if stop_loss_price:
            matching_position.stop_loss = stop_loss_price
        if take_profit_price:
            matching_position.take_profit = take_profit_price
    else:
        # Create new position
        position_id = f"position-{uuid.uuid4().hex[:8]}"
        margin_used = units * current_price * 0.03  # Simplified margin calculation

        account_db.positions_db[position_id] = Position(
            id=position_id,
            account_id=account_id,
            instrument=instrument,
            units=units,
            direction=direction,
            avg_price=current_price,
            unrealized_pl=0.0,
            realized_pl=0.0,
            status=PositionStatus.OPEN,
            open_time=datetime.now(),
            stop_loss=stop_loss_price,
            take_profit=take_profit_price,
            margin_used=margin_used,
        )

        # Update account
        account.open_position_count += 1
        account.margin_used += margin_used
        account.margin_available -= margin_used

    # Create a trade
    trade_id = f"trade-{uuid.uuid4().hex[:8]}"
    account_db.trades_db[trade_id] = Trade(
        id=trade_id,
        account_id=account_id,
        instrument=instrument,
        position_id=position_id,
        units=units,
        price=current_price,
        direction=direction,
        unrealized_pl=0.0,
        realized_pl=0.0,
        open_time=datetime.now(),
        strategy_id=strategy_id,
        strategy_name=None,  # Would be populated from strategy DB in real implementation
    )

    # Update the order status
    order.status = OrderStatus.FILLED
    order.filled_at = datetime.now()
    order.execution_price = current_price

    return ExecutionResult(
        success=True,
        order_id=order_id,
        position_id=position_id,
        trade_id=trade_id,
        status=OrderStatus.FILLED,
        message="Order executed successfully",
        filled_units=units,
        filled_price=current_price,
        transaction_ids=[transaction_id],
        timestamp=datetime.now(),
    )


def execute_limit_order(
    account_id: str,
    instrument: str,
    direction: OrderDirection,
    price: float,
    position_sizing: Dict[str, Any],
    stop_loss_pips: Optional[int] = None,
    take_profit_pips: Optional[int] = None,
    time_in_force: TimeInForce = TimeInForce.GTC,
    expiry: Optional[datetime] = None,
    client_request_id: Optional[str] = None,
    strategy_id: Optional[str] = None,
) -> ExecutionResult:
    """Create a pending limit order."""
    # Get account
    account = account_db.get_account_by_id(account_id)
    if not account:
        return ExecutionResult(
            success=False,
            message=f"Account {account_id} not found",
            timestamp=datetime.now(),
        )

    # Get current market price (simplified mock)
    current_price = exchange_rates.get(instrument, 1.0)

    # Validate the limit price based on direction
    if direction == OrderDirection.BUY and price >= current_price:
        return ExecutionResult(
            success=False,
            message=f"Buy limit price ({price}) must be below current price ({current_price})",
            timestamp=datetime.now(),
        )
    elif direction == OrderDirection.SELL and price <= current_price:
        return ExecutionResult(
            success=False,
            message=f"Sell limit price ({price}) must be above current price ({current_price})",
            timestamp=datetime.now(),
        )

    # Determine position size
    units = 0.0
    sizing_type = position_sizing.get("type", PositionSizeType.FIXED)

    if sizing_type == PositionSizeType.FIXED:
        units = position_sizing.get("value", 1000.0)
    elif sizing_type == PositionSizeType.RISK_PERCENT:
        # Calculate units based on risk percent
        risk_percent = position_sizing.get("value", 1.0)
        if stop_loss_pips:
            pip_value = get_pip_value(instrument, 1.0)
            risk_amount = account.balance * (risk_percent / 100)
            units = risk_amount / (stop_loss_pips * pip_value)
            units = round(units / 1000) * 1000  # Round to nearest 1000
        else:
            units = 1000.0  # Default to micro lot if no stop loss

    # Apply direction
    actual_units = units if direction == OrderDirection.BUY else -units

    # Calculate stop loss and take profit prices
    stop_loss_price, take_profit_price = _calculate_sl_tp_prices(
        instrument, direction, price, stop_loss_pips, take_profit_pips
    )

    # Generate order ID
    order_id = f"order-{uuid.uuid4().hex[:8]}"

    # Create order (using account_db to maintain consistency)
    order_data = {
        "instrument": instrument,
        "units": actual_units,
        "type": OrderType.LIMIT,
        "price": price,
        "stop_loss": stop_loss_price,
        "take_profit": take_profit_price,
        "time_in_force": time_in_force.value,
        "expiry": expiry,
    }

    order = account_db.create_order(account_id, order_data)

    if not order:
        return ExecutionResult(
            success=False,
            message="Failed to create limit order",
            timestamp=datetime.now(),
        )

    # Add to automated orders to be processed later
    if expiry:
        automated_orders_db[order_id] = {
            "type": "LIMIT",
            "instrument": instrument,
            "price": price,
            "direction": direction,
            "units": units,
            "stop_loss": stop_loss_price,
            "take_profit": take_profit_price,
            "expiry": expiry,
            "strategy_id": strategy_id,
            "client_request_id": client_request_id,
        }

    # Create a transaction for the order creation
    transaction_id = f"transaction-{uuid.uuid4().hex[:8]}"
    account_db.transactions_db[transaction_id] = Transaction(
        id=transaction_id,
        account_id=account_id,
        type="ORDER_CREATE",
        instrument=instrument,
        units=actual_units,
        price=price,
        timestamp=datetime.now(),
        details={
            "reason": "LIMIT_ORDER",
            "order_id": order_id,
            "client_request_id": client_request_id,
            "strategy_id": strategy_id,
        },
    )

    # Add execution to history
    if account_id not in execution_history_db:
        execution_history_db[account_id] = []

    execution_history_db[account_id].append(
        {
            "type": "LIMIT_ORDER",
            "instrument": instrument,
            "direction": direction,
            "units": units,
            "price": price,
            "timestamp": datetime.now(),
            "order_id": order_id,
            "transaction_id": transaction_id,
            "status": "PENDING",
        }
    )

    return ExecutionResult(
        success=True,
        order_id=order_id,
        position_id=None,
        trade_id=None,
        status=OrderStatus.PENDING,
        message="Limit order created successfully",
        filled_units=0.0,
        filled_price=None,
        transaction_ids=[transaction_id],
        timestamp=datetime.now(),
    )


def execute_stop_order(
    account_id: str,
    instrument: str,
    direction: OrderDirection,
    price: float,
    position_sizing: Dict[str, Any],
    stop_loss_pips: Optional[int] = None,
    take_profit_pips: Optional[int] = None,
    time_in_force: TimeInForce = TimeInForce.GTC,
    expiry: Optional[datetime] = None,
    client_request_id: Optional[str] = None,
    strategy_id: Optional[str] = None,
) -> ExecutionResult:
    """Create a pending stop order."""
    # Get account
    account = account_db.get_account_by_id(account_id)
    if not account:
        return ExecutionResult(
            success=False,
            message=f"Account {account_id} not found",
            timestamp=datetime.now(),
        )

    # Get current market price (simplified mock)
    current_price = exchange_rates.get(instrument, 1.0)

    # Validate the stop price based on direction
    if direction == OrderDirection.BUY and price <= current_price:
        return ExecutionResult(
            success=False,
            message=f"Buy stop price ({price}) must be above current price ({current_price})",
            timestamp=datetime.now(),
        )
    elif direction == OrderDirection.SELL and price >= current_price:
        return ExecutionResult(
            success=False,
            message=f"Sell stop price ({price}) must be below current price ({current_price})",
            timestamp=datetime.now(),
        )

    # Determine position size
    units = 0.0
    sizing_type = position_sizing.get("type", PositionSizeType.FIXED)

    if sizing_type == PositionSizeType.FIXED:
        units = position_sizing.get("value", 1000.0)
    elif sizing_type == PositionSizeType.RISK_PERCENT:
        # Calculate units based on risk percent
        risk_percent = position_sizing.get("value", 1.0)
        if stop_loss_pips:
            pip_value = get_pip_value(instrument, 1.0)
            risk_amount = account.balance * (risk_percent / 100)
            units = risk_amount / (stop_loss_pips * pip_value)
            units = round(units / 1000) * 1000  # Round to nearest 1000
        else:
            units = 1000.0  # Default to micro lot if no stop loss

    # Apply direction
    actual_units = units if direction == OrderDirection.BUY else -units

    # Calculate stop loss and take profit prices
    stop_loss_price, take_profit_price = _calculate_sl_tp_prices(
        instrument, direction, price, stop_loss_pips, take_profit_pips
    )

    # Generate order ID
    order_id = f"order-{uuid.uuid4().hex[:8]}"

    # Create order (using account_db to maintain consistency)
    order_data = {
        "instrument": instrument,
        "units": actual_units,
        "type": OrderType.STOP,
        "price": price,
        "stop_loss": stop_loss_price,
        "take_profit": take_profit_price,
        "time_in_force": time_in_force.value,
        "expiry": expiry,
    }

    order = account_db.create_order(account_id, order_data)

    if not order:
        return ExecutionResult(
            success=False,
            message="Failed to create stop order",
            timestamp=datetime.now(),
        )

    # Add to automated orders to be processed later
    if expiry:
        automated_orders_db[order_id] = {
            "type": "STOP",
            "instrument": instrument,
            "price": price,
            "direction": direction,
            "units": units,
            "stop_loss": stop_loss_price,
            "take_profit": take_profit_price,
            "expiry": expiry,
            "strategy_id": strategy_id,
            "client_request_id": client_request_id,
        }

    # Create a transaction for the order creation
    transaction_id = f"transaction-{uuid.uuid4().hex[:8]}"
    account_db.transactions_db[transaction_id] = Transaction(
        id=transaction_id,
        account_id=account_id,
        type="ORDER_CREATE",
        instrument=instrument,
        units=actual_units,
        price=price,
        timestamp=datetime.now(),
        details={
            "reason": "STOP_ORDER",
            "order_id": order_id,
            "client_request_id": client_request_id,
            "strategy_id": strategy_id,
        },
    )

    # Add execution to history
    if account_id not in execution_history_db:
        execution_history_db[account_id] = []

    execution_history_db[account_id].append(
        {
            "type": "STOP_ORDER",
            "instrument": instrument,
            "direction": direction,
            "units": units,
            "price": price,
            "timestamp": datetime.now(),
            "order_id": order_id,
            "transaction_id": transaction_id,
            "status": "PENDING",
        }
    )

    return ExecutionResult(
        success=True,
        order_id=order_id,
        position_id=None,
        trade_id=None,
        status=OrderStatus.PENDING,
        message="Stop order created successfully",
        filled_units=0.0,
        filled_price=None,
        transaction_ids=[transaction_id],
        timestamp=datetime.now(),
    )


def process_pending_orders():
    """Process all pending orders against current market prices.

    In a real implementation, this would be called by a recurring task or event listener.
    For mock purposes, this can be called manually or upon price updates.
    """
    current_datetime = datetime.now()

    # Get all pending orders
    for order_id, order in list(account_db.orders_db.items()):
        if order.status != OrderStatus.PENDING:
            continue

        # Check for expiry
        if order.expiry and order.expiry < current_datetime:
            # Cancel expired order
            account_db.cancel_order(order_id)
            logger.info(f"Order {order_id} expired and was canceled")
            continue

        # Get current price for the instrument
        current_price = exchange_rates.get(order.instrument, 1.0)

        # Check if order should be triggered
        should_trigger = False

        if order.type == OrderType.LIMIT:
            # Limit order triggers when price goes below limit for buy, or above for sell
            if (
                order.direction == OrderDirection.BUY and current_price <= order.price
            ) or (
                order.direction == OrderDirection.SELL and current_price >= order.price
            ):
                should_trigger = True

        elif order.type == OrderType.STOP:
            # Stop order triggers when price goes above stop for buy, or below for sell
            if (
                order.direction == OrderDirection.BUY and current_price >= order.price
            ) or (
                order.direction == OrderDirection.SELL and current_price <= order.price
            ):
                should_trigger = True

        if should_trigger:
            # Trigger the order (simulate market execution)
            account = account_db.get_account_by_id(order.account_id)
            if not account:
                logger.error(
                    f"Account {order.account_id} not found for order {order_id}"
                )
                continue

            # Update order status
            order.status = OrderStatus.FILLED
            order.filled_at = current_datetime
            order.execution_price = current_price

            # Update account order count
            account.open_order_count -= 1

            # Create a position or update existing
            position_id = None
            trade_id = None

            # Get existing position or create new one
            existing_positions = account_db.get_positions(account.id)
            matching_position = None

            for pos in existing_positions:
                if (
                    pos.instrument == order.instrument
                    and pos.direction == order.direction
                ):
                    matching_position = pos
                    break

            # Extract units (absolute value)
            units = abs(order.units)

            if matching_position:
                # Update existing position
                position_id = matching_position.id
                matching_position.units += units
                matching_position.avg_price = (
                    matching_position.avg_price + current_price
                ) / 2

                # Update stop loss and take profit if specified
                if order.stop_loss:
                    matching_position.stop_loss = order.stop_loss
                if order.take_profit:
                    matching_position.take_profit = order.take_profit
            else:
                # Create new position
                position_id = f"position-{uuid.uuid4().hex[:8]}"
                margin_used = (
                    units * current_price * 0.03
                )  # Simplified margin calculation

                account_db.positions_db[position_id] = Position(
                    id=position_id,
                    account_id=account.id,
                    instrument=order.instrument,
                    units=units,
                    direction=order.direction,
                    avg_price=current_price,
                    unrealized_pl=0.0,
                    realized_pl=0.0,
                    status=PositionStatus.OPEN,
                    open_time=current_datetime,
                    stop_loss=order.stop_loss,
                    take_profit=order.take_profit,
                    margin_used=margin_used,
                )

                # Update account
                account.open_position_count += 1
                account.margin_used += margin_used
                account.margin_available -= margin_used

            # Create a trade
            trade_id = f"trade-{uuid.uuid4().hex[:8]}"
            account_db.trades_db[trade_id] = Trade(
                id=trade_id,
                account_id=account.id,
                instrument=order.instrument,
                position_id=position_id,
                units=units,
                price=current_price,
                direction=order.direction,
                unrealized_pl=0.0,
                realized_pl=0.0,
                open_time=current_datetime,
                strategy_id=order.strategy_id,
                strategy_name=None,  # Would be populated from strategy DB in real implementation
            )

            # Create a transaction for the execution
            transaction_id = f"transaction-{uuid.uuid4().hex[:8]}"
            account_db.transactions_db[transaction_id] = Transaction(
                id=transaction_id,
                account_id=account.id,
                type="ORDER_FILL",
                instrument=order.instrument,
                units=order.units,
                price=current_price,
                timestamp=current_datetime,
                details={
                    "reason": f"{order.type.value}_ORDER_FILL",
                    "order_id": order_id,
                    "client_request_id": order.client_request_id,
                    "strategy_id": order.strategy_id,
                },
            )

            # Add execution to history
            if account.id not in execution_history_db:
                execution_history_db[account.id] = []

            execution_history_db[account.id].append(
                {
                    "type": f"{order.type.value}_ORDER_FILL",
                    "instrument": order.instrument,
                    "direction": order.direction,
                    "units": units,
                    "price": current_price,
                    "timestamp": current_datetime,
                    "order_id": order_id,
                    "position_id": position_id,
                    "trade_id": trade_id,
                    "transaction_id": transaction_id,
                    "status": "FILLED",
                }
            )

            logger.info(
                f"Order {order_id} of type {order.type.value} triggered at price {current_price}"
            )


def create_order_trigger(
    account_id: str,
    instrument: str,
    trigger_type: OrderTriggerType,
    trigger_price: float,
    direction: OrderDirection,
    units: float,
    stop_loss_pips: Optional[int] = None,
    take_profit_pips: Optional[int] = None,
    expiry: Optional[datetime] = None,
    strategy_id: Optional[str] = None,
) -> str:
    """Create an order trigger that will execute when price conditions are met."""
    trigger_id = f"trigger-{uuid.uuid4().hex[:8]}"

    order_triggers_db[trigger_id] = {
        "account_id": account_id,
        "instrument": instrument,
        "trigger_type": trigger_type,
        "trigger_price": trigger_price,
        "direction": direction,
        "units": units,
        "stop_loss_pips": stop_loss_pips,
        "take_profit_pips": take_profit_pips,
        "created_at": datetime.now(),
        "expiry": expiry,
        "strategy_id": strategy_id,
        "status": "ACTIVE",
    }

    return trigger_id


def delete_order_trigger(trigger_id: str) -> bool:
    """Delete an order trigger."""
    if trigger_id in order_triggers_db:
        del order_triggers_db[trigger_id]
        return True
    return False


def process_order_triggers():
    """Process all active order triggers against current market prices.

    In a real implementation, this would be called by a recurring task or event listener.
    For mock purposes, this can be called manually or upon price updates.
    """
    current_datetime = datetime.now()

    # Process each trigger
    for trigger_id, trigger in list(order_triggers_db.items()):
        if trigger.get("status") != "ACTIVE":
            continue

        # Check for expiry
        if trigger.get("expiry") and trigger.get("expiry") < current_datetime:
            # Mark trigger as expired
            trigger["status"] = "EXPIRED"
            logger.info(f"Trigger {trigger_id} expired")
            continue

        # Get current price for the instrument
        current_price = exchange_rates.get(trigger.get("instrument"), 1.0)

        # Check if trigger should activate
        should_activate = False
        trigger_type = trigger.get("trigger_type")
        trigger_price = trigger.get("trigger_price")

        if (
            trigger_type == OrderTriggerType.PRICE_ABOVE
            and current_price > trigger_price
        ):
            should_activate = True
        elif (
            trigger_type == OrderTriggerType.PRICE_BELOW
            and current_price < trigger_price
        ):
            should_activate = True
        elif (
            trigger_type == OrderTriggerType.PRICE_EQUALS
            and current_price == trigger_price
        ):
            should_activate = True

        if should_activate:
            # Activate the trigger (execute the order)
            account_id = trigger.get("account_id")
            instrument = trigger.get("instrument")
            direction = trigger.get("direction")
            units = trigger.get("units")
            stop_loss_pips = trigger.get("stop_loss_pips")
            take_profit_pips = trigger.get("take_profit_pips")
            strategy_id = trigger.get("strategy_id")

            # Create a market order
            position_sizing = {"type": PositionSizeType.FIXED, "value": units}

            # Execute the market order
            result = execute_market_order(
                account_id=account_id,
                instrument=instrument,
                direction=direction,
                position_sizing=position_sizing,
                stop_loss_pips=stop_loss_pips,
                take_profit_pips=take_profit_pips,
                strategy_id=strategy_id,
            )

            # Update trigger status
            trigger["status"] = "TRIGGERED"
            trigger["order_id"] = result.order_id if result.success else None
            trigger["triggered_at"] = current_datetime
            trigger["result"] = "SUCCESS" if result.success else "FAILURE"

            logger.info(
                f"Trigger {trigger_id} activated and {'succeeded' if result.success else 'failed'}"
            )


# Initialize
initialize_preferences()

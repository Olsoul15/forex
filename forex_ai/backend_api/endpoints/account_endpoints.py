# Account Endpoints

"""
Account Management API Endpoints.

This module contains FastAPI endpoints for managing trading accounts, orders,
positions, and viewing account history and performance metrics.
"""

import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Path
from fastapi.responses import JSONResponse

from app.models.account_models import (
    AccountSummary,
    AccountMetrics,
    Order,
    Position,
    Trade,
    Transaction,
    BalanceEntry,
    OrderCreate,
    OrderUpdate,
    PositionClose,
    AccountListResponse,
    AccountDetailResponse,
    OrderListResponse,
    OrderDetailResponse,
    PositionListResponse,
    PositionDetailResponse,
    TradeListResponse,
    TradeDetailResponse,
    TransactionListResponse,
    BalanceHistoryResponse,
    PerformanceReportResponse,
)
from app.db import account_db

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/account", tags=["account"])


# Dependency for validating account ID
async def validate_account_id(
    account_id: str = Path(..., description="Account ID to operate on")
) -> str:
    """Validate that the account ID exists."""
    account = account_db.get_account_by_id(account_id)
    if not account:
        raise HTTPException(status_code=404, detail=f"Account {account_id} not found")
    return account_id


# Dependency for validating order ID
async def validate_order_id(
    order_id: str = Path(..., description="Order ID to operate on")
) -> str:
    """Validate that the order ID exists."""
    order = account_db.get_order_by_id(order_id)
    if not order:
        raise HTTPException(status_code=404, detail=f"Order {order_id} not found")
    return order_id


# Dependency for validating position ID
async def validate_position_id(
    position_id: str = Path(..., description="Position ID to operate on")
) -> str:
    """Validate that the position ID exists."""
    position = account_db.get_position_by_id(position_id)
    if not position:
        raise HTTPException(status_code=404, detail=f"Position {position_id} not found")
    return position_id


# Dependency for validating trade ID
async def validate_trade_id(
    trade_id: str = Path(..., description="Trade ID to operate on")
) -> str:
    """Validate that the trade ID exists."""
    trade = account_db.get_trade_by_id(trade_id)
    if not trade:
        raise HTTPException(status_code=404, detail=f"Trade {trade_id} not found")
    return trade_id


@router.get("/list", response_model=AccountListResponse)
async def list_accounts(
    provider: Optional[str] = Query(None, description="Filter accounts by provider")
):
    """
    List all trading accounts, optionally filtered by provider.

    Returns a list of account summaries from the connected broker(s).
    """
    try:
        accounts = account_db.get_accounts(provider)
        return AccountListResponse(accounts=accounts, count=len(accounts))
    except Exception as e:
        logger.error(f"Error listing accounts: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve accounts")


@router.get(
    "/{account_id}",
    response_model=AccountDetailResponse,
    dependencies=[Depends(validate_account_id)],
)
async def get_account_details(account_id: str):
    """
    Get detailed information about a specific trading account.

    Returns the account summary, along with basic metrics and open positions/orders count.
    """
    try:
        account = account_db.get_account_by_id(account_id)
        return AccountDetailResponse(
            account=account, metrics=account_db.get_account_metrics(account_id, "month")
        )
    except Exception as e:
        logger.error(f"Error getting account details: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve account details"
        )


@router.get(
    "/{account_id}/metrics",
    response_model=AccountMetrics,
    dependencies=[Depends(validate_account_id)],
)
async def get_account_metrics(
    account_id: str,
    time_period: str = Query(
        "month", description="Time period for metrics (day, week, month, year, all)"
    ),
):
    """
    Get performance metrics for a trading account.

    Returns metrics such as win rate, profit factor, and drawdown for the specified time period.
    """
    try:
        metrics = account_db.get_account_metrics(account_id, time_period)
        if not metrics:
            raise HTTPException(
                status_code=404,
                detail="Metrics not available for the specified account and time period",
            )
        return metrics
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting account metrics: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve account metrics"
        )


@router.get(
    "/{account_id}/history",
    response_model=TransactionListResponse,
    dependencies=[Depends(validate_account_id)],
)
async def get_account_history(
    account_id: str,
    from_time: Optional[datetime] = Query(
        None, description="Filter transactions from this time"
    ),
    to_time: Optional[datetime] = Query(
        None, description="Filter transactions to this time"
    ),
    count: Optional[int] = Query(50, description="Number of transactions to return"),
    type: Optional[str] = Query(None, description="Filter by transaction type"),
):
    """
    Get transaction history for a trading account.

    Returns a list of transactions, such as orders, trades, and funding operations.
    """
    try:
        transactions = account_db.get_account_history(
            account_id, from_time, to_time, count, type
        )
        return TransactionListResponse(
            transactions=transactions, count=len(transactions)
        )
    except Exception as e:
        logger.error(f"Error getting account history: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve account history"
        )


@router.get(
    "/{account_id}/balance-history",
    response_model=BalanceHistoryResponse,
    dependencies=[Depends(validate_account_id)],
)
async def get_balance_history(
    account_id: str,
    from_time: Optional[datetime] = Query(
        None, description="Filter entries from this time"
    ),
    to_time: Optional[datetime] = Query(
        None, description="Filter entries to this time"
    ),
    count: Optional[int] = Query(50, description="Number of entries to return"),
):
    """
    Get balance history for a trading account.

    Returns a list of balance entries, showing how the account balance and equity changed over time.
    """
    try:
        entries = account_db.get_balance_history(account_id, from_time, to_time, count)
        return BalanceHistoryResponse(entries=entries, count=len(entries))
    except Exception as e:
        logger.error(f"Error getting balance history: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve balance history"
        )


@router.get(
    "/{account_id}/orders",
    response_model=OrderListResponse,
    dependencies=[Depends(validate_account_id)],
)
async def list_orders(
    account_id: str,
    status: Optional[str] = Query(
        None, description="Filter orders by status (PENDING, OPEN, FILLED, CANCELLED)"
    ),
):
    """
    List orders for a trading account.

    Returns a list of orders, optionally filtered by status.
    """
    try:
        orders = account_db.get_orders(account_id, status)
        return OrderListResponse(orders=orders, count=len(orders))
    except Exception as e:
        logger.error(f"Error listing orders: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve orders")


@router.get(
    "/orders/{order_id}",
    response_model=OrderDetailResponse,
    dependencies=[Depends(validate_order_id)],
)
async def get_order_details(order_id: str):
    """
    Get detailed information about a specific order.

    Returns the complete order details, including execution information if filled.
    """
    try:
        order = account_db.get_order_by_id(order_id)
        return OrderDetailResponse(order=order)
    except Exception as e:
        logger.error(f"Error getting order details: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve order details")


@router.post(
    "/{account_id}/orders",
    response_model=OrderDetailResponse,
    dependencies=[Depends(validate_account_id)],
)
async def create_order(account_id: str, order_data: OrderCreate):
    """
    Create a new order for a trading account.

    Creates a market, limit, or stop order based on the provided order details.
    """
    try:
        order = account_db.create_order(account_id, order_data.dict())
        if not order:
            raise HTTPException(status_code=400, detail="Failed to create order")
        return OrderDetailResponse(order=order)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating order: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create order")


@router.put(
    "/orders/{order_id}",
    response_model=OrderDetailResponse,
    dependencies=[Depends(validate_order_id)],
)
async def update_order(order_id: str, order_data: OrderUpdate):
    """
    Update an existing order.

    Modifies the parameters of a pending order, such as price, units, or stop loss.
    """
    try:
        order = account_db.modify_order(order_id, order_data.dict(exclude_unset=True))
        if not order:
            raise HTTPException(status_code=400, detail="Failed to update order")
        return OrderDetailResponse(order=order)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating order: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update order")


@router.delete("/orders/{order_id}", dependencies=[Depends(validate_order_id)])
async def cancel_order(order_id: str):
    """
    Cancel a pending order.

    Cancels the specified order if it is still pending.
    """
    try:
        success = account_db.cancel_order(order_id)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to cancel order")
        return JSONResponse(content={"message": "Order cancelled successfully"})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling order: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to cancel order")


@router.get(
    "/{account_id}/positions",
    response_model=PositionListResponse,
    dependencies=[Depends(validate_account_id)],
)
async def list_positions(account_id: str):
    """
    List open positions for a trading account.

    Returns a list of currently open positions.
    """
    try:
        positions = account_db.get_positions(account_id)
        return PositionListResponse(positions=positions, count=len(positions))
    except Exception as e:
        logger.error(f"Error listing positions: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve positions")


@router.get(
    "/positions/{position_id}",
    response_model=PositionDetailResponse,
    dependencies=[Depends(validate_position_id)],
)
async def get_position_details(position_id: str):
    """
    Get detailed information about a specific position.

    Returns the complete position details, including profit/loss and margin information.
    """
    try:
        position = account_db.get_position_by_id(position_id)
        return PositionDetailResponse(position=position)
    except Exception as e:
        logger.error(f"Error getting position details: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve position details"
        )


@router.post(
    "/positions/{position_id}/close", dependencies=[Depends(validate_position_id)]
)
async def close_position(position_id: str, close_data: Optional[PositionClose] = None):
    """
    Close an open position.

    Closes the specified position, optionally partially if units are specified.
    """
    try:
        units = close_data.units if close_data else None
        success = account_db.close_position(position_id, units)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to close position")
        return JSONResponse(content={"message": "Position closed successfully"})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error closing position: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to close position")


@router.get(
    "/{account_id}/trades",
    response_model=TradeListResponse,
    dependencies=[Depends(validate_account_id)],
)
async def list_trades(
    account_id: str,
    from_time: Optional[datetime] = Query(
        None, description="Filter trades from this time"
    ),
    to_time: Optional[datetime] = Query(None, description="Filter trades to this time"),
    instrument: Optional[str] = Query(None, description="Filter by instrument"),
    status: Optional[str] = Query(None, description="Filter by status (OPEN, CLOSED)"),
):
    """
    List trades for a trading account.

    Returns a list of trades, optionally filtered by time, instrument, or status.
    """
    try:
        trades = account_db.get_trades(
            account_id, from_time, to_time, instrument, status
        )
        return TradeListResponse(trades=trades, count=len(trades))
    except Exception as e:
        logger.error(f"Error listing trades: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve trades")


@router.get(
    "/trades/{trade_id}",
    response_model=TradeDetailResponse,
    dependencies=[Depends(validate_trade_id)],
)
async def get_trade_details(trade_id: str):
    """
    Get detailed information about a specific trade.

    Returns the complete trade details, including profit/loss and related order/position info.
    """
    try:
        trade = account_db.get_trade_by_id(trade_id)
        return TradeDetailResponse(trade=trade)
    except Exception as e:
        logger.error(f"Error getting trade details: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve trade details")


@router.get(
    "/{account_id}/performance",
    response_model=PerformanceReportResponse,
    dependencies=[Depends(validate_account_id)],
)
async def get_account_performance(
    account_id: str,
    period: str = Query(
        "month", description="Time period for performance (day, week, month, year, all)"
    ),
):
    """
    Get a comprehensive performance report for a trading account.

    Returns detailed performance metrics, including instrument breakdown and strategy performance.
    """
    try:
        report = account_db.generate_performance_report(account_id, period)
        if not report:
            raise HTTPException(
                status_code=404,
                detail="Performance report not available for the specified account",
            )
        return PerformanceReportResponse(report=report)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting performance report: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Failed to generate performance report"
        )

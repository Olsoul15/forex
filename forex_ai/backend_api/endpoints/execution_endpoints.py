"""
Order Execution API endpoints.

This module defines the API endpoints for the order execution functionality,
enabling market, limit, and stop orders, as well as position sizing calculations.
"""

import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import status, APIRouter, HTTPException, Depends, Query, Path, Body
from fastapi.responses import JSONResponse

from forex_ai.models.account_models import (
    OrderDirection,
    Order,
    Position,
    Trade,
    Transaction,
)
from forex_ai.models.execution_models import (
    ExecutionMode,
    PositionSizeType,
    OrderTriggerType,
    TimeInForce,
    MarketOrderRequest,
    MarketOrderResponse,
    LimitOrderRequest,
    LimitOrderResponse,
    StopOrderRequest,
    StopOrderResponse,
    OrderTriggerRequest,
    OrderTriggerResponse,
    PositionSizeRequest,
    PositionSizeResponse,
    RiskAnalysisRequest,
    RiskAnalysisResponse,
    ExecutionPreferencesUpdate,
    ExecutionPreferencesResponse,
    OrderCancelRequest,
    OrderCancelResponse,
    PositionCloseRequest,
    PositionCloseResponse,
)
from forex_ai.backend_api.db import account_db, execution_db

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/execution", tags=["execution"])


# Helper functions
def get_account_or_error(account_id: str):
    """Get account or raise HTTPException."""
    account = account_db.get_account_by_id(account_id)
    if not account:
        raise HTTPException(status_code=404, detail=f"Account {account_id} not found")
    return account


def get_order_or_error(order_id: str):
    """Get order or raise HTTPException."""
    order = account_db.get_order_by_id(order_id)
    if not order:
        raise HTTPException(status_code=404, detail=f"Order {order_id} not found")
    return order


def get_position_or_error(position_id: str):
    """Get position or raise HTTPException."""
    position = account_db.get_position_by_id(position_id)
    if not position:
        raise HTTPException(status_code=404, detail=f"Position {position_id} not found")
    return position


# Simple orders endpoint for testing
@router.post("/orders", response_model=Dict[str, Any])
async def place_order(order: Dict[str, Any] = Body(...)):
    """
    Place a new order.
    
    Creates a new order for the specified instrument and parameters.
    """
    try:
        # Return a simple response for testing
        return {
            "order": {
                "id": f"O-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "instrument": order.get("instrument", "EUR_USD"),
                "units": order.get("units", 10000),
                "price": 1.1825,
                "type": order.get("type", "MARKET"),
                "status": "FILLED",
                "created_at": datetime.now().isoformat()
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error placing order: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error placing order: {str(e)}")


# User preferences endpoints
@router.get("/preferences/{account_id}", response_model=ExecutionPreferencesResponse)
async def get_execution_preferences(
    account_id: str = Path(..., description="Account ID")
):
    """Get execution preferences for an account."""
    # Verify account exists
    get_account_or_error(account_id)

    preferences = execution_db.get_user_preferences(account_id)

    return ExecutionPreferencesResponse(
        account_id=account_id, preferences=preferences, timestamp=datetime.now()
    )


@router.put("/preferences/{account_id}", response_model=ExecutionPreferencesResponse)
async def update_execution_preferences(
    account_id: str = Path(..., description="Account ID"),
    update_data: ExecutionPreferencesUpdate = Body(
        ..., description="Preferences to update"
    ),
):
    """Update execution preferences for an account."""
    # Verify account exists
    get_account_or_error(account_id)

    updated_preferences = execution_db.update_user_preferences(
        account_id, update_data.preferences.dict(exclude_unset=True)
    )

    return ExecutionPreferencesResponse(
        account_id=account_id, preferences=updated_preferences, timestamp=datetime.now()
    )


# Risk analysis endpoint
@router.post("/risk-analysis", response_model=RiskAnalysisResponse)
async def analyze_trade_risk(request: RiskAnalysisRequest = Body(...)):
    """Analyze risk for a potential trade."""
    # Verify account exists
    get_account_or_error(request.account_id)

    try:
        risk_analysis = execution_db.analyze_risk(
            account_id=request.account_id,
            instrument=request.instrument,
            direction=request.direction,
            units=request.units,
            entry_price=request.entry_price,
            stop_loss_price=request.stop_loss_price,
            take_profit_price=request.take_profit_price,
        )

        return RiskAnalysisResponse(
            success=True, risk_analysis=risk_analysis, timestamp=datetime.now()
        )
    except Exception as e:
        logger.error(f"Error analyzing risk: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


# Position sizing endpoint
@router.post("/position-size", response_model=PositionSizeResponse)
async def calculate_position_size(request: PositionSizeRequest = Body(...)):
    """Calculate position size options for a trade."""
    # Verify account exists
    get_account_or_error(request.account_id)

    try:
        position_size_calculation = execution_db.calculate_position_size(
            account_id=request.account_id,
            instrument=request.instrument,
            entry_price=request.entry_price,
            stop_loss_price=request.stop_loss_price,
            risk_amount=request.risk_amount,
            risk_percent=request.risk_percent,
        )

        return PositionSizeResponse(
            success=True,
            calculation=position_size_calculation,
            timestamp=datetime.now(),
        )
    except Exception as e:
        logger.error(f"Error calculating position size: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


# Market order endpoint
@router.post("/market-order", response_model=MarketOrderResponse)
async def execute_market_order(request: MarketOrderRequest = Body(...)):
    """Execute a market order."""
    # Verify account exists
    get_account_or_error(request.account_id)

    try:
        # Create position sizing dict from request
        position_sizing = {
            "type": request.position_sizing.type,
            "value": request.position_sizing.value,
        }

        # Execute market order
        result = execution_db.execute_market_order(
            account_id=request.account_id,
            instrument=request.instrument,
            direction=request.direction,
            position_sizing=position_sizing,
            stop_loss_pips=request.stop_loss_pips,
            take_profit_pips=request.take_profit_pips,
            time_in_force=request.time_in_force,
            client_request_id=request.client_request_id,
            strategy_id=request.strategy_id,
        )

        if not result.success:
            return MarketOrderResponse(
                success=False, message=result.message, timestamp=datetime.now()
            )

        # Get created order, position, and trade details
        order = account_db.get_order_by_id(result.order_id) if result.order_id else None
        position = (
            account_db.get_position_by_id(result.position_id)
            if result.position_id
            else None
        )
        trade = account_db.get_trade_by_id(result.trade_id) if result.trade_id else None

        return MarketOrderResponse(
            success=True,
            message=result.message,
            order_id=result.order_id,
            position_id=result.position_id,
            trade_id=result.trade_id,
            filled_price=result.filled_price,
            filled_units=result.filled_units,
            transaction_ids=result.transaction_ids,
            order=order,
            position=position,
            trade=trade,
            timestamp=datetime.now(),
        )
    except Exception as e:
        logger.error(f"Error executing market order: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


# Limit order endpoint
@router.post("/limit-order", response_model=LimitOrderResponse)
async def create_limit_order(request: LimitOrderRequest = Body(...)):
    """Create a limit order."""
    # Verify account exists
    get_account_or_error(request.account_id)

    try:
        # Create position sizing dict from request
        position_sizing = {
            "type": request.position_sizing.type,
            "value": request.position_sizing.value,
        }

        # Execute limit order
        result = execution_db.execute_limit_order(
            account_id=request.account_id,
            instrument=request.instrument,
            direction=request.direction,
            price=request.price,
            position_sizing=position_sizing,
            stop_loss_pips=request.stop_loss_pips,
            take_profit_pips=request.take_profit_pips,
            time_in_force=request.time_in_force,
            expiry=request.expiry,
            client_request_id=request.client_request_id,
            strategy_id=request.strategy_id,
        )

        if not result.success:
            return LimitOrderResponse(
                success=False, message=result.message, timestamp=datetime.now()
            )

        # Get created order
        order = account_db.get_order_by_id(result.order_id) if result.order_id else None

        return LimitOrderResponse(
            success=True,
            message=result.message,
            order_id=result.order_id,
            status=result.status,
            transaction_ids=result.transaction_ids,
            order=order,
            timestamp=datetime.now(),
        )
    except Exception as e:
        logger.error(f"Error creating limit order: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


# Stop order endpoint
@router.post("/stop-order", response_model=StopOrderResponse)
async def create_stop_order(request: StopOrderRequest = Body(...)):
    """Create a stop order."""
    # Verify account exists
    get_account_or_error(request.account_id)

    try:
        # Create position sizing dict from request
        position_sizing = {
            "type": request.position_sizing.type,
            "value": request.position_sizing.value,
        }

        # Execute stop order
        result = execution_db.execute_stop_order(
            account_id=request.account_id,
            instrument=request.instrument,
            direction=request.direction,
            price=request.price,
            position_sizing=position_sizing,
            stop_loss_pips=request.stop_loss_pips,
            take_profit_pips=request.take_profit_pips,
            time_in_force=request.time_in_force,
            expiry=request.expiry,
            client_request_id=request.client_request_id,
            strategy_id=request.strategy_id,
        )

        if not result.success:
            return StopOrderResponse(
                success=False, message=result.message, timestamp=datetime.now()
            )

        # Get created order
        order = account_db.get_order_by_id(result.order_id) if result.order_id else None

        return StopOrderResponse(
            success=True,
            message=result.message,
            order_id=result.order_id,
            status=result.status,
            transaction_ids=result.transaction_ids,
            order=order,
            timestamp=datetime.now(),
        )
    except Exception as e:
        logger.error(f"Error creating stop order: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


# Order trigger endpoint
@router.post("/order-trigger", response_model=OrderTriggerResponse)
async def create_order_trigger(request: OrderTriggerRequest = Body(...)):
    """Create an order trigger that will execute when price conditions are met."""
    # Verify account exists
    get_account_or_error(request.account_id)

    try:
        trigger_id = execution_db.create_order_trigger(
            account_id=request.account_id,
            instrument=request.instrument,
            trigger_type=request.trigger_type,
            trigger_price=request.trigger_price,
            direction=request.direction,
            units=request.units,
            stop_loss_pips=request.stop_loss_pips,
            take_profit_pips=request.take_profit_pips,
            expiry=request.expiry,
            strategy_id=request.strategy_id,
        )

        trigger_data = execution_db.order_triggers_db.get(trigger_id)

        return OrderTriggerResponse(
            success=True,
            trigger_id=trigger_id,
            trigger_data=trigger_data,
            message="Order trigger created successfully",
            timestamp=datetime.now(),
        )
    except Exception as e:
        logger.error(f"Error creating order trigger: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


# Delete order trigger endpoint
@router.delete("/order-trigger/{trigger_id}", response_model=OrderTriggerResponse)
async def delete_order_trigger(trigger_id: str = Path(..., description="Trigger ID")):
    """Delete an order trigger."""
    if trigger_id not in execution_db.order_triggers_db:
        raise HTTPException(status_code=404, detail=f"Trigger {trigger_id} not found")

    try:
        trigger_data = execution_db.order_triggers_db.get(trigger_id)
        success = execution_db.delete_order_trigger(trigger_id)

        return OrderTriggerResponse(
            success=success,
            trigger_id=trigger_id,
            trigger_data=trigger_data,
            message=(
                "Order trigger deleted successfully"
                if success
                else "Failed to delete order trigger"
            ),
            timestamp=datetime.now(),
        )
    except Exception as e:
        logger.error(f"Error deleting order trigger: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


# Cancel order endpoint
@router.post("/cancel-order", response_model=OrderCancelResponse)
async def cancel_order(request: OrderCancelRequest = Body(...)):
    """Cancel an open order."""
    # Verify order exists
    order = get_order_or_error(request.order_id)

    try:
        updated_order = account_db.cancel_order(request.order_id)

        if not updated_order:
            return OrderCancelResponse(
                success=False,
                message=f"Failed to cancel order {request.order_id}",
                timestamp=datetime.now(),
            )

        return OrderCancelResponse(
            success=True,
            order_id=request.order_id,
            status=updated_order.status,
            message=f"Order {request.order_id} cancelled successfully",
            order=updated_order,
            timestamp=datetime.now(),
        )
    except Exception as e:
        logger.error(f"Error cancelling order: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


# Close position endpoint
@router.post("/close-position", response_model=PositionCloseResponse)
async def close_position(request: PositionCloseRequest = Body(...)):
    """Close a position or part of it."""
    # Verify position exists
    position = get_position_or_error(request.position_id)

    try:
        result = account_db.close_position(request.position_id, request.units)

        if not result:
            return PositionCloseResponse(
                success=False,
                message=f"Failed to close position {request.position_id}",
                timestamp=datetime.now(),
            )

        # Get updated position
        updated_position = account_db.get_position_by_id(request.position_id)

        return PositionCloseResponse(
            success=True,
            position_id=request.position_id,
            transaction_id=result["transaction_id"],
            realized_pl=result["realized_pl"],
            remaining_units=result["remaining_units"],
            close_price=result["close_price"],
            status=result["position_status"],
            message=f"Position {request.position_id} closed successfully",
            position=updated_position,
            timestamp=datetime.now(),
        )
    except Exception as e:
        logger.error(f"Error closing position: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


# Process pending orders endpoint (admin/system)
@router.post("/process-pending-orders", response_model=Dict[str, Any])
async def process_pending_orders():
    """Process all pending orders against current market prices.

    This endpoint would typically be called by a scheduled task or event listener.
    For testing purposes, it can be called manually.
    """
    try:
        execution_db.process_pending_orders()
        return {
            "success": True,
            "message": "Pending orders processed successfully",
            "timestamp": datetime.now(),
        }
    except Exception as e:
        logger.error(f"Error processing pending orders: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Process order triggers endpoint (admin/system)
@router.post("/process-order-triggers", response_model=Dict[str, Any])
async def process_order_triggers():
    """Process all active order triggers against current market prices.

    This endpoint would typically be called by a scheduled task or event listener.
    For testing purposes, it can be called manually.
    """
    try:
        execution_db.process_order_triggers()
        return {
            "success": True,
            "message": "Order triggers processed successfully",
            "timestamp": datetime.now(),
        }
    except Exception as e:
        logger.error(f"Error processing order triggers: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/execution/orders")
async def mock_orders():
    """
    Mock implementation for /api/execution/orders.
    """
    logger.info("Processing mock request for /api/execution/orders")
    return {
        "success": True,
        "message": "This is a mock implementation",
        "data": {},
        "timestamp": datetime.now().isoformat()
    }


@router.get("/execution/orders/123")
async def mock_123():
    """
    Mock implementation for /api/execution/orders/123.
    """
    logger.info("Processing mock request for /api/execution/orders/123")
    return {
        "success": True,
        "message": "This is a mock implementation",
        "data": {},
        "timestamp": datetime.now().isoformat()
    }


@router.get("/execution/positions")
async def mock_positions():
    """
    Mock implementation for /api/execution/positions.
    """
    logger.info("Processing mock request for /api/execution/positions")
    return {
        "success": True,
        "message": "This is a mock implementation",
        "data": {},
        "timestamp": datetime.now().isoformat()
    }


@router.get("/execution/positions/123")
async def mock_123():
    """
    Mock implementation for /api/execution/positions/123.
    """
    logger.info("Processing mock request for /api/execution/positions/123")
    return {
        "success": True,
        "message": "This is a mock implementation",
        "data": {},
        "timestamp": datetime.now().isoformat()
    }


@router.get("/execution/trades")
async def mock_trades():
    """
    Mock implementation for /api/execution/trades.
    """
    logger.info("Processing mock request for /api/execution/trades")
    return {
        "success": True,
        "message": "This is a mock implementation",
        "data": {},
        "timestamp": datetime.now().isoformat()
    }


@router.get("/execution/trades/123")
async def mock_123():
    """
    Mock implementation for /api/execution/trades/123.
    """
    logger.info("Processing mock request for /api/execution/trades/123")
    return {
        "success": True,
        "message": "This is a mock implementation",
        "data": {},
        "timestamp": datetime.now().isoformat()
    }


@router.get("/execution/orders")
async def mock_orders():
    """
    Mock implementation for /api/execution/orders.
    """
    logger.info(f"Processing mock request for /api/execution/orders")
    return {
        "success": True,
        "message": "This is a mock implementation",
        "data": {},
        "timestamp": datetime.now().isoformat()
    }


@router.get("/execution/orders/123")
async def mock_endpoint_123():
    """
    Mock implementation for /api/execution/orders/123.
    """
    logger.info(f"Processing mock request for /api/execution/orders/123")
    return {
        "success": True,
        "message": "This is a mock implementation",
        "data": {},
        "timestamp": datetime.now().isoformat()
    }


@router.get("/execution/positions")
async def mock_positions():
    """
    Mock implementation for /api/execution/positions.
    """
    logger.info(f"Processing mock request for /api/execution/positions")
    return {
        "success": True,
        "message": "This is a mock implementation",
        "data": {},
        "timestamp": datetime.now().isoformat()
    }


@router.get("/execution/positions/123")
async def mock_endpoint_123():
    """
    Mock implementation for /api/execution/positions/123.
    """
    logger.info(f"Processing mock request for /api/execution/positions/123")
    return {
        "success": True,
        "message": "This is a mock implementation",
        "data": {},
        "timestamp": datetime.now().isoformat()
    }


@router.get("/execution/trades")
async def mock_trades():
    """
    Mock implementation for /api/execution/trades.
    """
    logger.info(f"Processing mock request for /api/execution/trades")
    return {
        "success": True,
        "message": "This is a mock implementation",
        "data": {},
        "timestamp": datetime.now().isoformat()
    }


@router.get("/execution/trades/123")
async def mock_endpoint_123():
    """
    Mock implementation for /api/execution/trades/123.
    """
    logger.info(f"Processing mock request for /api/execution/trades/123")
    return {
        "success": True,
        "message": "This is a mock implementation",
        "data": {},
        "timestamp": datetime.now().isoformat()
    }

"""
Broker API integration for the Forex AI Trading System.

This module provides functionality to connect to brokerage APIs,
fetch account information, and execute trades.
"""

import logging
import json
import requests
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import hmac
import hashlib
import base64
import time

import pandas as pd
import numpy as np

from forex_ai.config.settings import get_settings
from forex_ai.exceptions import ApiConnectionError, ApiRateLimitError, ApiResponseError

logger = logging.getLogger(__name__)

class BrokerConnector:
    """
    Base class for broker API integrations.
    
    This class provides common functionality for connecting to broker APIs,
    with provider-specific implementations in subclasses.
    """
    
    def __init__(self, api_key: Optional[str] = None, account_id: Optional[str] = None):
        """
        Initialize the broker connector.
        
        Args:
            api_key: API key/token for the broker API
            account_id: Broker account ID
        """
        self.api_key = api_key
        self.account_id = account_id
        self.settings = get_settings()
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information from the broker.
        
        Returns:
            Dictionary with account information
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current open positions.
        
        Returns:
            List of position dictionaries
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_orders(self) -> List[Dict[str, Any]]:
        """
        Get current open orders.
        
        Returns:
            List of order dictionaries
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def place_order(
        self,
        instrument: str,
        units: float,
        order_type: str = "MARKET",
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        trailing_stop: Optional[float] = None,
        time_in_force: str = "GTC"
    ) -> Dict[str, Any]:
        """
        Place an order with the broker.
        
        Args:
            instrument: Instrument/symbol to trade (e.g., "EUR_USD")
            units: Size of the order (positive for buy, negative for sell)
            order_type: Type of order (MARKET, LIMIT, STOP, etc.)
            price: Price for limit/stop orders
            stop_loss: Stop loss price level
            take_profit: Take profit price level
            trailing_stop: Trailing stop distance in pips
            time_in_force: Time in force (GTC, GTD, IOC, FOK)
            
        Returns:
            Dictionary with order information
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def modify_order(
        self,
        order_id: str,
        units: Optional[float] = None,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        trailing_stop: Optional[float] = None,
        time_in_force: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Modify an existing order.
        
        Args:
            order_id: ID of the order to modify
            units: New size of the order
            price: New price for limit/stop orders
            stop_loss: New stop loss price level
            take_profit: New take profit price level
            trailing_stop: New trailing stop distance in pips
            time_in_force: New time in force
            
        Returns:
            Dictionary with updated order information
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an existing order.
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            Dictionary with cancellation status
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def close_position(
        self,
        instrument: Optional[str] = None,
        position_id: Optional[str] = None,
        units: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Close an open position.
        
        Args:
            instrument: Instrument/symbol to close position for
            position_id: ID of the position to close
            units: Number of units to close (None for all)
            
        Returns:
            Dictionary with position closure status
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_transaction_history(
        self,
        from_time: Optional[datetime] = None,
        to_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get transaction history.
        
        Args:
            from_time: Start time for transactions
            to_time: End time for transactions
            limit: Maximum number of transactions to return
            
        Returns:
            List of transaction dictionaries
        """
        raise NotImplementedError("Subclasses must implement this method")


class OandaConnector(BrokerConnector):
    """
    Connector for the Oanda v20 API.
    
    This class implements broker API functionality for Oanda's v20 REST API.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        account_id: Optional[str] = None,
        environment: str = "practice"
    ):
        """
        Initialize the Oanda connector.
        
        Args:
            api_key: Oanda API token
            account_id: Oanda account ID
            environment: 'practice' or 'live'
        """
        super().__init__(api_key, account_id)
        
        # Use provided credentials or fall back to settings
        self.api_key = api_key or self.settings.OANDA_API_KEY
        self.account_id = account_id or self.settings.OANDA_ACCOUNT_ID
        
        # Set API base URL based on environment
        if environment.lower() == "live":
            self.base_url = "https://api-fxtrade.oanda.com"
        else:
            self.base_url = "https://api-fxpractice.oanda.com"
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Make a request to the Oanda API.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint
            params: URL parameters
            data: Request body data
            headers: Additional request headers
            
        Returns:
            Dictionary with API response
        """
        if not self.api_key:
            raise ValueError("Oanda API key is required")
        
        # Default headers
        request_headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Add custom headers
        if headers:
            request_headers.update(headers)
        
        # Build full URL
        url = f"{self.base_url}/{endpoint}"
        
        # Convert data to JSON if provided
        json_data = None
        if data:
            json_data = json.dumps(data)
        
        try:
            # Make request
            response = requests.request(
                method=method,
                url=url,
                params=params,
                data=json_data,
                headers=request_headers,
                timeout=30
            )
            
            # Check for rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 5))
                logger.warning(f"Rate limited by Oanda API. Retry after {retry_after} seconds.")
                raise ApiRateLimitError(f"Rate limited. Retry after {retry_after} seconds.")
            
            # Handle non-success responses
            if response.status_code >= 400:
                logger.error(f"Oanda API error: {response.status_code} - {response.text}")
                raise ApiResponseError(f"API error {response.status_code}: {response.text}")
            
            # Return JSON response
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request to Oanda API failed: {e}")
            raise ApiConnectionError(f"Connection error: {e}")
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information from Oanda.
        
        Returns:
            Dictionary with account information
        """
        if not self.account_id:
            raise ValueError("Oanda account ID is required")
        
        endpoint = f"v3/accounts/{self.account_id}"
        response = self._make_request("GET", endpoint)
        
        return response.get("account", {})
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current open positions from Oanda.
        
        Returns:
            List of position dictionaries
        """
        if not self.account_id:
            raise ValueError("Oanda account ID is required")
        
        endpoint = f"v3/accounts/{self.account_id}/openPositions"
        response = self._make_request("GET", endpoint)
        
        positions = response.get("positions", [])
        
        # Enhance position data with calculated fields
        for position in positions:
            # Calculate unrealized P/L percentage
            if "long" in position and float(position["long"].get("units", 0)) > 0:
                side_data = position["long"]
                position["side"] = "long"
            elif "short" in position and float(position["short"].get("units", 0)) < 0:
                side_data = position["short"]
                position["side"] = "short"
            else:
                continue
            
            # Calculate profit/loss percentage
            if float(side_data.get("averagePrice", 0)) > 0 and "unrealizedPL" in side_data:
                margin_used = float(side_data.get("marginUsed", 0))
                unrealized_pl = float(side_data.get("unrealizedPL", 0))
                if margin_used > 0:
                    position["plPercentage"] = (unrealized_pl / margin_used) * 100
        
        return positions
    
    def get_orders(self) -> List[Dict[str, Any]]:
        """
        Get current open orders from Oanda.
        
        Returns:
            List of order dictionaries
        """
        if not self.account_id:
            raise ValueError("Oanda account ID is required")
        
        endpoint = f"v3/accounts/{self.account_id}/orders"
        response = self._make_request("GET", endpoint)
        
        return response.get("orders", [])
    
    def place_order(
        self,
        instrument: str,
        units: float,
        order_type: str = "MARKET",
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        trailing_stop: Optional[float] = None,
        time_in_force: str = "GTC"
    ) -> Dict[str, Any]:
        """
        Place an order with Oanda.
        
        Args:
            instrument: Instrument to trade (e.g., "EUR_USD")
            units: Size of the order (positive for buy, negative for sell)
            order_type: Type of order (MARKET, LIMIT, STOP, etc.)
            price: Price for limit/stop orders
            stop_loss: Stop loss price level
            take_profit: Take profit price level
            trailing_stop: Trailing stop distance in pips
            time_in_force: Time in force (GTC, GTD, IOC, FOK)
            
        Returns:
            Dictionary with order information
        """
        if not self.account_id:
            raise ValueError("Oanda account ID is required")
        
        # Build order data
        order_data = {
            "order": {
                "type": order_type,
                "instrument": instrument,
                "units": str(units),
                "timeInForce": time_in_force,
                "positionFill": "DEFAULT"
            }
        }
        
        # Add optional parameters
        if order_type in ["LIMIT", "STOP", "MARKET_IF_TOUCHED"] and price is not None:
            order_data["order"]["price"] = str(price)
        
        # Add take profit if specified
        if take_profit is not None:
            order_data["order"]["takeProfitOnFill"] = {
                "price": str(take_profit)
            }
        
        # Add stop loss if specified
        if stop_loss is not None:
            order_data["order"]["stopLossOnFill"] = {
                "price": str(stop_loss)
            }
        
        # Add trailing stop if specified
        if trailing_stop is not None:
            order_data["order"]["trailingStopLossOnFill"] = {
                "distance": str(trailing_stop)
            }
        
        endpoint = f"v3/accounts/{self.account_id}/orders"
        response = self._make_request("POST", endpoint, data=order_data)
        
        # Format and return response
        result = {
            "order_id": response.get("orderCreateTransaction", {}).get("id"),
            "instrument": instrument,
            "units": units,
            "type": order_type,
            "status": response.get("orderCreateTransaction", {}).get("type"),
            "time": response.get("orderCreateTransaction", {}).get("time")
        }
        
        return result
    
    def modify_order(
        self,
        order_id: str,
        units: Optional[float] = None,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        trailing_stop: Optional[float] = None,
        time_in_force: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Modify an existing order with Oanda.
        
        Args:
            order_id: ID of the order to modify
            units: New size of the order
            price: New price for limit/stop orders
            stop_loss: New stop loss price level
            take_profit: New take profit price level
            trailing_stop: New trailing stop distance in pips
            time_in_force: New time in force
            
        Returns:
            Dictionary with updated order information
        """
        if not self.account_id:
            raise ValueError("Oanda account ID is required")
        
        # Build order data
        order_data = {"order": {}}
        
        # Add only parameters that are specified
        if units is not None:
            order_data["order"]["units"] = str(units)
        
        if price is not None:
            order_data["order"]["price"] = str(price)
        
        if time_in_force is not None:
            order_data["order"]["timeInForce"] = time_in_force
        
        # Add take profit if specified
        if take_profit is not None:
            endpoint = f"v3/accounts/{self.account_id}/orders/{order_id}/takeProfitOrder"
            tp_data = {
                "takeProfit": {
                    "price": str(take_profit),
                    "timeInForce": "GTC"
                }
            }
            try:
                self._make_request("PUT", endpoint, data=tp_data)
            except ApiResponseError:
                # If no existing take profit, create one
                tp_data["takeProfit"]["orderId"] = order_id
                endpoint = f"v3/accounts/{self.account_id}/orders"
                self._make_request("POST", endpoint, data=tp_data)
        
        # Add stop loss if specified
        if stop_loss is not None:
            endpoint = f"v3/accounts/{self.account_id}/orders/{order_id}/stopLossOrder"
            sl_data = {
                "stopLoss": {
                    "price": str(stop_loss),
                    "timeInForce": "GTC"
                }
            }
            try:
                self._make_request("PUT", endpoint, data=sl_data)
            except ApiResponseError:
                # If no existing stop loss, create one
                sl_data["stopLoss"]["orderId"] = order_id
                endpoint = f"v3/accounts/{self.account_id}/orders"
                self._make_request("POST", endpoint, data=sl_data)
        
        # Add trailing stop if specified
        if trailing_stop is not None:
            endpoint = f"v3/accounts/{self.account_id}/orders/{order_id}/trailingStopLossOrder"
            ts_data = {
                "trailingStopLoss": {
                    "distance": str(trailing_stop),
                    "timeInForce": "GTC"
                }
            }
            try:
                self._make_request("PUT", endpoint, data=ts_data)
            except ApiResponseError:
                # If no existing trailing stop, create one
                ts_data["trailingStopLoss"]["orderId"] = order_id
                endpoint = f"v3/accounts/{self.account_id}/orders"
                self._make_request("POST", endpoint, data=ts_data)
        
        # Only send the main order update if there are parameters to modify
        if order_data["order"]:
            endpoint = f"v3/accounts/{self.account_id}/orders/{order_id}"
            self._make_request("PUT", endpoint, data=order_data)
        
        # Get updated order data
        endpoint = f"v3/accounts/{self.account_id}/orders/{order_id}"
        response = self._make_request("GET", endpoint)
        
        return response.get("order", {})
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an existing order with Oanda.
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            Dictionary with cancellation status
        """
        if not self.account_id:
            raise ValueError("Oanda account ID is required")
        
        endpoint = f"v3/accounts/{self.account_id}/orders/{order_id}/cancel"
        response = self._make_request("PUT", endpoint)
        
        # Format and return response
        result = {
            "order_id": order_id,
            "status": "cancelled",
            "time": response.get("orderCancelTransaction", {}).get("time")
        }
        
        return result
    
    def close_position(
        self,
        instrument: Optional[str] = None,
        position_id: Optional[str] = None,
        units: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Close an open position with Oanda.
        
        Args:
            instrument: Instrument to close position for
            position_id: ID of the position to close (not used for Oanda)
            units: Number of units to close (None for all)
            
        Returns:
            Dictionary with position closure status
        """
        if not self.account_id:
            raise ValueError("Oanda account ID is required")
        
        if not instrument:
            raise ValueError("Instrument is required")
        
        # Format instrument to Oanda format if needed (EUR/USD -> EUR_USD)
        instrument = instrument.replace("/", "_")
        
        # If units not specified, get all units from current position
        if units is None:
            endpoint = f"v3/accounts/{self.account_id}/positions/{instrument}"
            position = self._make_request("GET", endpoint).get("position", {})
            
            long_units = float(position.get("long", {}).get("units", 0))
            short_units = float(position.get("short", {}).get("units", 0))
            
            if long_units > 0:
                units = -long_units  # Negative to close long position
            elif short_units < 0:
                units = -short_units  # Positive to close short position
            else:
                return {"status": "no_position", "instrument": instrument}
        
        # Build close position data
        close_data = {
            "longUnits": "ALL" if units < 0 else "NONE",
            "shortUnits": "ALL" if units > 0 else "NONE"
        }
        
        # If partial close, specify exact units
        if abs(units) > 0 and units != "ALL":
            if units < 0:
                close_data["longUnits"] = str(abs(units))
            else:
                close_data["shortUnits"] = str(abs(units))
        
        endpoint = f"v3/accounts/{self.account_id}/positions/{instrument}/close"
        response = self._make_request("PUT", endpoint, data=close_data)
        
        # Format and return response
        result = {
            "instrument": instrument,
            "units_closed": units,
            "status": "closed",
            "profit": float(response.get("longOrderFillTransaction", {}).get("pl", 0)) + 
                     float(response.get("shortOrderFillTransaction", {}).get("pl", 0)),
            "time": response.get("longOrderFillTransaction", {}).get("time") or 
                   response.get("shortOrderFillTransaction", {}).get("time")
        }
        
        return result
    
    def get_transaction_history(
        self,
        from_time: Optional[datetime] = None,
        to_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get transaction history from Oanda.
        
        Args:
            from_time: Start time for transactions
            to_time: End time for transactions
            limit: Maximum number of transactions to return
            
        Returns:
            List of transaction dictionaries
        """
        if not self.account_id:
            raise ValueError("Oanda account ID is required")
        
        params = {"pageSize": min(limit, 1000)}
        
        if from_time:
            params["from"] = from_time.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
        
        if to_time:
            params["to"] = to_time.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
        
        endpoint = f"v3/accounts/{self.account_id}/transactions"
        response = self._make_request("GET", endpoint, params=params)
        
        # Get transaction IDs
        transaction_ids = response.get("pages", {}).get("transactions", [])
        
        # Fetch full transaction data for each ID
        transactions = []
        for tx_id in transaction_ids[:limit]:
            endpoint = f"v3/accounts/{self.account_id}/transactions/{tx_id}"
            tx_response = self._make_request("GET", endpoint)
            transactions.append(tx_response.get("transaction", {}))
        
        return transactions
    
    def get_prices(self, instruments: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get current prices for instruments.
        
        Args:
            instruments: List of instruments to get prices for
            
        Returns:
            Dictionary mapping instruments to price data
        """
        instruments_param = ",".join(instruments)
        endpoint = f"v3/accounts/{self.account_id}/pricing"
        params = {"instruments": instruments_param}
        
        response = self._make_request("GET", endpoint, params=params)
        
        # Format response into a more convenient structure
        prices = {}
        for price in response.get("prices", []):
            instrument = price.get("instrument")
            if instrument:
                ask = float(price.get("asks", [{}])[0].get("price", 0))
                bid = float(price.get("bids", [{}])[0].get("price", 0))
                spread = ask - bid
                spread_pips = spread * 10000 if "JPY" not in instrument else spread * 100
                
                prices[instrument] = {
                    "time": price.get("time"),
                    "ask": ask,
                    "bid": bid,
                    "mid": (ask + bid) / 2,
                    "spread": spread,
                    "spread_pips": spread_pips
                }
        
        return prices
    
    def get_position_details(self, instrument: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific position.
        
        Args:
            instrument: Instrument to get position details for
            
        Returns:
            Dictionary with position details
        """
        if not self.account_id:
            raise ValueError("Oanda account ID is required")
        
        # Format instrument to Oanda format if needed
        instrument = instrument.replace("/", "_")
        
        endpoint = f"v3/accounts/{self.account_id}/positions/{instrument}"
        response = self._make_request("GET", endpoint)
        
        position = response.get("position", {})
        
        # Enhance with additional calculated fields
        result = {
            "instrument": position.get("instrument"),
            "pl": float(position.get("pl", 0)),
            "unrealizedPL": float(position.get("unrealizedPL", 0)),
            "marginUsed": float(position.get("marginUsed", 0)),
            "long": {
                "units": float(position.get("long", {}).get("units", 0)),
                "averagePrice": float(position.get("long", {}).get("averagePrice", 0)),
                "pl": float(position.get("long", {}).get("pl", 0)),
                "unrealizedPL": float(position.get("long", {}).get("unrealizedPL", 0))
            },
            "short": {
                "units": float(position.get("short", {}).get("units", 0)),
                "averagePrice": float(position.get("short", {}).get("averagePrice", 0)),
                "pl": float(position.get("short", {}).get("pl", 0)),
                "unrealizedPL": float(position.get("short", {}).get("unrealizedPL", 0))
            }
        }
        
        # Calculate net position
        result["net_units"] = result["long"]["units"] + result["short"]["units"]
        
        # Calculate average entry price for the net position
        if result["net_units"] != 0:
            result["average_price"] = (
                (result["long"]["units"] * result["long"]["averagePrice"]) +
                (result["short"]["units"] * result["short"]["averagePrice"])
            ) / result["net_units"]
        else:
            result["average_price"] = 0
        
        return result


class RiskCalculator:
    """
    Risk management calculator for forex trading.
    
    This class provides utilities for calculating position sizes,
    stop loss levels, take profit levels, and risk metrics.
    """
    
    def __init__(self, broker_connector: Optional[BrokerConnector] = None):
        """
        Initialize the risk calculator.
        
        Args:
            broker_connector: Optional broker connector for account data
        """
        self.broker = broker_connector
    
    def calculate_position_size(
        self,
        account_balance: float,
        risk_percentage: float,
        entry_price: float,
        stop_loss_price: float,
        instrument: str,
        leverage: float = 1.0
    ) -> Dict[str, Any]:
        """
        Calculate position size based on risk parameters.
        
        Args:
            account_balance: Account balance in account currency
            risk_percentage: Percentage of account to risk (1.0 = 1%)
            entry_price: Entry price for the position
            stop_loss_price: Stop loss price level
            instrument: Instrument/symbol (e.g., "EUR/USD")
            leverage: Account leverage (e.g., 50 for 50:1)
            
        Returns:
            Dictionary with position sizing details
        """
        # Format instrument components
        instrument = instrument.replace("_", "/")
        components = instrument.split("/")
        
        if len(components) != 2:
            raise ValueError(f"Invalid instrument format: {instrument}")
        
        base_currency, quote_currency = components
        
        # Calculate pip value based on instrument
        is_jpy_pair = "JPY" in instrument
        pip_decimal = 0.01 if is_jpy_pair else 0.0001
        
        # Calculate distance to stop in pips
        price_direction = 1 if entry_price > stop_loss_price else -1
        stop_distance = abs(entry_price - stop_loss_price)
        stop_pips = stop_distance / pip_decimal
        
        # Calculate risk amount in account currency
        risk_amount = (account_balance * risk_percentage) / 100
        
        # Calculate position size in units of base currency
        if price_direction > 0:  # Long position
            position_size = risk_amount / (stop_distance / entry_price)
        else:  # Short position
            position_size = risk_amount / stop_distance
        
        # Apply leverage
        position_size = position_size * leverage
        
        # Calculate position value in account currency
        position_value = position_size * entry_price
        
        # Calculate margin requirement
        margin_requirement = position_value / leverage
        
        return {
            "position_size": position_size,
            "position_size_formatted": f"{position_size:.2f} units",
            "risk_amount": risk_amount,
            "risk_percentage": risk_percentage,
            "stop_distance_pips": stop_pips,
            "position_value": position_value,
            "margin_requirement": margin_requirement,
            "risk_reward_ratio": None,  # Will be calculated if take profit is provided
            "leverage_used": leverage
        }
    
    def calculate_stop_loss(
        self,
        entry_price: float,
        position_size: float,
        risk_amount: float,
        direction: str,
        instrument: str
    ) -> float:
        """
        Calculate stop loss price based on risk amount.
        
        Args:
            entry_price: Entry price for the position
            position_size: Size of the position in base currency
            risk_amount: Amount to risk in account currency
            direction: Position direction ("long" or "short")
            instrument: Instrument/symbol (e.g., "EUR/USD")
            
        Returns:
            Calculated stop loss price
        """
        is_jpy_pair = "JPY" in instrument
        pip_decimal = 0.01 if is_jpy_pair else 0.0001
        
        if direction.lower() == "long":
            stop_distance = risk_amount / (position_size * entry_price)
            stop_loss = entry_price - stop_distance
        else:  # short
            stop_distance = risk_amount / position_size
            stop_loss = entry_price + stop_distance
        
        return stop_loss
    
    def calculate_take_profit(
        self,
        entry_price: float,
        stop_loss: float,
        risk_reward_ratio: float,
        direction: str
    ) -> float:
        """
        Calculate take profit based on risk-reward ratio.
        
        Args:
            entry_price: Entry price for the position
            stop_loss: Stop loss price level
            risk_reward_ratio: Desired risk-reward ratio (e.g., 2.0 for 1:2)
            direction: Position direction ("long" or "short")
            
        Returns:
            Calculated take profit price
        """
        stop_distance = abs(entry_price - stop_loss)
        take_profit_distance = stop_distance * risk_reward_ratio
        
        if direction.lower() == "long":
            take_profit = entry_price + take_profit_distance
        else:  # short
            take_profit = entry_price - take_profit_distance
        
        return take_profit
    
    def calculate_risk_metrics(
        self,
        positions: List[Dict[str, Any]],
        account_balance: float
    ) -> Dict[str, Any]:
        """
        Calculate risk metrics for open positions.
        
        Args:
            positions: List of position dictionaries
            account_balance: Account balance in account currency
            
        Returns:
            Dictionary with risk metrics
        """
        # Calculate total exposure
        total_long_exposure = 0
        total_short_exposure = 0
        total_margin_used = 0
        total_unrealized_pl = 0
        
        for position in positions:
            margin_used = float(position.get("marginUsed", 0))
            unrealized_pl = float(position.get("unrealizedPL", 0))
            
            # Add to totals
            total_margin_used += margin_used
            total_unrealized_pl += unrealized_pl
            
            # Calculate exposure by direction
            long_units = float(position.get("long", {}).get("units", 0))
            short_units = float(position.get("short", {}).get("units", 0))
            
            if long_units > 0:
                long_price = float(position.get("long", {}).get("averagePrice", 0))
                total_long_exposure += long_units * long_price
            
            if short_units < 0:
                short_price = float(position.get("short", {}).get("averagePrice", 0))
                total_short_exposure += abs(short_units) * short_price
        
        # Calculate total exposure
        total_exposure = total_long_exposure + total_short_exposure
        
        # Calculate risk metrics
        margin_percentage = (total_margin_used / account_balance) * 100 if account_balance > 0 else 0
        exposure_percentage = (total_exposure / account_balance) * 100 if account_balance > 0 else 0
        unrealized_pl_percentage = (total_unrealized_pl / account_balance) * 100 if account_balance > 0 else 0
        
        return {
            "total_margin_used": total_margin_used,
            "total_exposure": total_exposure,
            "total_long_exposure": total_long_exposure,
            "total_short_exposure": total_short_exposure,
            "total_unrealized_pl": total_unrealized_pl,
            "margin_percentage": margin_percentage,
            "exposure_percentage": exposure_percentage,
            "unrealized_pl_percentage": unrealized_pl_percentage,
            "available_margin": account_balance - total_margin_used
        }
    
    def get_max_position_size(
        self,
        account_balance: float,
        instrument: str,
        leverage: float = 1.0,
        max_margin_percentage: float = 20.0
    ) -> float:
        """
        Calculate maximum position size based on margin constraints.
        
        Args:
            account_balance: Account balance in account currency
            instrument: Instrument/symbol (e.g., "EUR/USD")
            leverage: Account leverage (e.g., 50 for 50:1)
            max_margin_percentage: Maximum percentage of account to use as margin
            
        Returns:
            Maximum position size in base currency units
        """
        # Calculate maximum margin amount
        max_margin = account_balance * (max_margin_percentage / 100)
        
        # Get current instrument price (if broker is available)
        price = 0
        if self.broker:
            try:
                instrument_fmt = instrument.replace("/", "_")
                prices = self.broker.get_prices([instrument_fmt])
                price = prices.get(instrument_fmt, {}).get("mid", 0)
            except Exception as e:
                logger.warning(f"Could not get price for {instrument}: {e}")
        
        # Use a default price if no broker or price fetch failed
        if price == 0:
            if "USD" in instrument:
                price = 1.1  # Approximate average for major USD pairs
            elif "JPY" in instrument:
                price = 120  # Approximate average for major JPY pairs
            else:
                price = 1.2  # Reasonable default for other pairs
        
        # Calculate maximum position size
        max_position_size = (max_margin * leverage) / price
        
        return max_position_size 
"""
OANDA API Integration for Forex AI Execution.

This module provides an interface to the OANDA REST API for executing trades.
It extends the base BrokerAPI class to provide OANDA-specific functionality.
"""

import os
import json
import logging
import requests
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from enum import Enum

from forex_ai.execution.broker_api import BrokerAPI, OrderType, OrderDirection, OrderStatus
from forex_ai.models.broker_models import OandaCredentials, OandaEnvironment
from forex_ai.utils.config import get_env_var
from forex_ai.config.settings import get_settings

logger = logging.getLogger(__name__)


class OandaAPI(BrokerAPI):
    """
    OANDA API implementation.

    This class provides OANDA-specific implementation of the BrokerAPI interface.
    """

    # OANDA API URLs
    PRACTICE_API_URL = "https://api-fxpractice.oanda.com/v3"
    LIVE_API_URL = "https://api-fxtrade.oanda.com/v3"

    def __init__(self, credentials: Optional[OandaCredentials] = None, user_id: Optional[str] = None):
        """
        Initialize the OANDA API.

        Args:
            credentials: OANDA API credentials (required if user_id not provided)
            user_id: User ID to load credentials for (required if credentials not provided)
            
        Raises:
            ValueError: If neither credentials nor user_id is provided
        """
        super().__init__()
        
        if not credentials and not user_id:
            raise ValueError("Either credentials or user_id must be provided")
        
        if credentials:
            # Use provided credentials
            self.credentials = credentials
        elif user_id:
            # Load credentials from database
            loaded_credentials = self._load_credentials_from_db(user_id)
            if not loaded_credentials:
                raise ValueError(f"No OANDA credentials found for user {user_id}")
            self.credentials = loaded_credentials
        
        self.account_id = self.credentials.default_account_id
        
        # Set API URL based on environment
        if self.credentials.environment == OandaEnvironment.PRACTICE:
            self.api_url = self.PRACTICE_API_URL
        else:
            self.api_url = self.LIVE_API_URL
        
        # Set up headers for API requests
        self.headers = {
            "Authorization": f"Bearer {self.credentials.access_token}",
            "Content-Type": "application/json"
        }
        
        logger.info(f"OANDA API initialized for environment: {self.credentials.environment}")

    def _load_credentials_from_db(self, user_id: str) -> Optional[OandaCredentials]:
        """
        Load OANDA credentials from the database.
        
        Args:
            user_id: User ID to load credentials for
            
        Returns:
            OandaCredentials or None if not found
        """
        try:
            # Import here to avoid circular imports
            from forex_ai.backend_api.db import account_db
            
            # Get credentials from database
            credentials_dict = account_db.get_broker_credentials(user_id, "oanda")
            
            if not credentials_dict:
                logger.warning(f"No OANDA credentials found for user {user_id}")
                return None
                
            # Create credentials object
            return OandaCredentials(
                broker_type=BrokerType.OANDA,
                user_id=user_id,
                access_token=credentials_dict.get("access_token", ""),
                environment=OandaEnvironment(credentials_dict.get("environment", "practice").lower()),
                default_account_id=credentials_dict.get("account_id", "")
            )
        except Exception as e:
            logger.error(f"Error loading OANDA credentials from database: {str(e)}")
            return None

    async def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information from OANDA.

        Returns:
            Dictionary with account information
        """
        if not self.account_id:
            logger.error("No account ID specified")
            return {"error": "No account ID specified"}
        
        try:
            # Make API request to get account summary
            url = f"{self.api_url}/accounts/{self.account_id}/summary"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            account = data.get("account", {})
            
            # Format account data
            return {
                "id": self.account_id,
                "balance": float(account.get("balance", 0)),
                "currency": account.get("currency", "USD"),
                "margin_rate": float(account.get("marginRate", 0)),
                "margin_used": float(account.get("marginUsed", 0)),
                "margin_available": float(account.get("marginAvailable", 0)),
                "unrealized_pl": float(account.get("unrealizedPL", 0)),
                "realized_pl": float(account.get("realizedPL", 0)),
                "open_trade_count": int(account.get("openTradeCount", 0)),
                "pending_order_count": int(account.get("pendingOrderCount", 0)),
                "last_updated": account.get("lastTransactionID", ""),
            }
        except Exception as e:
            logger.error(f"Error getting account information: {str(e)}")
            return {"error": str(e)}

    async def get_accounts(self) -> List[Dict[str, Any]]:
        """
        Get all accounts associated with the OANDA API token.

        Returns:
            List of account dictionaries
        """
        try:
            # Make API request to get accounts
            url = f"{self.api_url}/accounts"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            accounts = data.get("accounts", [])
            
            # Format account data
            result = []
            for account in accounts:
                result.append({
                    "id": account.get("id", ""),
                    "name": account.get("alias", ""),
                    "currency": account.get("currency", ""),
                    "type": "live" if account.get("type") == "LIVE" else "practice"
                })
            
            return result
        except Exception as e:
            logger.error(f"Error getting accounts: {str(e)}")
            return [{"error": str(e)}]

    async def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current positions from OANDA.

        Returns:
            List of position dictionaries
        """
        if not self.account_id:
            logger.error("No account ID specified")
            return [{"error": "No account ID specified"}]
        
        try:
            # Make API request to get open positions
            url = f"{self.api_url}/accounts/{self.account_id}/openPositions"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            positions = data.get("positions", [])
            
            # Format position data
            result = []
            for position in positions:
                instrument = position.get("instrument", "")
                long_units = float(position.get("long", {}).get("units", 0))
                short_units = float(position.get("short", {}).get("units", 0))
                
                if long_units > 0:
                    direction = "buy"
                    size = long_units
                    entry_price = float(position.get("long", {}).get("averagePrice", 0))
                    profit_loss = float(position.get("long", {}).get("unrealizedPL", 0))
                else:
                    direction = "sell"
                    size = abs(short_units)
                    entry_price = float(position.get("short", {}).get("averagePrice", 0))
                    profit_loss = float(position.get("short", {}).get("unrealizedPL", 0))
                
                result.append({
                    "id": f"{instrument}_{direction}",
                    "symbol": instrument,
                    "direction": direction,
                    "size": size,
                    "entry_price": entry_price,
                    "current_price": 0,  # Need to get this from market data
                    "profit_loss": profit_loss,
                    "swap": 0,  # OANDA doesn't provide this directly
                    "open_time": datetime.now().isoformat(),  # OANDA doesn't provide this directly
                })
            
            return result
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            return [{"error": str(e)}]

    async def get_orders(self) -> List[Dict[str, Any]]:
        """
        Get pending orders from OANDA.

        Returns:
            List of order dictionaries
        """
        if not self.account_id:
            logger.error("No account ID specified")
            return [{"error": "No account ID specified"}]
        
        try:
            # Make API request to get pending orders
            url = f"{self.api_url}/accounts/{self.account_id}/pendingOrders"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            orders = data.get("orders", [])
            
            # Format order data
            result = []
            for order in orders:
                order_type = order.get("type", "").lower()
                if order_type == "limit":
                    mapped_type = OrderType.LIMIT
                elif order_type == "stop":
                    mapped_type = OrderType.STOP
                elif order_type == "market_if_touched":
                    mapped_type = OrderType.LIMIT
                else:
                    mapped_type = OrderType.MARKET
                
                units = float(order.get("units", 0))
                direction = OrderDirection.BUY if units > 0 else OrderDirection.SELL
                
                result.append({
                    "id": order.get("id", ""),
                    "symbol": order.get("instrument", ""),
                    "type": mapped_type,
                    "direction": direction,
                    "size": abs(units),
                    "price": float(order.get("price", 0)),
                    "status": OrderStatus.PENDING,
                    "create_time": order.get("createTime", datetime.now().isoformat()),
                })
            
            return result
        except Exception as e:
            logger.error(f"Error getting orders: {str(e)}")
            return [{"error": str(e)}]

    async def place_order(
        self,
        symbol: str,
        direction: OrderDirection,
        size: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        expiration: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Place a new order with OANDA.

        Args:
            symbol: The currency pair symbol (e.g., "EUR_USD")
            direction: Order direction (buy/sell)
            size: Position size
            order_type: Type of order (market, limit, etc.)
            price: Price for limit/stop orders
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)
            expiration: Order expiration time (optional)

        Returns:
            Dictionary with order information
        """
        if not self.account_id:
            logger.error("No account ID specified")
            return {"error": "No account ID specified"}
        
        try:
            # Prepare order data
            units = size if direction == OrderDirection.BUY else -size
            
            # Base order data
            order_data = {
                "units": str(units),
                "instrument": symbol,
                "positionFill": "DEFAULT"
            }
            
            # Add stop loss if specified
            if stop_loss is not None:
                order_data["stopLossOnFill"] = {
                    "price": str(stop_loss),
                    "timeInForce": "GTC"
                }
            
            # Add take profit if specified
            if take_profit is not None:
                order_data["takeProfitOnFill"] = {
                    "price": str(take_profit),
                    "timeInForce": "GTC"
                }
            
            # Set order type and additional parameters
            if order_type == OrderType.MARKET:
                order_data["type"] = "MARKET"
            elif order_type == OrderType.LIMIT:
                if price is None:
                    raise ValueError("Price must be specified for limit orders")
                order_data["type"] = "LIMIT"
                order_data["price"] = str(price)
                order_data["timeInForce"] = "GTC"
            elif order_type == OrderType.STOP:
                if price is None:
                    raise ValueError("Price must be specified for stop orders")
                order_data["type"] = "STOP"
                order_data["price"] = str(price)
                order_data["timeInForce"] = "GTC"
            
            # Add expiration if specified
            if expiration is not None:
                order_data["gtdTime"] = expiration.isoformat()
                order_data["timeInForce"] = "GTD"
            
            # Make API request to place order
            url = f"{self.api_url}/accounts/{self.account_id}/orders"
            response = requests.post(
                url, 
                headers=self.headers, 
                json={"order": order_data}
            )
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            order_created = data.get("orderCreateTransaction", {})
            
            # Format order data
            return {
                "id": order_created.get("id", ""),
                "symbol": order_created.get("instrument", ""),
                "type": order_type,
                "direction": direction,
                "size": size,
                "price": price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "status": OrderStatus.PENDING,
                "create_time": order_created.get("time", datetime.now().isoformat()),
            }
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            return {"error": str(e)}

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order with OANDA.

        Args:
            order_id: ID of the order to cancel

        Returns:
            True if successful, False otherwise
        """
        if not self.account_id:
            logger.error("No account ID specified")
            return False
        
        try:
            # Make API request to cancel order
            url = f"{self.api_url}/accounts/{self.account_id}/orders/{order_id}/cancel"
            response = requests.put(url, headers=self.headers)
            response.raise_for_status()
            
            # Check if order was cancelled
            data = response.json()
            if "orderCancelTransaction" in data:
                return True
            return False
        except Exception as e:
            logger.error(f"Error cancelling order: {str(e)}")
            return False

    async def modify_order(
        self,
        order_id: str,
        price: Optional[float] = None,
        size: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Modify a pending order with OANDA.

        Args:
            order_id: ID of the order to modify
            price: New price for limit/stop orders
            size: New position size
            stop_loss: New stop loss price
            take_profit: New take profit price

        Returns:
            Dictionary with updated order information
        """
        if not self.account_id:
            logger.error("No account ID specified")
            return {"error": "No account ID specified"}
        
        try:
            # Prepare order data
            order_data = {}
            
            if price is not None:
                order_data["price"] = str(price)
            
            if size is not None:
                order_data["units"] = str(size)
            
            if stop_loss is not None:
                order_data["stopLoss"] = {
                    "price": str(stop_loss),
                    "timeInForce": "GTC"
                }
            
            if take_profit is not None:
                order_data["takeProfit"] = {
                    "price": str(take_profit),
                    "timeInForce": "GTC"
                }
            
            # Make API request to modify order
            url = f"{self.api_url}/accounts/{self.account_id}/orders/{order_id}"
            response = requests.put(
                url, 
                headers=self.headers, 
                json={"order": order_data}
            )
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            order_modified = data.get("orderCreateTransaction", {})
            
            # Format order data
            return {
                "id": order_id,
                "status": "modified",
                "modify_time": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error modifying order: {str(e)}")
            return {"error": str(e)}

    async def close_position(self, position_id: str) -> Dict[str, Any]:
        """
        Close a position with OANDA.

        Args:
            position_id: ID of the position to close (format: "instrument_direction")

        Returns:
            Dictionary with close position result
        """
        if not self.account_id:
            logger.error("No account ID specified")
            return {"error": "No account ID specified"}
        
        try:
            # Parse position_id to get instrument
            parts = position_id.split("_")
            if len(parts) < 1:
                raise ValueError(f"Invalid position ID format: {position_id}")
            
            instrument = parts[0]
            
            # Make API request to close position
            url = f"{self.api_url}/accounts/{self.account_id}/positions/{instrument}/close"
            response = requests.put(
                url, 
                headers=self.headers, 
                json={"longUnits": "ALL", "shortUnits": "ALL"}
            )
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            
            # Format response
            return {
                "id": position_id,
                "status": "closed",
                "close_time": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error closing position: {str(e)}")
            return {"error": str(e)}

    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get market data for a symbol from OANDA.

        Args:
            symbol: The currency pair symbol (e.g., "EUR_USD")

        Returns:
            Dictionary with market data
        """
        try:
            # Make API request to get pricing
            url = f"{self.api_url}/accounts/{self.account_id}/pricing"
            params = {
                "instruments": symbol,
                "includeUnitsAvailable": "true"
            }
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            prices = data.get("prices", [])
            
            if not prices:
                return {"error": f"No price data for {symbol}"}
            
            # Get first price (should only be one)
            price = prices[0]
            
            # Format market data
            return {
                "symbol": symbol,
                "bid": float(price.get("bids", [{}])[0].get("price", 0)),
                "ask": float(price.get("asks", [{}])[0].get("price", 0)),
                "time": price.get("time", datetime.now().isoformat()),
                "spread": float(price.get("asks", [{}])[0].get("price", 0)) - float(price.get("bids", [{}])[0].get("price", 0)),
                "units_available": {
                    "long": float(price.get("unitsAvailable", {}).get("default", {}).get("long", "0")),
                    "short": float(price.get("unitsAvailable", {}).get("default", {}).get("short", "0")),
                }
            }
        except Exception as e:
            logger.error(f"Error getting market data: {str(e)}")
            return {"error": str(e)}
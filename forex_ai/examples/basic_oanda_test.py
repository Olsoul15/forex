"""
Basic OANDA API test script.

This script is for DEVELOPMENT USE ONLY. It tests the connection to OANDA's API 
and retrieves current price data using their REST API instead of WebSocket.

Required environment variables:
- OANDA_ACCESS_TOKEN or OANDA_API_KEY
- OANDA_ACCOUNT_ID

WARNING: This script should never be used in production. In production,
credentials should always come from the database/API.
"""

import os
import sys
import requests
import json
import logging
from datetime import datetime
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from forex_ai.utils.logging import setup_logging, get_logger

# Set up logging
setup_logging()
logger = get_logger(__name__)

# Instruments to get prices for
INSTRUMENTS = ["EUR_USD", "GBP_USD", "USD_JPY"]

# OANDA API endpoints
PRICES_ENDPOINT = "https://api-fxpractice.oanda.com/v3/accounts/{}/pricing"
INSTRUMENTS_ENDPOINT = "https://api-fxpractice.oanda.com/v3/accounts/{}/instruments"


def get_oanda_credentials():
    """
    Get OANDA credentials from environment variables.
    
    Returns:
        tuple: (access_token, account_id) or (None, None) if not found
        
    Raises:
        SystemExit: If required environment variables are not set
    """
    logger.warning("DEVELOPMENT MODE: Using environment variables for OANDA credentials")
    
    # Check for required environment variables
    access_token = os.environ.get("OANDA_ACCESS_TOKEN") or os.environ.get("OANDA_API_KEY")
    account_id = os.environ.get("OANDA_ACCOUNT_ID")
    
    if not access_token:
        logger.error("Neither OANDA_ACCESS_TOKEN nor OANDA_API_KEY environment variable is set")
        print("Error: OANDA_ACCESS_TOKEN or OANDA_API_KEY environment variable must be set")
        return None, None
        
    if not account_id:
        logger.error("OANDA_ACCOUNT_ID environment variable not set")
        print("Error: OANDA_ACCOUNT_ID environment variable must be set")
        return None, None
        
    return access_token, account_id


def get_current_prices(access_token, account_id):
    """
    Get current prices from OANDA's REST API.
    
    Args:
        access_token: OANDA API access token
        account_id: OANDA account ID
        
    Returns:
        dict: Price data or None if request failed
    """
    # Prepare API request
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    instruments_param = ",".join(INSTRUMENTS)
    url = PRICES_ENDPOINT.format(account_id)
    params = {"instruments": instruments_param}

    try:
        print(f"Requesting prices for {instruments_param}...")
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  # Raise exception for HTTP errors

        # Parse response
        data = response.json()

        if "prices" in data:
            print("\nCurrent Prices:")
            print("==============")

            for price in data["prices"]:
                instrument = price.get("instrument", "")
                bid = price.get("bids", [{}])[0].get("price", "N/A")
                ask = price.get("asks", [{}])[0].get("price", "N/A")
                time = price.get("time", "")

                print(f"{instrument}: Bid = {bid}, Ask = {ask}, Time = {time}")

                # Calculate simple technical indicators
                if "closeoutBid" in price and "closeoutAsk" in price:
                    mid = (
                        float(price["closeoutBid"]) + float(price["closeoutAsk"])
                    ) / 2
                    print(f"  Mid price: {mid:.5f}")

            return data
        else:
            print("Error: No prices found in response")
            print(f"Response: {data}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Error connecting to OANDA API: {str(e)}")
        return None


def get_account_instruments(access_token, account_id):
    """
    Get available instruments for the account.
    
    Args:
        access_token: OANDA API access token
        account_id: OANDA account ID
        
    Returns:
        dict: Instruments data or None if request failed
    """
    # Prepare API request
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    url = INSTRUMENTS_ENDPOINT.format(account_id)

    try:
        print(f"Requesting instruments for account {account_id}...")
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise exception for HTTP errors

        # Parse response
        data = response.json()

        if "instruments" in data:
            print("\nAvailable Instruments:")
            print("====================")

            for instrument in data["instruments"][:10]:  # Show first 10 instruments
                name = instrument.get("name", "")
                type_ = instrument.get("type", "")
                display_name = instrument.get("displayName", "")

                print(f"{name} ({display_name}): {type_}")

            if len(data["instruments"]) > 10:
                print(f"... and {len(data['instruments']) - 10} more instruments")

            return data
        else:
            print("Error: No instruments found in response")
            print(f"Response: {data}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Error connecting to OANDA API: {str(e)}")
        return None


def main():
    """Main function."""
    print("OANDA API Test (DEVELOPMENT USE ONLY)")
    print("====================================")
    
    # Get OANDA credentials
    access_token, account_id = get_oanda_credentials()
    if not access_token or not account_id:
        sys.exit(1)
    
    print(f"Using access token: {access_token[:5]}...{access_token[-5:]}")
    print(f"Using account ID: {account_id}")

    # Get account instruments
    instruments_data = get_account_instruments(access_token, account_id)

    # Get current prices
    prices_data = get_current_prices(access_token, account_id)

    if prices_data:
        print("\nSuccessfully retrieved price data from OANDA API")

        # Calculate simple market analysis
        print("\nSimple Market Analysis:")
        print("=====================")
        for price in prices_data.get("prices", []):
            instrument = price.get("instrument", "")
            if not instrument:
                continue

            bid = float(price.get("bids", [{}])[0].get("price", 0))
            ask = float(price.get("asks", [{}])[0].get("price", 0))

            # Calculate spread
            spread = ask - bid
            spread_pips = spread * 10000 if "_JPY" not in instrument else spread * 100

            print(f"{instrument}:")
            print(f"  Spread: {spread_pips:.1f} pips")

            # Simple trend indicator (just an example, not real technical analysis)
            if "closeoutBid" in price and instrument in ["EUR_USD", "GBP_USD"]:
                current_price = float(price["closeoutBid"])

                # These would normally come from historical data
                # Just using arbitrary values for demonstration
                if instrument == "EUR_USD":
                    ma_20 = 1.0730  # Simulated 20-day moving average
                    ma_50 = 1.0710  # Simulated 50-day moving average
                else:  # GBP_USD
                    ma_20 = 1.2650  # Simulated 20-day moving average
                    ma_50 = 1.2630  # Simulated 50-day moving average

                print(f"  Current: {current_price:.5f}")
                print(f"  MA(20): {ma_20:.5f}")
                print(f"  MA(50): {ma_50:.5f}")

                if current_price > ma_20 and ma_20 > ma_50:
                    print("  Trend: Strong Bullish")
                elif current_price > ma_20:
                    print("  Trend: Bullish")
                elif current_price < ma_20 and ma_20 < ma_50:
                    print("  Trend: Strong Bearish")
                elif current_price < ma_20:
                    print("  Trend: Bearish")
                else:
                    print("  Trend: Neutral")


if __name__ == "__main__":
    main()

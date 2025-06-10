"""
Basic OANDA API test script.

This script tests the connection to OANDA's API and retrieves
current price data using their REST API instead of WebSocket.
"""

import os
import requests
import json
from datetime import datetime

# OANDA API credentials
OANDA_API_KEY = "f1086a5e2c1718a39aa9c8dd0c38f5c9-0329c78b1d1cc274a149ff4151365df0"
OANDA_ACCOUNT_ID = os.environ.get("OANDA_ACCOUNT_ID", "")

# Instruments to get prices for
INSTRUMENTS = ["EUR_USD", "GBP_USD", "USD_JPY"]

# OANDA API endpoints
PRICES_ENDPOINT = "https://api-fxpractice.oanda.com/v3/accounts/{}/pricing"
INSTRUMENTS_ENDPOINT = "https://api-fxpractice.oanda.com/v3/accounts/{}/instruments"


def get_current_prices():
    """
    Get current prices from OANDA's REST API.
    """
    if not OANDA_ACCOUNT_ID:
        print("Error: OANDA_ACCOUNT_ID environment variable not set")
        account_id = input("Enter your OANDA account ID: ")
        if not account_id:
            print("Error: OANDA account ID is required")
            return None
        os.environ["OANDA_ACCOUNT_ID"] = account_id
    else:
        account_id = OANDA_ACCOUNT_ID

    # Prepare API request
    headers = {
        "Authorization": f"Bearer {OANDA_API_KEY}",
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


def get_account_instruments():
    """
    Get available instruments for the account.
    """
    if not OANDA_ACCOUNT_ID:
        print("Error: OANDA_ACCOUNT_ID environment variable not set")
        account_id = input("Enter your OANDA account ID: ")
        if not account_id:
            print("Error: OANDA account ID is required")
            return None
        os.environ["OANDA_ACCOUNT_ID"] = account_id
    else:
        account_id = OANDA_ACCOUNT_ID

    # Prepare API request
    headers = {
        "Authorization": f"Bearer {OANDA_API_KEY}",
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
    print("OANDA API Test")
    print("=============")
    print(f"Using API key: {OANDA_API_KEY[:5]}...{OANDA_API_KEY[-5:]}")

    # Get account instruments
    instruments_data = get_account_instruments()

    # Get current prices
    prices_data = get_current_prices()

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

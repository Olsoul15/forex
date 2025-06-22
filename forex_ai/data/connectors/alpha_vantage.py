"""
Alpha Vantage connector for the Forex AI Trading System.

This module provides functionality to fetch forex and financial data from the Alpha Vantage API.
"""

import logging
import requests
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import time

import pandas as pd

from forex_ai.config.settings import get_settings
from forex_ai.exceptions import DataSourceError, ApiConnectionError, ApiRateLimitError, ApiResponseError
from forex_ai.data.storage.supabase_client import get_supabase_db_client

logger = logging.getLogger(__name__)

class AlphaVantageConnector:
    """
    Connector for Alpha Vantage API.
    
    This connector provides methods to fetch forex and financial data from the Alpha Vantage API.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Alpha Vantage connector.
        
        Args:
            api_key: Alpha Vantage API key. If not provided, it will be read from settings.
        """
        settings = get_settings()
        self.api_key = api_key or settings.ALPHA_VANTAGE_API_KEY
        self.base_url = "https://www.alphavantage.co/query"
        self.db_client = get_supabase_db_client()
        
        if not self.api_key or self.api_key == "placeholder":
            logger.critical("Alpha Vantage API key not provided. API calls will fail.")
            raise ValueError("Alpha Vantage API key is required. Please set ALPHA_VANTAGE_API_KEY in environment variables.")
    
    def get_forex_rate(
        self,
        from_currency: str,
        to_currency: str,
    ) -> Dict[str, Any]:
        """
        Get current forex exchange rate.
        
        Args:
            from_currency: From currency code (e.g., "EUR").
            to_currency: To currency code (e.g., "USD").
            
        Returns:
            Dictionary containing exchange rate information.
            
        Raises:
            DataSourceError: If fetching the exchange rate fails.
            ApiConnectionError: If connection to Alpha Vantage fails.
            ApiRateLimitError: If API rate limit is exceeded.
            ApiResponseError: If response from Alpha Vantage is invalid.
        """
        try:
            # Prepare request parameters
            params = {
                "function": "CURRENCY_EXCHANGE_RATE",
                "from_currency": from_currency,
                "to_currency": to_currency,
                "apikey": self.api_key
            }
            
            # Make request to Alpha Vantage API
            response = requests.get(self.base_url, params=params)
            
            # Check for HTTP errors
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            
            # Check for API errors
            if "Error Message" in data:
                raise ApiResponseError(f"Alpha Vantage API error: {data['Error Message']}")
            
            if "Information" in data and "call frequency" in data["Information"].lower():
                raise ApiRateLimitError("Alpha Vantage API rate limit exceeded")
            
            if "Realtime Currency Exchange Rate" not in data:
                raise ApiResponseError("Invalid response format from Alpha Vantage API")
            
            # Extract and format exchange rate data
            exchange_rate_data = data["Realtime Currency Exchange Rate"]
            result = {
                "from_currency": exchange_rate_data.get("1. From_Currency Code", from_currency),
                "to_currency": exchange_rate_data.get("3. To_Currency Code", to_currency),
                "exchange_rate": float(exchange_rate_data.get("5. Exchange Rate", 0)),
                "last_refreshed": exchange_rate_data.get("6. Last Refreshed", datetime.now().isoformat()),
                "time_zone": exchange_rate_data.get("7. Time Zone", "UTC"),
                "bid_price": float(exchange_rate_data.get("8. Bid Price", 0)) if "8. Bid Price" in exchange_rate_data else None,
                "ask_price": float(exchange_rate_data.get("9. Ask Price", 0)) if "9. Ask Price" in exchange_rate_data else None
            }
            
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to Alpha Vantage API: {str(e)}")
            raise ApiConnectionError(f"Failed to connect to Alpha Vantage API: {str(e)}")
        except ApiRateLimitError:
            logger.error("Alpha Vantage API rate limit exceeded")
            raise
        except ApiResponseError as e:
            logger.error(f"Invalid response from Alpha Vantage API: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error fetching forex rate from Alpha Vantage: {str(e)}")
            raise DataSourceError(f"Failed to fetch forex rate: {str(e)}")
    
    def get_forex_intraday(
        self,
        from_currency: str,
        to_currency: str,
        interval: str = "5min",
        output_size: str = "compact",
    ) -> pd.DataFrame:
        """
        Get intraday forex data.
        
        Args:
            from_currency: From currency code (e.g., "EUR").
            to_currency: To currency code (e.g., "USD").
            interval: Time interval between data points (1min, 5min, 15min, 30min, 60min).
            output_size: Output size (compact = latest 100 data points, full = all data points).
            
        Returns:
            DataFrame containing intraday forex data.
            
        Raises:
            DataSourceError: If fetching the intraday data fails.
            ApiConnectionError: If connection to Alpha Vantage fails.
            ApiRateLimitError: If API rate limit is exceeded.
            ApiResponseError: If response from Alpha Vantage is invalid.
        """
        try:
            # Prepare request parameters
            params = {
                "function": "FX_INTRADAY",
                "from_symbol": from_currency,
                "to_symbol": to_currency,
                "interval": interval,
                "outputsize": output_size,
                "apikey": self.api_key
            }
            
            # Make request to Alpha Vantage API
            response = requests.get(self.base_url, params=params)
            
            # Check for HTTP errors
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            
            # Check for API errors
            if "Error Message" in data:
                raise ApiResponseError(f"Alpha Vantage API error: {data['Error Message']}")
            
            if "Information" in data and "call frequency" in data["Information"].lower():
                raise ApiRateLimitError("Alpha Vantage API rate limit exceeded")
            
            # Extract metadata
            metadata = data.get("Meta Data", {})
            
            # Extract time series data
            time_series_key = f"Time Series FX ({interval})"
            if time_series_key not in data:
                raise ApiResponseError(f"Invalid response format from Alpha Vantage API: missing {time_series_key}")
            
            time_series = data[time_series_key]
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient="index")
            
            # Rename columns
            df.rename(columns={
                "1. open": "open",
                "2. high": "high",
                "3. low": "low",
                "4. close": "close"
            }, inplace=True)
            
            # Convert values to float
            for col in ["open", "high", "low", "close"]:
                df[col] = df[col].astype(float)
            
            # Add metadata columns
            df["from_currency"] = from_currency
            df["to_currency"] = to_currency
            df["interval"] = interval
            
            # Set index to datetime
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to Alpha Vantage API: {str(e)}")
            raise ApiConnectionError(f"Failed to connect to Alpha Vantage API: {str(e)}")
        except ApiRateLimitError:
            logger.error("Alpha Vantage API rate limit exceeded")
            raise
        except ApiResponseError as e:
            logger.error(f"Invalid response from Alpha Vantage API: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error fetching forex intraday data from Alpha Vantage: {str(e)}")
            raise DataSourceError(f"Failed to fetch forex intraday data: {str(e)}")
    
    def get_forex_daily(
        self,
        from_currency: str,
        to_currency: str,
        output_size: str = "compact",
    ) -> pd.DataFrame:
        """
        Get daily forex data.
        
        Args:
            from_currency: From currency code (e.g., "EUR").
            to_currency: To currency code (e.g., "USD").
            output_size: Output size (compact = latest 100 data points, full = all data points).
            
        Returns:
            DataFrame containing daily forex data.
            
        Raises:
            DataSourceError: If fetching the daily data fails.
            ApiConnectionError: If connection to Alpha Vantage fails.
            ApiRateLimitError: If API rate limit is exceeded.
            ApiResponseError: If response from Alpha Vantage is invalid.
        """
        try:
            # Prepare request parameters
            params = {
                "function": "FX_DAILY",
                "from_symbol": from_currency,
                "to_symbol": to_currency,
                "outputsize": output_size,
                "apikey": self.api_key
            }
            
            # Make request to Alpha Vantage API
            response = requests.get(self.base_url, params=params)
            
            # Check for HTTP errors
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            
            # Check for API errors
            if "Error Message" in data:
                raise ApiResponseError(f"Alpha Vantage API error: {data['Error Message']}")
            
            if "Information" in data and "call frequency" in data["Information"].lower():
                raise ApiRateLimitError("Alpha Vantage API rate limit exceeded")
            
            # Extract metadata
            metadata = data.get("Meta Data", {})
            
            # Extract time series data
            time_series_key = "Time Series FX (Daily)"
            if time_series_key not in data:
                raise ApiResponseError(f"Invalid response format from Alpha Vantage API: missing {time_series_key}")
            
            time_series = data[time_series_key]
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient="index")
            
            # Rename columns
            df.rename(columns={
                "1. open": "open",
                "2. high": "high",
                "3. low": "low",
                "4. close": "close"
            }, inplace=True)
            
            # Convert values to float
            for col in ["open", "high", "low", "close"]:
                df[col] = df[col].astype(float)
            
            # Add metadata columns
            df["from_currency"] = from_currency
            df["to_currency"] = to_currency
            
            # Set index to datetime
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to Alpha Vantage API: {str(e)}")
            raise ApiConnectionError(f"Failed to connect to Alpha Vantage API: {str(e)}")
        except ApiRateLimitError:
            logger.error("Alpha Vantage API rate limit exceeded")
            raise
        except ApiResponseError as e:
            logger.error(f"Invalid response from Alpha Vantage API: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error fetching forex daily data from Alpha Vantage: {str(e)}")
            raise DataSourceError(f"Failed to fetch forex daily data: {str(e)}")
    
    def get_forex_weekly(
        self,
        from_currency: str,
        to_currency: str,
    ) -> pd.DataFrame:
        """
        Get weekly forex data.
        
        Args:
            from_currency: From currency code (e.g., "EUR").
            to_currency: To currency code (e.g., "USD").
            
        Returns:
            DataFrame containing weekly forex data.
            
        Raises:
            DataSourceError: If fetching the weekly data fails.
            ApiConnectionError: If connection to Alpha Vantage fails.
            ApiRateLimitError: If API rate limit is exceeded.
            ApiResponseError: If response from Alpha Vantage is invalid.
        """
        try:
            # Prepare request parameters
            params = {
                "function": "FX_WEEKLY",
                "from_symbol": from_currency,
                "to_symbol": to_currency,
                "apikey": self.api_key
            }
            
            # Make request to Alpha Vantage API
            response = requests.get(self.base_url, params=params)
            
            # Check for HTTP errors
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            
            # Check for API errors
            if "Error Message" in data:
                raise ApiResponseError(f"Alpha Vantage API error: {data['Error Message']}")
            
            if "Information" in data and "call frequency" in data["Information"].lower():
                raise ApiRateLimitError("Alpha Vantage API rate limit exceeded")
            
            # Extract metadata
            metadata = data.get("Meta Data", {})
            
            # Extract time series data
            time_series_key = "Time Series FX (Weekly)"
            if time_series_key not in data:
                raise ApiResponseError(f"Invalid response format from Alpha Vantage API: missing {time_series_key}")
            
            time_series = data[time_series_key]
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient="index")
            
            # Rename columns
            df.rename(columns={
                "1. open": "open",
                "2. high": "high",
                "3. low": "low",
                "4. close": "close"
            }, inplace=True)
            
            # Convert values to float
            for col in ["open", "high", "low", "close"]:
                df[col] = df[col].astype(float)
            
            # Add metadata columns
            df["from_currency"] = from_currency
            df["to_currency"] = to_currency
            
            # Set index to datetime
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to Alpha Vantage API: {str(e)}")
            raise ApiConnectionError(f"Failed to connect to Alpha Vantage API: {str(e)}")
        except ApiRateLimitError:
            logger.error("Alpha Vantage API rate limit exceeded")
            raise
        except ApiResponseError as e:
            logger.error(f"Invalid response from Alpha Vantage API: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error fetching forex weekly data from Alpha Vantage: {str(e)}")
            raise DataSourceError(f"Failed to fetch forex weekly data: {str(e)}")
    
    def get_forex_monthly(
        self,
        from_currency: str,
        to_currency: str,
    ) -> pd.DataFrame:
        """
        Get monthly forex data.
        
        Args:
            from_currency: From currency code (e.g., "EUR").
            to_currency: To currency code (e.g., "USD").
            
        Returns:
            DataFrame containing monthly forex data.
            
        Raises:
            DataSourceError: If fetching the monthly data fails.
            ApiConnectionError: If connection to Alpha Vantage fails.
            ApiRateLimitError: If API rate limit is exceeded.
            ApiResponseError: If response from Alpha Vantage is invalid.
        """
        try:
            # Prepare request parameters
            params = {
                "function": "FX_MONTHLY",
                "from_symbol": from_currency,
                "to_symbol": to_currency,
                "apikey": self.api_key
            }
            
            # Make request to Alpha Vantage API
            response = requests.get(self.base_url, params=params)
            
            # Check for HTTP errors
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            
            # Check for API errors
            if "Error Message" in data:
                raise ApiResponseError(f"Alpha Vantage API error: {data['Error Message']}")
            
            if "Information" in data and "call frequency" in data["Information"].lower():
                raise ApiRateLimitError("Alpha Vantage API rate limit exceeded")
            
            # Extract metadata
            metadata = data.get("Meta Data", {})
            
            # Extract time series data
            time_series_key = "Time Series FX (Monthly)"
            if time_series_key not in data:
                raise ApiResponseError(f"Invalid response format from Alpha Vantage API: missing {time_series_key}")
            
            time_series = data[time_series_key]
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient="index")
            
            # Rename columns
            df.rename(columns={
                "1. open": "open",
                "2. high": "high",
                "3. low": "low",
                "4. close": "close"
            }, inplace=True)
            
            # Convert values to float
            for col in ["open", "high", "low", "close"]:
                df[col] = df[col].astype(float)
            
            # Add metadata columns
            df["from_currency"] = from_currency
            df["to_currency"] = to_currency
            
            # Set index to datetime
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to Alpha Vantage API: {str(e)}")
            raise ApiConnectionError(f"Failed to connect to Alpha Vantage API: {str(e)}")
        except ApiRateLimitError:
            logger.error("Alpha Vantage API rate limit exceeded")
            raise
        except ApiResponseError as e:
            logger.error(f"Invalid response from Alpha Vantage API: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error fetching forex monthly data from Alpha Vantage: {str(e)}")
            raise DataSourceError(f"Failed to fetch forex monthly data: {str(e)}")
            
    def save_to_database(self, data: pd.DataFrame, table_name: str) -> int:
        """
        Save data to the database.
        
        Args:
            data: DataFrame to save.
            table_name: Table name to save to.
            
        Returns:
            Number of records saved.
            
        Raises:
            DataSourceError: If saving to the database fails.
        """
        try:
            # Convert DataFrame to list of dictionaries
            records = data.reset_index().to_dict(orient="records")
            
            # Insert records into the database
            result = self.db_client.insert_many(table_name, records)
            
            # Return the number of records saved
            return len(records)
        except Exception as e:
            logger.error(f"Error saving data to database: {str(e)}")
            raise DataSourceError(f"Failed to save data to database: {str(e)}") 
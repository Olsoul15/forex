"""
Market data processing pipeline for the Forex AI Trading System.

This module provides functionality for processing, transforming,
and managing market data from various sources.
"""

import os
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import json
import csv
from io import StringIO

import pandas as pd
import numpy as np
from pandas.tseries.frequencies import to_offset

from forex_ai.config import settings
from forex_ai.data.storage.redis_client import RedisClient
from forex_ai.custom_types import MarketDataPoint, CurrencyPair, TimeFrame
from forex_ai.exceptions import DataProcessingError, DataValidationError
from forex_ai.data.storage.supabase_client import get_supabase_db_client
from forex_ai.data.connectors.alpha_vantage import AlphaVantageConnector

logger = logging.getLogger(__name__)

# Mapping of timeframes to pandas frequencies
TIMEFRAME_TO_FREQ = {
    TimeFrame.M1: "1min",
    TimeFrame.M5: "5min",
    TimeFrame.M15: "15min",
    TimeFrame.M30: "30min",
    TimeFrame.H1: "1H",
    TimeFrame.H4: "4H",
    TimeFrame.D1: "1D",
    TimeFrame.W1: "1W",
    TimeFrame.MN: "1M",
}

def get_currency_pair_id(base: str, quote: str) -> int:
    """
    Get the ID of a currency pair from the database.
    
    Args:
        base: Base currency code (e.g., 'EUR').
        quote: Quote currency code (e.g., 'USD').
        
    Returns:
        The ID of the currency pair.
        
    Raises:
        ValueError: If the currency pair is not found.
    """
    client = get_supabase_db_client()
    pair = client.fetch_one(
        "currency_pairs",
        {"base_currency": base.upper(), "quote_currency": quote.upper()}
    )
    
    if not pair:
        raise ValueError(f"Currency pair {base}/{quote} not found in database")
    
    return pair["id"]

def get_currency_pair_by_id(pair_id: int) -> Tuple[str, str]:
    """
    Get the base and quote currencies for a currency pair ID.
    
    Args:
        pair_id: The ID of the currency pair.
        
    Returns:
        A tuple of (base, quote) currency codes.
        
    Raises:
        ValueError: If the currency pair ID is not found.
    """
    client = get_supabase_db_client()
    pair = client.fetch_one("currency_pairs", {"id": pair_id})
    
    if not pair:
        raise ValueError(f"Currency pair with ID {pair_id} not found in database")
    
    return pair["base_currency"].strip(), pair["quote_currency"].strip()

def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Validate a market data DataFrame.
    
    Args:
        df: The DataFrame to validate.
        
    Returns:
        True if the DataFrame is valid, False otherwise.
        
    Raises:
        DataValidationError: If the DataFrame is invalid.
    """
    # Check for required columns
    required_columns = ["timestamp", "open", "high", "low", "close"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise DataValidationError(f"Missing required columns: {', '.join(missing_columns)}")
    
    # Check data types
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        raise DataValidationError("Column 'timestamp' must be a datetime type")
    
    for col in ["open", "high", "low", "close"]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise DataValidationError(f"Column '{col}' must be a numeric type")
    
    # Validate price relationships
    price_errors = []
    
    for idx, row in df.iterrows():
        if row["high"] < row["low"]:
            price_errors.append(f"Row {idx}: high ({row['high']}) is less than low ({row['low']})")
            
        if row["open"] < row["low"] or row["open"] > row["high"]:
            price_errors.append(f"Row {idx}: open ({row['open']}) is outside high-low range ({row['low']}-{row['high']})")
            
        if row["close"] < row["low"] or row["close"] > row["high"]:
            price_errors.append(f"Row {idx}: close ({row['close']}) is outside high-low range ({row['low']}-{row['high']})")
    
    if price_errors:
        error_msg = "\n".join(price_errors[:10])
        if len(price_errors) > 10:
            error_msg += f"\n... and {len(price_errors) - 10} more errors"
        raise DataValidationError(f"Price relationship validation errors:\n{error_msg}")
    
    # Check for duplicates
    if df.duplicated(subset=["timestamp"]).any():
        duplicate_times = df[df.duplicated(subset=["timestamp"], keep=False)]["timestamp"].tolist()
        times_str = ", ".join([str(t) for t in duplicate_times[:5]])
        if len(duplicate_times) > 5:
            times_str += f", ... and {len(duplicate_times) - 5} more"
        raise DataValidationError(f"Duplicate timestamps detected: {times_str}")
    
    return True

def fetch_market_data(
    currency_pair: Union[str, CurrencyPair],
    timeframe: Union[str, TimeFrame],
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Fetch market data from the database.
    
    Args:
        currency_pair: Currency pair (e.g., "EUR/USD" or CurrencyPair instance).
        timeframe: Timeframe (e.g., "1h" or TimeFrame instance).
        start_date: Start date for the data range.
        end_date: End date for the data range.
        
    Returns:
        A pandas DataFrame with the market data.
        
    Raises:
        DataProcessingError: If fetching or processing the data fails.
    """
    try:
        # Handle string currency pair
        if isinstance(currency_pair, str):
            if len(currency_pair) != 7 or currency_pair[3] != '/':
                raise ValueError(f"Invalid currency pair format: {currency_pair}. Expected format: 'XXX/YYY'")
            base = currency_pair[:3]
            quote = currency_pair[4:]
        else:
            base = currency_pair.base
            quote = currency_pair.quote
        
        # Handle string timeframe
        if isinstance(timeframe, str):
            timeframe_str = timeframe
        else:
            timeframe_str = timeframe.value
        
        # Get currency pair ID
        currency_pair_id = get_currency_pair_id(base, quote)
        
        # Prepare query conditions
        conditions = {
            "currency_pair_id": currency_pair_id,
            "timeframe": timeframe_str
        }
        
        # Add date range conditions if provided
        if start_date or end_date:
            date_conditions = {}
            if start_date:
                date_conditions["gte"] = start_date.isoformat()
            if end_date:
                date_conditions["lte"] = end_date.isoformat()
            
            conditions["timestamp"] = date_conditions
        
        # Query the database using Supabase
        client = get_supabase_db_client()
        data = client.fetch_all(
            "market_data",
            where=conditions,
            order_by="timestamp"
        )
        
        # Convert to DataFrame
        if not data:
            logger.warning(f"No data found for {base}/{quote} {timeframe_str}")
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        
        df = pd.DataFrame(data)
        
        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Set timestamp as index
        df.set_index("timestamp", inplace=True)
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching market data: {str(e)}")
        raise DataProcessingError(f"Failed to fetch market data: {str(e)}")

def save_market_data(
    df: pd.DataFrame,
    currency_pair: Union[str, CurrencyPair],
    timeframe: Union[str, TimeFrame],
    overwrite: bool = False
) -> int:
    """
    Save market data to the database.
    
    Args:
        df: DataFrame with market data.
        currency_pair: Currency pair (e.g., "EUR/USD" or CurrencyPair instance).
        timeframe: Timeframe (e.g., "1h" or TimeFrame instance).
        overwrite: Whether to overwrite existing data.
        
    Returns:
        Number of rows saved.
        
    Raises:
        DataProcessingError: If saving the data fails.
    """
    try:
        # Validate the DataFrame
        validate_dataframe(df)
        
        # Handle string currency pair
        if isinstance(currency_pair, str):
            if len(currency_pair) != 7 or currency_pair[3] != '/':
                raise ValueError(f"Invalid currency pair format: {currency_pair}. Expected format: 'XXX/YYY'")
            base = currency_pair[:3]
            quote = currency_pair[4:]
        else:
            base = currency_pair.base
            quote = currency_pair.quote
        
        # Handle string timeframe
        if isinstance(timeframe, str):
            timeframe_str = timeframe
        else:
            timeframe_str = timeframe.value
        
        # Get currency pair ID
        currency_pair_id = get_currency_pair_id(base, quote)
        
        # Reset index if timestamp is the index
        if df.index.name == "timestamp":
            df = df.reset_index()
        
        # Prepare data for insertion
        records = []
        for _, row in df.iterrows():
            record = {
                "currency_pair_id": currency_pair_id,
                "timeframe": timeframe_str,
                "timestamp": row["timestamp"].isoformat(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0)),
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }
            records.append(record)
        
        # If overwrite is True, delete existing data for this currency pair and timeframe
        client = get_supabase_db_client()
        if overwrite:
            # Get the timestamp range from the DataFrame
            min_timestamp = df["timestamp"].min().isoformat()
            max_timestamp = df["timestamp"].max().isoformat()
            
            # Delete existing data in this range
            client.delete(
                "market_data",
                {
                    "currency_pair_id": currency_pair_id,
                    "timeframe": timeframe_str,
                    "timestamp": {
                        "gte": min_timestamp,
                        "lte": max_timestamp
                    }
                }
            )
        
        # Insert new data
        result = client.insert_many("market_data", records)
        
        return len(records)
        
    except Exception as e:
        logger.error(f"Error saving market data: {str(e)}")
        raise DataProcessingError(f"Failed to save market data: {str(e)}")

def export_trade_history(
    strategy_id: str,
    file_path: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    format: str = "csv"
) -> None:
    """
    Export trade history for a strategy to a file.
    
    Args:
        strategy_id: ID of the strategy.
        file_path: Path to save the file.
        start_date: Start date for the export.
        end_date: End date for the export.
        format: Export format (csv, json, excel).
        
    Raises:
        DataProcessingError: If exporting fails.
    """
    try:
        # Prepare query conditions
        conditions = {"strategy_id": strategy_id}
        
        # Add date range conditions if provided
        if start_date or end_date:
            date_conditions = {}
            if start_date:
                date_conditions["gte"] = start_date.isoformat()
            if end_date:
                date_conditions["lte"] = end_date.isoformat()
            
            conditions["timestamp"] = date_conditions
        
        # Query the database
        client = get_supabase_db_client()
        trades = client.fetch_all(
            "trades",
            where=conditions,
            order_by="timestamp"
        )
        
        # Convert to DataFrame
        if not trades:
            logger.warning(f"No trades found for strategy {strategy_id}")
            df = pd.DataFrame(columns=["timestamp", "order_id", "type", "price", "size", "profit"])
        else:
            df = pd.DataFrame(trades)
            
            # Convert timestamp to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Export to file
        if format.lower() == "csv":
            df.to_csv(file_path, index=False)
        elif format.lower() == "json":
            df.to_json(file_path, orient="records", date_format="iso")
        elif format.lower() == "excel":
            df.to_excel(file_path, index=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Exported {len(df)} trades to {file_path}")
        
    except Exception as e:
        logger.error(f"Error exporting trade history: {str(e)}")
        raise DataProcessingError(f"Failed to export trade history: {str(e)}")

def import_from_csv(
    file_path: str,
    timeframe: Union[str, TimeFrame],
    currency_pair: Optional[Union[str, CurrencyPair]] = None,
    timestamp_format: str = "%Y-%m-%d %H:%M:%S",
    delimiter: str = ",",
    save_to_db: bool = False
) -> pd.DataFrame:
    """
    Import market data from a CSV file.
    
    Args:
        file_path: Path to the CSV file.
        timeframe: Timeframe of the data.
        currency_pair: Currency pair. Required if save_to_db is True.
        timestamp_format: Format of the timestamp column.
        delimiter: CSV delimiter.
        save_to_db: Whether to save the data to the database.
        
    Returns:
        A pandas DataFrame with the market data.
        
    Raises:
        DataProcessingError: If importing or processing the data fails.
    """
    try:
        # Read CSV file
        df = pd.read_csv(file_path, delimiter=delimiter)
        
        # Ensure required columns exist
        required_columns = ["timestamp", "open", "high", "low", "close"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise DataValidationError(f"Missing required columns in CSV: {', '.join(missing_columns)}")
        
        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], format=timestamp_format)
        
        # Set timestamp as index
        df.set_index("timestamp", inplace=True)
        
        # Sort by timestamp
        df.sort_index(inplace=True)
        
        # Validate DataFrame
        validate_dataframe(df.reset_index())
        
        # Save to database if requested
        if save_to_db:
            if not currency_pair:
                raise ValueError("Currency pair is required when save_to_db is True")
                
            # Handle string timeframe
            if isinstance(timeframe, str):
                timeframe_obj = TimeFrame(timeframe)
            else:
                timeframe_obj = timeframe
                
            # Create Alpha Vantage connector for database saving
            connector = AlphaVantageConnector()
            connector.save_to_database(df, currency_pair, timeframe_obj)
            
        return df
        
    except Exception as e:
        error_msg = f"Error importing market data from CSV: {str(e)}"
        logger.error(error_msg)
        raise DataProcessingError(error_msg) from e

def export_to_csv(
    data: pd.DataFrame,
    file_path: str,
    timestamp_format: str = "%Y-%m-%d %H:%M:%S",
    include_index: bool = True,
    delimiter: str = ","
) -> None:
    """
    Export market data to a CSV file.
    
    Args:
        data: The market data DataFrame.
        file_path: Path to save the CSV file.
        timestamp_format: Format of the timestamp column.
        include_index: Whether to include the index in the output.
        delimiter: CSV delimiter.
        
    Raises:
        DataProcessingError: If exporting the data fails.
    """
    try:
        # Make a copy of the DataFrame to avoid modifying the original
        df = data.copy()
        
        # Convert timezone-aware datetimes to naive for better CSV compatibility
        if include_index and isinstance(df.index, pd.DatetimeIndex):
            if df.index.tzinfo is not None:
                df.index = df.index.tz_localize(None)
                
            # Format the index if it's a datetime
            if timestamp_format:
                df.index = df.index.strftime(timestamp_format)
        
        # Export to CSV
        df.to_csv(file_path, index=include_index, delimiter=delimiter)
        logger.info(f"Exported market data to {file_path}")
        
    except Exception as e:
        error_msg = f"Error exporting market data to CSV: {str(e)}"
        logger.error(error_msg)
        raise DataProcessingError(error_msg) from e

def export_to_json(
    data: pd.DataFrame,
    file_path: str,
    orient: str = "records",
    date_format: str = "iso",
    indent: int = 4
) -> None:
    """
    Export market data to a JSON file.
    
    Args:
        data: The market data DataFrame.
        file_path: Path to save the JSON file.
        orient: The format of the JSON output.
        date_format: Format of the timestamp column.
        indent: Indentation level for the JSON.
        
    Raises:
        DataProcessingError: If exporting the data fails.
    """
    try:
        # Make a copy of the DataFrame to avoid modifying the original
        df = data.copy()
        
        # Reset index if it's a datetime index to include it in the JSON
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        
        # Export to JSON
        df.to_json(file_path, orient=orient, date_format=date_format, indent=indent)
        logger.info(f"Exported market data to {file_path}")
        
    except Exception as e:
        error_msg = f"Error exporting market data to JSON: {str(e)}"
        logger.error(error_msg)
        raise DataProcessingError(error_msg) from e

def export_to_excel(
    data: pd.DataFrame,
    file_path: str,
    sheet_name: str = "Market Data",
    include_index: bool = True
) -> None:
    """
    Export market data to an Excel file.
    
    Args:
        data: The market data DataFrame.
        file_path: Path to save the Excel file.
        sheet_name: Name of the Excel sheet.
        include_index: Whether to include the index in the output.
        
    Raises:
        DataProcessingError: If exporting the data fails.
    """
    try:
        # Export to Excel
        data.to_excel(file_path, sheet_name=sheet_name, index=include_index)
        logger.info(f"Exported market data to {file_path}")
        
    except Exception as e:
        error_msg = f"Error exporting market data to Excel: {str(e)}"
        logger.error(error_msg)
        raise DataProcessingError(error_msg) from e

def convert_timeframe(
    data: pd.DataFrame,
    source_timeframe: Union[str, TimeFrame],
    target_timeframe: Union[str, TimeFrame]
) -> pd.DataFrame:
    """
    Convert market data from one timeframe to another.
    
    Args:
        data: The market data DataFrame.
        source_timeframe: Source timeframe (e.g., "1h" or TimeFrame.H1).
        target_timeframe: Target timeframe (e.g., "4h" or TimeFrame.H4).
        
    Returns:
        A pandas DataFrame with the converted market data.
        
    Raises:
        DataProcessingError: If converting the data fails.
        ValueError: If the conversion is not supported.
    """
    try:
        # Handle string timeframes
        if isinstance(source_timeframe, str):
            source_timeframe_obj = TimeFrame(source_timeframe)
        else:
            source_timeframe_obj = source_timeframe
            
        if isinstance(target_timeframe, str):
            target_timeframe_obj = TimeFrame(target_timeframe)
        else:
            target_timeframe_obj = target_timeframe
        
        # Get pandas frequencies
        source_freq = TIMEFRAME_TO_FREQ[source_timeframe_obj]
        target_freq = TIMEFRAME_TO_FREQ[target_timeframe_obj]
        
        # Check if the target timeframe is higher than the source timeframe
        if to_offset(source_freq) >= to_offset(target_freq):
            raise ValueError(f"Target timeframe ({target_timeframe_obj.value}) must be higher than source timeframe ({source_timeframe_obj.value})")
        
        # Ensure the DataFrame is sorted by timestamp
        df = data.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")
            
        df.sort_index(inplace=True)
        
        # Resample the data
        resampled = df.resample(target_freq).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum" if "volume" in df.columns else None
        })
        
        # Drop any rows with NaN values that might have been introduced
        resampled.dropna(inplace=True)
        
        return resampled
        
    except Exception as e:
        error_msg = f"Error converting timeframe: {str(e)}"
        logger.error(error_msg)
        raise DataProcessingError(error_msg) from e

def merge_data_sources(
    dataframes: List[pd.DataFrame],
    priority: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Merge market data from multiple sources.
    
    Args:
        dataframes: List of market data DataFrames.
        priority: Priority of sources (indices of dataframes) for resolving conflicts.
                 Lower indices have higher priority.
                 
    Returns:
        A pandas DataFrame with the merged market data.
        
    Raises:
        DataProcessingError: If merging the data fails.
    """
    try:
        if not dataframes:
            return pd.DataFrame()
            
        if len(dataframes) == 1:
            return dataframes[0].copy()
        
        # Ensure all DataFrames have a DatetimeIndex
        for i, df in enumerate(dataframes):
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError(f"DataFrame {i} does not have a DatetimeIndex")
        
        # Set default priority if not provided
        if priority is None:
            priority = list(range(len(dataframes)))
            
        # Check priority list
        if len(priority) != len(dataframes):
            raise ValueError("Priority list length must match the number of DataFrames")
            
        if set(priority) != set(range(len(dataframes))):
            raise ValueError("Priority list must contain all indices from 0 to len(dataframes)-1")
        
        # Create a copy of the first DataFrame as the base
        result = dataframes[priority[0]].copy()
        
        # Merge data from other DataFrames
        for i in priority[1:]:
            df = dataframes[i].copy()
            
            # Find timestamps that are in df but not in result
            new_timestamps = df.index.difference(result.index)
            
            # Add new data
            if not new_timestamps.empty:
                result = pd.concat([result, df.loc[new_timestamps]])
                
        # Sort by timestamp
        result.sort_index(inplace=True)
        
        return result
        
    except Exception as e:
        error_msg = f"Error merging data sources: {str(e)}"
        logger.error(error_msg)
        raise DataProcessingError(error_msg) from e

def normalize_data(data: pd.DataFrame, method: str = "minmax") -> pd.DataFrame:
    """
    Normalize market data for machine learning.
    
    Args:
        data: The market data DataFrame.
        method: Normalization method (minmax, zscore).
        
    Returns:
        A pandas DataFrame with the normalized market data.
        
    Raises:
        DataProcessingError: If normalizing the data fails.
        ValueError: If the normalization method is not supported.
    """
    try:
        # Make a copy of the DataFrame to avoid modifying the original
        df = data.copy()
        
        # Select columns to normalize
        price_columns = ["open", "high", "low", "close"]
        if "volume" in df.columns:
            columns_to_normalize = price_columns + ["volume"]
        else:
            columns_to_normalize = price_columns
            
        # Apply normalization based on the method
        if method == "minmax":
            # Min-max normalization (scales to [0, 1])
            for col in columns_to_normalize:
                min_val = df[col].min()
                max_val = df[col].max()
                df[col] = (df[col] - min_val) / (max_val - min_val)
                
        elif method == "zscore":
            # Z-score normalization (scales to mean=0, std=1)
            for col in columns_to_normalize:
                mean = df[col].mean()
                std = df[col].std()
                df[col] = (df[col] - mean) / std
                
        else:
            raise ValueError(f"Unsupported normalization method: {method}")
            
        return df
        
    except Exception as e:
        error_msg = f"Error normalizing data: {str(e)}"
        logger.error(error_msg)
        raise DataProcessingError(error_msg) from e 
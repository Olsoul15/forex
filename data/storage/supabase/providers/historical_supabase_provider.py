"""
Provides historical OHLCV data by querying a Supabase (Postgres + TimescaleDB) table.
"""

import logging
import os
import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime

import pandas as pd
from supabase import create_client, Client
from supabase.lib.client_options import ClientOptions
from dotenv import load_dotenv
import httpx

from forex_ai.exceptions import DatabaseError, DataError

logger = logging.getLogger(__name__)

class HistoricalSupabaseDataProvider:
    """
    Fetches historical OHLCV data from the Supabase 'ohlcv_data' hypertable.
    """

    TABLE_NAME = "ohlcv_data"

    def __init__(self):
        """Initialize the Supabase connection."""
        try:
            load_dotenv()
            supabase_url = os.getenv("SUPABASE_URL")
            # Use the service key for backend data access
            supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
            
            logger.info(f"[DataProvider Init] Loaded SUPABASE_URL: {supabase_url is not None}")
            logger.info(f"[DataProvider Init] Loaded SUPABASE_SERVICE_KEY: {supabase_key is not None} (Length: {len(supabase_key) if supabase_key else 0})")
            
            if not supabase_url or not supabase_key:
                raise ValueError("Missing Supabase credentials for historical data provider.")

            # Configure client options with a timeout
            client_opts = ClientOptions(postgrest_client_timeout=60) # 60 seconds timeout

            self.client: Client = create_client(supabase_url, supabase_key, options=client_opts)
            logger.info("HistoricalSupabaseDataProvider initialized.")

        except Exception as e:
            logger.error(f"Error initializing HistoricalSupabaseDataProvider: {e}")
            raise DatabaseError(f"HistoricalSupabaseDataProvider initialization failed: {e}")

    async def get_historical_candles(
        self,
        pair: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
        # count is ignored if start/end are provided for DB query
        count: Optional[int] = None 
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetches historical OHLCV data from Supabase within a date range.

        Args:
            pair: Currency pair (e.g., 'EUR_USD').
            timeframe: Timeframe (e.g., 'H1').
            start_time: Start datetime (inclusive).
            end_time: End datetime (inclusive).
            count: Number of candles (ignored in this implementation).

        Returns:
            A dictionary containing the OHLCV data, indexed by timestamp,
            or an empty DataFrame if no data is found or an error occurs.
        """
        logger.info(f"Fetching historical data for {pair} ({timeframe}) from {start_time} to {end_time}")
        
        # Enhanced pair normalization
        db_pair = pair.replace('/', '_')
        if len(db_pair) == 6 and '_' not in db_pair:
            db_pair = f"{db_pair[:3]}_{db_pair[3:]}"
        logger.info(f"Querying database with instrument identifier: {db_pair}")

        try:
            # Format timestamps for Supabase query
            start_iso = start_time.isoformat()
            end_iso = end_time.isoformat()
            
            all_data = []
            offset = 0
            # batch_size = 5000 # This was the logical batch size per loop iteration
            supabase_request_limit = 1000 # Actual limit per Supabase request

            logger.info(f"Supabase query filter values: start_iso='{start_iso}', end_iso='{end_iso}'")
            # logger.info(f"Starting batched data fetch with batch_size={batch_size}") # Old log
            logger.info(f"Starting batched data fetch with Supabase request limit={supabase_request_limit}")

            while True:
                logger.info(f"Fetching batch with offset={offset}, limit={supabase_request_limit}")
                log_prefix = f"{db_pair} - {timeframe} - Offset {offset}"
                logger.info(f"[{log_prefix}] Building Supabase query: table='{self.TABLE_NAME}', instrument='{db_pair}', timeframe='{timeframe}', start='{start_iso}', end='{end_iso}', offset={offset}, limit={supabase_request_limit}")

                query = self.client.table(self.TABLE_NAME) \
                    .select("timestamp, open, high, low, close, volume") \
                    .eq("instrument", db_pair) \
                    .eq("timeframe", timeframe) \
                    .gte("timestamp", start_iso) \
                    .lt("timestamp", end_iso) \
                    .order("timestamp", desc=False) \
                    .range(offset, offset + supabase_request_limit - 1) # PostgREST range is inclusive

                # response_obj = None # Ensure response_obj is defined
                
                # Removed asyncio.to_thread and asyncio.wait_for
                logger.info(f"[{log_prefix}] Preparing to execute Supabase query directly (experimental)...")
                try:
                    response_obj = query.execute() # Direct call
                    logger.info(f"[{log_prefix}] Direct query.execute() returned a response.")
                except httpx.TimeoutException as httpx_timeout_exc: # More specific catch for httpx timeouts
                    logger.error(f"[{log_prefix}] Direct query.execute() timed out via httpx: {httpx_timeout_exc!r}")
                    response_obj = None
                except Exception as e: # Catch other potential errors from query.execute()
                    logger.error(f"[{log_prefix}] Exception during direct query.execute(): {e!r}")
                    response_obj = None

                if response_obj and hasattr(response_obj, 'data') and response_obj.data:
                    # self.logger.debug(f"[{log_prefix}] Received {len(response_obj.data)} records from Supabase.")
                    logger.info(f"Fetched {len(response_obj.data)} records for this batch.")
                    batch_df = pd.DataFrame(response_obj.data)
                    all_data.append(batch_df)

                    # Increment offset by the number of records actually fetched
                    offset += len(response_obj.data) # Prepare for next batch
                else:
                    # This else block now reliably indicates the end or an error
                    # because the loop continues as long as response_obj has *any* length > 0.
                    logger.info("No more data returned for this offset, stopping batch fetch.")
                    break # Exit loop if no data is returned

            if not all_data:
                logger.warning(f"No historical data found for {pair} ({timeframe}) in the specified range after batching.")
                return {'candles': pd.DataFrame()} # Return empty DataFrame

            # Concatenate all fetched batches
            df = pd.concat(all_data, ignore_index=True) # ignore_index needed before setting timestamp index

            logger.info(f"Successfully fetched a total of {len(df)} historical records for {pair} ({timeframe}) via batching.")
            
            # --- Post-processing after concatenating all batches ---
            # Convert timestamp to datetime objects and set as index
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')

            # Ensure numeric types (Supabase client might return strings/decimals)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Handle potential NaN from volume conversion if volume is optional/null
            if 'volume' in df.columns:
                 df['volume'] = df['volume'].fillna(0).astype(int) # Fill NaN with 0, ensure int

            # ---------------------------------------------------------

            # Return the DataFrame wrapped in a dictionary
            return {'candles': df}

        except Exception as e:
            logger.error(f"Error fetching historical data from Supabase for {pair} ({timeframe}): {e}", exc_info=True)
            # Return an empty DataFrame wrapped in the expected dictionary structure on error
            return {'candles': pd.DataFrame()} 

    async def data_exists_for_range(self, pair: str, timeframe: str, start_time: datetime, end_time: datetime) -> bool:
        """
        Checks if any historical data exists for the given pair, timeframe, and date range.
        Returns True if at least one record is found, False otherwise.
        """
        logger.info(f"Checking data existence for {pair} ({timeframe}) between {start_time} and {end_time}...")
        # Enhanced pair normalization
        db_pair = pair.replace('/', '_')
        if len(db_pair) == 6 and '_' not in db_pair:
            db_pair = f"{db_pair[:3]}_{db_pair[3:]}"
        
        try:
            # Construct the query to fetch just one record if any exist
            query = self.client.table(self.TABLE_NAME) \
                .select("timestamp") \
                .eq("instrument", db_pair) \
                .eq("timeframe", timeframe) \
                .gte("timestamp", start_time.isoformat()) \
                .lt("timestamp", end_time.isoformat()) \
                .limit(1)

            response = query.execute()

            if response.data:
                logger.info(f"Data found for {pair} ({timeframe}) in range.")
                return True
            else:
                logger.info(f"No data found for {pair} ({timeframe}) in range.")
                return False
        except Exception as e:
            logger.error(f"Unexpected error checking data existence for {pair} ({timeframe}): {e}", exc_info=True)
            return False # Assume no data on other errors

    async def get_latest_timestamp(self, pair: str, timeframe: str) -> Optional[str]:
        """
        Fetches the most recent timestamp for the given pair and timeframe.
        Returns the timestamp as an ISO string if found, None otherwise.
        """
        logger.info(f"Fetching latest timestamp for {pair} ({timeframe})...")
        # Enhanced pair normalization
        db_pair = pair.replace('/', '_')
        if len(db_pair) == 6 and '_' not in db_pair:
            db_pair = f"{db_pair[:3]}_{db_pair[3:]}"

        try:
            query = self.client.table(self.TABLE_NAME) \
                .select("timestamp") \
                .eq("instrument", db_pair) \
                .eq("timeframe", timeframe) \
                .order("timestamp", desc=True) \
                .limit(1)

            response = query.execute()

            if response.data and response.data[0].get("timestamp"):
                latest_ts_str = response.data[0]["timestamp"]
                logger.info(f"Latest timestamp found for {db_pair} ({timeframe}): {latest_ts_str}") # Ensure this uses db_pair
                return latest_ts_str
            else:
                logger.info(f"No timestamp found at all for {db_pair} ({timeframe}).") # Ensure this uses db_pair
                return None
        except Exception as e:
            logger.error(f"Unexpected error fetching latest timestamp for {db_pair} ({timeframe}): {e}", exc_info=True) # Ensure this uses db_pair
            return None

# Example usage (for testing this provider directly):
# async def main():
# ... existing code ... 
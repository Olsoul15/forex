import os
import sys
from dotenv import load_dotenv
from supabase import create_client, Client, PostgrestAPIResponse
from datetime import datetime, timezone, timedelta # Added timedelta
import pandas as pd # Using pandas for potentially better display
import argparse # Import argparse
from typing import Optional

# Add project root to path if the script is run from elsewhere
# project_root = os.path.dirname(os.path.abspath(__file__)) 
# sys.path.insert(0, project_root)

# --- Configuration ---
# INSTRUMENT_TO_CHECK = "EUR_USD" # Will be passed as argument
TIMEFRAME_TO_CHECK = "M1" # Hardcoded to M1 per user request
TABLE_NAME = "ohlcv_data"
# START_DATE_FILTER = "2025-01-01T00:00:00+00:00" # REMOVED - Will be calculated dynamically
ENV_PATH = '.env' # Assume .env is in the same directory as the script or project root
# ---------------------

# --- Specific Timeframe for Examination --- REMOVED
# Determined from previous run showing data concentration
# EXAMINE_START_TIME_ISO = "2025-04-22T07:38:00+00:00"
# EXAMINE_END_TIME_ISO = "2025-04-22T09:21:00+00:00"
# RECORD_LIMIT = 30
# ---------------------------------------- REMOVED

def check_duplicates(supabase: Client, instrument: str, timeframe: str):
    """Fetches timestamps and counts duplicates in Python for a given instrument/timeframe."""
    print(f"\n--- Checking for Duplicates ({instrument} {timeframe}) ---")
    try:
        # Fetch all timestamps for the specified instrument and timeframe
        # This might be slow for millions of rows
        print(f"Fetching all timestamps for {instrument} {timeframe} to check distinct count...")
        response = supabase.table(TABLE_NAME).select("timestamp") \
                                 .eq('instrument', instrument) \
                                 .eq('timeframe', timeframe) \
                                 .execute()

        if hasattr(response, 'data') and response.data:
            timestamps = [d['timestamp'] for d in response.data]
            distinct_timestamps = set(timestamps)
            num_total = len(timestamps)
            num_distinct = len(distinct_timestamps)
            num_duplicates = num_total - num_distinct

            print(f"  Total records found: {num_total}") # Use count from this query
            print(f"  Distinct timestamps: {num_distinct}")
            if num_duplicates > 0:
                 print(f"  WARNING: Found {num_duplicates} duplicate timestamp entries!")
            else:
                 print("  No duplicate timestamps found.")
        else:
            print("  Could not fetch timestamp data to check distinct count.")
            if hasattr(response, 'error') and response.error:
                 print(f"  Supabase Error: {response.error}")

    except Exception as e:
        print(f"An error occurred during duplicate check: {e}")
        import traceback
        traceback.print_exc()

def query_ohlcv_data(instrument_to_check: str, specific_start_date: Optional[str] = None, specific_end_date: Optional[str] = None):
    """Connects to Supabase and queries OHLCV data. 
       If specific_start_date and specific_end_date are provided, queries that range.
       Otherwise, queries for data >= 3 years ago.
    """

    print(f"Loading environment variables from: {os.path.abspath(ENV_PATH)}")
    if not os.path.exists(ENV_PATH):
        print(f"Error: .env file not found at {os.path.abspath(ENV_PATH)}")
        print("Please ensure the .env file with Supabase credentials exists in the project root.")
        return
        
    load_dotenv(dotenv_path=ENV_PATH)

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY")

    if not supabase_url or not supabase_key:
        print("Error: SUPABASE_URL or SUPABASE_SERVICE_KEY not found in environment variables.")
        print("Please check your .env file.")
        return

    print("Attempting to connect to Supabase...")
    try:
        supabase: Client = create_client(supabase_url, supabase_key)
        print("Supabase client created.")
    except Exception as e:
        print(f"Error creating Supabase client: {e}")
        return

    # Determine date range for query
    if specific_start_date and specific_end_date:
        start_date_filter_iso = specific_start_date
        end_date_filter_iso = specific_end_date
        print(f"\nQuerying table '{TABLE_NAME}' for specific date range:")
        print(f"  Start Date: {start_date_filter_iso}")
        print(f"  End Date:   {end_date_filter_iso}")
    else:
        # Calculate the start date (3 years ago from now, in UTC)
        now_utc = datetime.now(timezone.utc)
        three_years_ago = now_utc - timedelta(days=3*365) # Approximation: 3 years
        start_date_filter_iso = three_years_ago.isoformat()
        end_date_filter_iso = None # No specific end date for the 3-year query, goes to latest
        print(f"\nQuerying table '{TABLE_NAME}' for data >= {start_date_filter_iso} (approx 3 years ago):")

    print(f"  Instrument: {instrument_to_check}")
    print(f"  Timeframe: {TIMEFRAME_TO_CHECK}")

    try:
        query = supabase.table(TABLE_NAME)\
            .select("timestamp", count='exact')\
            .eq("instrument", instrument_to_check)\
            .eq("timeframe", TIMEFRAME_TO_CHECK)\
            .gte("timestamp", start_date_filter_iso)
        
        if end_date_filter_iso:
            query = query.lt("timestamp", end_date_filter_iso) # Use Less Than for end date as per server log logic

        response: PostgrestAPIResponse = query.execute()

        if hasattr(response, 'count'):
            count = response.count
            print(f"\n--- Query Results ({instrument_to_check} {TIMEFRAME_TO_CHECK}) ---")
            if specific_start_date and specific_end_date:
                print(f"Found a total of {count} records between {start_date_filter_iso} and {end_date_filter_iso}.")
            else:
                print(f"Found a total of {count} records >= {start_date_filter_iso} (using PostgREST count).")
            
            if count > 0:
                # Fetch first and last timestamps separately for range
                range_query_start = supabase.table(TABLE_NAME)\
                                   .select("timestamp")\
                                   .eq("instrument", instrument_to_check)\
                                   .eq("timeframe", TIMEFRAME_TO_CHECK)\
                                   .gte("timestamp", start_date_filter_iso)
                if end_date_filter_iso:
                    range_query_start = range_query_start.lt("timestamp", end_date_filter_iso)
                
                range_response_start = range_query_start.order("timestamp", desc=False).limit(1).execute()
                start_data = range_response_start.data

                range_query_end = supabase.table(TABLE_NAME)\
                                   .select("timestamp")\
                                   .eq("instrument", instrument_to_check)\
                                   .eq("timeframe", TIMEFRAME_TO_CHECK)\
                                   .gte("timestamp", start_date_filter_iso)
                if end_date_filter_iso:
                    range_query_end = range_query_end.lt("timestamp", end_date_filter_iso)

                range_response_end = range_query_end.order("timestamp", desc=True).limit(1).execute()
                end_data = range_response_end.data

                first_timestamp = pd.to_datetime(start_data[0]['timestamp']) if start_data else "N/A"
                latest_timestamp = pd.to_datetime(end_data[0]['timestamp']) if end_data else "N/A"
                print(f"  Actual Start in range: {first_timestamp}")
                print(f"  Actual End in range:   {latest_timestamp}")
            else:
                print("  Actual Start: N/A")
                print("  Actual End:   N/A")
        elif hasattr(response, 'error') and response.error:
             print(f"Supabase API Error during count: {response.error}")
        else:
             # Fallback if count attribute isn't available (older libraries?)
             print("Count attribute not found in response. Fetching all data to count manually...")
             fallback_response = supabase.table(TABLE_NAME)\
                                    .select("timestamp")\
                                    .eq("instrument", instrument_to_check)\
                                    .eq("timeframe", TIMEFRAME_TO_CHECK)\
                                    .gte("timestamp", start_date_filter_iso)\
                                    .execute()
             if hasattr(fallback_response, 'data'):
                 count = len(fallback_response.data)
                 print(f"\n--- Query Results ({instrument_to_check} {TIMEFRAME_TO_CHECK}) ---")
                 print(f"Found a total of {count} records >= {start_date_filter_iso} (manual count).")
                 # ... (Can still calculate range if needed) ...
             else:
                  print("Unexpected response structure from Supabase during count fallback.")
                  print(f"Raw response: {fallback_response}")

    except Exception as e:
        print(f"\nAn error occurred during the query: {e}")
        import traceback
        traceback.print_exc()

    # --- Add call to duplicate check ---
    check_duplicates(supabase, instrument_to_check, TIMEFRAME_TO_CHECK)
    # -----------------------------------

if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description=f"Query Supabase '{TABLE_NAME}' for M1 data range starting approximately 3 years ago and check duplicates.") # Reverted description
    parser.add_argument(
        "instrument",
        type=str,
        help="The instrument/currency pair to query (e.g., EUR_USD)."
    )
    parser.add_argument(
        "--start_date",
        type=str,
        help="Optional. The specific start date for the query (ISO format, e.g., 2024-04-10T00:00:00Z).",
        default=None
    )
    parser.add_argument(
        "--end_date",
        type=str,
        help="Optional. The specific end date for the query (ISO format, e.g., 2024-04-10T03:00:00Z).",
        default=None
    )
    args = parser.parse_args()

    # Call the function with the parsed argument
    query_ohlcv_data(instrument_to_check=args.instrument, specific_start_date=args.start_date, specific_end_date=args.end_date) 
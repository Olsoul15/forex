import os
import sys
from dotenv import load_dotenv
from supabase import create_client, Client, PostgrestAPIResponse
import argparse
from datetime import datetime

# Add project root to sys.path if necessary
# project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# --- Configuration ---
TABLE_NAME = "ohlcv_data"
ENV_PATH = '.env'
# ---------------------

def count_records_in_range(instrument: str, timeframe: str, start_date_iso: str, end_date_iso: str):
    """Connects to Supabase and counts records within a specific date range."""

    print(f"Loading environment variables from: {os.path.abspath(ENV_PATH)}")
    if not os.path.exists(ENV_PATH):
        print(f"Error: .env file not found at {os.path.abspath(ENV_PATH)}")
        print("Please ensure the .env file with Supabase credentials exists.")
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

    print(f"Querying table '{TABLE_NAME}' for count:")
    print(f"  Instrument: {instrument}")
    print(f"  Timeframe:  {timeframe}")
    print(f"  Start Date: {start_date_iso}")
    print(f"  End Date:   {end_date_iso}")

    try:
        # Validate ISO format before querying (optional but good practice)
        datetime.fromisoformat(start_date_iso.replace('Z', '+00:00'))
        datetime.fromisoformat(end_date_iso.replace('Z', '+00:00'))

        response: PostgrestAPIResponse = (
            supabase.table(TABLE_NAME)
            .select("*", count='exact') # Select anything, we only need the count
            .eq("instrument", instrument)
            .eq("timeframe", timeframe)
            .gte("timestamp", start_date_iso)
            .lte("timestamp", end_date_iso) # Use lte for inclusive end date
            .execute()
        )

        if hasattr(response, 'count'):
             count = response.count
             print(f"--- Query Result ---")
             print(f"Found {count} records for {instrument} {timeframe} between {start_date_iso} and {end_date_iso}.")
        elif hasattr(response, 'error') and response.error:
             print(f"Supabase API Error during count: {response.error}")
        else:
            print("Count attribute not found in response. The query might have failed or the library version is incompatible.")
            print(f"Raw Response: {response}")


    except ValueError as ve:
        print(f"Error: Invalid date format provided. Please use ISO 8601 format (e.g., YYYY-MM-DDTHH:MM:SS+00:00 or YYYY-MM-DDTHH:MM:SSZ). Details: {ve}")
    except Exception as e:
        print(f"An error occurred during the query: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Count records in Supabase '{TABLE_NAME}' within a specific date range.")
    parser.add_argument("instrument", type=str, help="Instrument (e.g., EUR_USD)")
    parser.add_argument("timeframe", type=str, help="Timeframe (e.g., M1)")
    parser.add_argument("start_date", type=str, help="Start date/time in ISO 8601 format (e.g., 2024-04-01T00:00:00+00:00 or ...Z)")
    parser.add_argument("end_date", type=str, help="End date/time in ISO 8601 format (e.g., 2024-05-01T00:00:00+00:00 or ...Z)")

    args = parser.parse_args()

    # Ensure Z is converted to +00:00 if present, Supabase might prefer the offset format
    start_date_iso = args.start_date.replace('Z', '+00:00')
    end_date_iso = args.end_date.replace('Z', '+00:00')


    count_records_in_range(args.instrument, args.timeframe, start_date_iso, end_date_iso) 
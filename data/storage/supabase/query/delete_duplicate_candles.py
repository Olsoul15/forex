import os
import sys
import time
from dotenv import load_dotenv
from supabase import create_client, Client
import argparse
import math

# --- Configuration ---
TABLE_NAME = "ohlcv_data"
ENV_PATH = '.env' # Assume .env is in the same directory or project root
BATCH_SIZE = 5000  # How many duplicates to delete in each transaction
DELAY_SECONDS = 0.5 # Optional delay between batches to reduce load
FUNCTION_NAME = "get_duplicate_ctids" # The required SQL function name
# ---------------------

# --- SQL Function Definition (MUST BE CREATED IN SUPABASE SQL EDITOR FIRST) ---
# This function efficiently finds the ctid (unique row identifier) of duplicate rows
# based on instrument, timeframe, and timestamp.
REQUIRED_SQL_FUNCTION = f"""
-- Drop the function if it already exists (optional, for easy recreation)
DROP FUNCTION IF EXISTS {FUNCTION_NAME}(text, text);

-- Create the function
CREATE OR REPLACE FUNCTION {FUNCTION_NAME}(
    target_instrument text DEFAULT NULL,
    target_timeframe text DEFAULT NULL
)
RETURNS TABLE(duplicate_ctid tid) AS $$
BEGIN
    RETURN QUERY
    SELECT ct.ctid
    FROM (
        SELECT
            ctid,
            ROW_NUMBER() OVER(
                PARTITION BY ohlcv_data.instrument, ohlcv_data.timeframe, ohlcv_data."timestamp"
                ORDER BY ohlcv_data.ctid -- Keep the 'first' row based on internal id
            ) as rn
        FROM
            ohlcv_data
        WHERE
            (target_instrument IS NULL OR ohlcv_data.instrument = target_instrument) AND
            (target_timeframe IS NULL OR ohlcv_data.timeframe = target_timeframe)
    ) ct
    WHERE ct.rn > 1; -- Select only the duplicates (row number > 1)
END;
$$ LANGUAGE plpgsql;

-- Example Usage in SQL Editor (optional):
-- SELECT * FROM {FUNCTION_NAME}('EUR_USD', 'M1'); -- Find duplicates for specific pair/timeframe
-- SELECT count(*) FROM {FUNCTION_NAME}(); -- Count all duplicates in the table
"""
# --------------------------------------------------------------------------

def delete_duplicates_in_batches(supabase: Client, instrument: str | None, timeframe: str | None):
    """Finds and deletes duplicate records in batches."""

    print(f"\n--- Step 1: Finding Duplicate Records ---")
    print(f"Calling Supabase function '{FUNCTION_NAME}'...")
    print(f"  Filtering by Instrument: {instrument if instrument else 'All'}")
    print(f"  Filtering by Timeframe: {timeframe if timeframe else 'All'}")

    try:
        # Call the RPC function created in Supabase
        # Pass params even if None, the SQL function handles NULL properly
        params = {'target_instrument': instrument, 'target_timeframe': timeframe}
        response = supabase.rpc(FUNCTION_NAME, params).execute()

        if response.data:
            duplicate_ctids = [item['duplicate_ctid'] for item in response.data]
            total_duplicates = len(duplicate_ctids)
            print(f"Found {total_duplicates} duplicate records to delete.")
        else:
            print("No duplicate records found matching the criteria.")
            if response.error:
                 print(f"Supabase RPC Error: {response.error}")
            return

        if total_duplicates == 0:
            return

    except Exception as e:
        print(f"Error calling Supabase function '{FUNCTION_NAME}': {e}")
        print("Please ensure the SQL function has been created correctly in the Supabase SQL Editor.")
        return

    # --- Confirmation ---
    print(f"\n--- Step 2: Confirmation ---")
    confirm = input(f"Proceed with deleting {total_duplicates} duplicate records from '{TABLE_NAME}'? (yes/no): ")
    if confirm.lower() != 'yes':
        print("Deletion cancelled by user.")
        return

    # --- Batch Deletion ---
    print(f"\n--- Step 3: Batch Deletion (Batch Size: {BATCH_SIZE}) ---")
    deleted_count = 0
    num_batches = math.ceil(total_duplicates / BATCH_SIZE)

    for i in range(num_batches):
        batch_start_index = i * BATCH_SIZE
        batch_end_index = min((i + 1) * BATCH_SIZE, total_duplicates)
        batch_ctids = duplicate_ctids[batch_start_index:batch_end_index]

        print(f"Processing Batch {i + 1}/{num_batches} (Records {batch_start_index + 1} to {batch_end_index})...", end='')

        try:
            # Use PostgREST to delete by ctid (ctid must be cast for the filter)
            delete_response = supabase.table(TABLE_NAME).delete().filter(
                'ctid', 'in', f'({",".join([f"\'{c}\'::tid" for c in batch_ctids])})' # Correct casting and formatting
            ).execute()

            # Supabase-py v1 doesn't directly give count for delete, v2 might differ
            # We assume success if no error is raised, check 'error' attribute if available
            if hasattr(delete_response, 'error') and delete_response.error:
                 print(f" Error: {delete_response.error}")
                 # Decide whether to stop or continue on batch error
                 cont = input("An error occurred in this batch. Continue with next batch? (yes/no): ")
                 if cont.lower() != 'yes':
                     print("Stopping batch deletion.")
                     break
            else:
                 print(f" Success.")
                 deleted_count += len(batch_ctids)


        except Exception as e:
            print(f" Error during batch {i + 1}: {e}")
            cont = input("An error occurred in this batch. Continue with next batch? (yes/no): ")
            if cont.lower() != 'yes':
                print("Stopping batch deletion.")
                break

        # Optional delay
        if DELAY_SECONDS > 0 and i < num_batches - 1:
            time.sleep(DELAY_SECONDS)

    print(f"\n--- Deletion Complete ---")
    print(f"Attempted to delete {deleted_count}/{total_duplicates} duplicate records.")


if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description=f"Deletes duplicate records from the '{TABLE_NAME}' table in Supabase in batches.")
    parser.add_argument(
        "-i", "--instrument",
        type=str,
        help="Optional: The specific instrument/currency pair to clean (e.g., EUR_USD). If omitted, cleans all instruments.",
        default=None
    )
    parser.add_argument(
        "-t", "--timeframe",
        type=str,
        help="Optional: The specific timeframe to clean (e.g., M1). Requires --instrument. If omitted, cleans all timeframes for the specified instrument (or all instruments if -i is also omitted).",
        default=None
    )
    args = parser.parse_args()

    if args.timeframe and not args.instrument:
         parser.error("--timeframe requires --instrument to be specified.")


    # --- Pre-computation Step: Ensure SQL Function Exists ---
    print("--- IMPORTANT PRE-REQUISITE --- ")
    print(f"This script requires a SQL function named '{FUNCTION_NAME}' to exist in your Supabase database.")
    print("Please run the following SQL code in your Supabase SQL Editor BEFORE proceeding:")
    print("-" * 70)
    print(REQUIRED_SQL_FUNCTION)
    print("-" * 70)
    confirm_sql = input(f"Have you created or verified the '{FUNCTION_NAME}' SQL function? (yes/no): ")
    if confirm_sql.lower() != 'yes':
        print("Exiting script. Please create the SQL function first.")
        sys.exit(1)

    # --- Main Execution ---
    print(f"Loading environment variables from: {os.path.abspath(ENV_PATH)}")
    if not os.path.exists(ENV_PATH):
        print(f"Error: .env file not found at {os.path.abspath(ENV_PATH)}")
        sys.exit(1)

    load_dotenv(dotenv_path=ENV_PATH)

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY") # Use service key for delete permissions

    if not supabase_url or not supabase_key:
        print("Error: SUPABASE_URL or SUPABASE_SERVICE_KEY not found in environment variables.")
        sys.exit(1)

    print("Attempting to connect to Supabase...")
    try:
        supabase: Client = create_client(supabase_url, supabase_key)
        print("Supabase client created.")
    except Exception as e:
        print(f"Error creating Supabase client: {e}")
        sys.exit(1)

    delete_duplicates_in_batches(supabase, args.instrument, args.timeframe)

    print("\nScript finished.") 
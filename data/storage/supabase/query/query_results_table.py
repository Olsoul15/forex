import os
import sys
import argparse
import json
from dotenv import load_dotenv
from supabase import create_client, Client, PostgrestAPIResponse
import pandas as pd

# --- Configuration ---
ENV_PATH = '.env'
# ---------------------

def query_table(table_name: str):
    """Connects to Supabase and queries the specified table."""

    print(f"Loading environment variables from: {os.path.abspath(ENV_PATH)}")
    if not os.path.exists(ENV_PATH):
        print(f"Error: .env file not found at {os.path.abspath(ENV_PATH)}")
        print("Please ensure the .env file with Supabase credentials exists.")
        return

    load_dotenv(dotenv_path=ENV_PATH)

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY") # Use service key for broad access

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

    try:
        print(f"\nQuerying table '{table_name}' for all data...")
        response: PostgrestAPIResponse = (
            supabase.table(table_name)
            .select("*", count='exact') # Select all columns and get count
            .execute()
        )

        # Check response structure carefully
        if hasattr(response, 'data'):
            data = response.data if response.data is not None else []
            count = response.count if hasattr(response, 'count') and response.count is not None else len(data)

            print(f"\n--- Query Results ({table_name}) ---")
            print(f"Found a total of {count} record(s).")

            if count > 0:
                # Option 1: Pretty Print JSON (Good for nested data)
                # print(json.dumps(data, indent=2, default=str)) 

                # Option 2: Use Pandas DataFrame (Better for tabular data)
                try:
                    df = pd.DataFrame(data)
                    pd.set_option('display.max_rows', None)
                    pd.set_option('display.max_columns', None)
                    pd.set_option('display.width', 1000) # Adjust width as needed
                    pd.set_option('display.max_colwidth', None)
                    print(df.to_string())
                except Exception as pd_err:
                    print(f"Error creating DataFrame, falling back to JSON print: {pd_err}")
                    print(json.dumps(data, indent=2, default=str))
            else:
                print("No data found in the table.")

        elif hasattr(response, 'error') and response.error:
            print(f"Supabase API Error: {response.error}")
        else:
            print("Unexpected response structure from Supabase.")
            print(f"Raw response: {response}")

    except Exception as e:
        print(f"\nAn error occurred during the query for table '{table_name}': {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Query a specified Supabase table.")
    parser.add_argument(
        "table_name",
        type=str,
        help="The name of the Supabase table to query."
    )
    args = parser.parse_args()

    # Call the function with the parsed argument
    query_table(table_name=args.table_name) 
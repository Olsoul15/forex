import os
import sys
import argparse # Import argparse
import json     # Import json
from dotenv import load_dotenv
from supabase import create_client, Client, PostgrestAPIResponse
import pandas as pd # Using pandas for potentially better display

# --- Configuration ---
TABLE_NAME = "strategy_configurations"
ENV_PATH = '.env' # Assume .env is in the same directory as the script or project root
# ---------------------

def query_strategy_configurations(strategy_id: str | None = None): # Accept optional strategy_id
    """Connects to Supabase and queries the strategy_configurations table.
    If strategy_id is provided, fetches the full config for that ID.
    Otherwise, lists basic info for all strategies.
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

    try:
        if strategy_id:
            print(f"\nQuerying table '{TABLE_NAME}' for strategy ID: {strategy_id}...")
            response: PostgrestAPIResponse = (
                supabase.table(TABLE_NAME)
                .select("*") # Select all columns
                .eq('id', strategy_id) # Filter by ID
                .single() # Fetch a single record
                .execute()
            )

            # ADDED: Print raw response for debugging
            print(f"Raw Supabase Response: {response}") 

            if hasattr(response, 'data') and response.data:
                 print(f"\n--- Configuration for Strategy ID: {strategy_id} ---")
                 # Pretty print the JSON data
                 print(json.dumps(response.data, indent=2, default=str)) # Use default=str for non-serializable types like datetime

            elif hasattr(response, 'error') and response.error:
                 print(f"Supabase API Error: {response.error}")
                 # Handle specific case where single() returns no rows
                 if 'PGRST116' in str(response.error): # Check if the error code is for zero rows
                     print(f"Strategy with ID '{strategy_id}' not found.")
                 else:
                     print(f"An unexpected Supabase error occurred: {response.error}")
            elif hasattr(response, 'data') and not response.data:
                print(f"Strategy with ID '{strategy_id}' not found.")
            else:
                 print("Unexpected response structure from Supabase.")
                 print(f"Raw response: {response}")

        else:
            # Original logic to list all strategies
            print(f"\nQuerying table '{TABLE_NAME}' for all available strategies...")
            response: PostgrestAPIResponse = (
                supabase.table(TABLE_NAME)
                # Select relevant columns including the new origin AND logic strings
                .select("id, name, strategy_type, strategy_origin, entry_logic, exit_logic", count='exact') # ADDED logic fields
                .order("name", desc=False) # Order by name
                .execute()
            )

            if hasattr(response, 'data'):
                 data = response.data if response.data else []
                 count = response.count if hasattr(response, 'count') and response.count is not None else len(data)

                 print(f"\n--- Query Results ({TABLE_NAME}) ---")
                 print(f"Found a total of {count} strategy configurations.")

                 if count > 0:
                     df = pd.DataFrame(data)
                     # Optional: Display the full DataFrame if needed
                     pd.set_option('display.max_rows', None)
                     pd.set_option('display.max_columns', None)
                     pd.set_option('display.width', 1000) # Increase width for better display
                     pd.set_option('display.max_colwidth', None) # Show full logic strings
                     print("--- Strategy Data ---") # Header for data
                     print(df.to_string())
                 else:
                     print("No strategies found in the table.") # Explicit message

            elif hasattr(response, 'error') and response.error:
                 print(f"Supabase API Error: {response.error}")
            else:
                 print("Unexpected response structure from Supabase.")
                 print(f"Raw response: {response}")

    except Exception as e:
        print(f"\nAn error occurred during the query: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description=f"Query the Supabase '{TABLE_NAME}' table.")
    parser.add_argument(
        "--strategy-id",
        type=str,
        help="Optional: The specific strategy ID to query for full details."
    )
    args = parser.parse_args()

    # Call the function with the parsed argument
    query_strategy_configurations(strategy_id=args.strategy_id)
    print("\nScript finished.") # Add a final message 
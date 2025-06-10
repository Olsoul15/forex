'''
Queries the backtest_results table in Supabase for a specific job ID.
'''

import os
import sys
from dotenv import load_dotenv
from supabase import create_client, Client

# Ensure the forex_ai module can be found
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load environment variables
load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

TABLE_NAME = "backtest_results"
# JOB_ID_TO_CHECK = "9df66e78-0072-4d3b-bb52-f70291756eea" # Previous job ID
# JOB_ID_TO_CHECK = "18451a20-aa2f-4e93-a612-b4c3c735be54" # Latest job ID with debug logs
# JOB_ID_TO_CHECK = "ad93a288-4269-48c7-bd3b-71baca6ae21e" # Previous VWAP test attempt
# JOB_ID_TO_CHECK = "a7acf375-be04-4a1b-84c6-75e0b26d06cf" # Current OBV test
# JOB_ID_TO_CHECK = "1cb28251-24e6-4ef6-af6d-f02f9604a84c" # Previous Pivot Point test (with fix)
# JOB_ID_TO_CHECK = "f7ca6a09-7a8d-43a7-9cbc-471f41b37b7c" # Current Pivot Point test (try-except fix)
# JOB_ID_TO_CHECK = "79e79e2a-38c5-4dc9-8ab5-4a7cfe5f677e" # Current Pivot Point test (CustomPivot fix)
# JOB_ID_TO_CHECK = "77f59f72-7de4-4d7c-820f-235eef593b05" # Current Pivot Point test (Import fix)
# JOB_ID_TO_CHECK = "5553210c-a27d-472f-a9aa-54c4bcc2672c" # Current Pivot Point test (Simplified Logic)
# JOB_ID_TO_CHECK = "788c5b75-1668-4bc1-b498-6f7e53681e68" # Current Pivot Point test (Manual Calc)
# JOB_ID_TO_CHECK = "c1276ebe-df77-4ace-94e0-9ee6b4fabc03" # Current Pivot Point test (Refined Manual Calc)
# JOB_ID_TO_CHECK = "4844c8e3-0c07-4b6d-a650-71bb3683bb80"  # <-- UPDATE THIS WITH THE ACTUAL JOB ID
# JOB_ID_TO_CHECK = "23d411b3-0b54-4c1e-b9f1-92a1b11eba57" # Latest Pivot Point test
# JOB_ID_TO_CHECK = "7f6de1c3-0c07-4b6d-a650-71bb3683bb80" # Test after moving pivot calc
# JOB_ID_TO_CHECK = "c5fe8c17-4bc8-4739-9db0-100529f0a300" # Latest Pivot Point re-test
# JOB_ID_TO_CHECK = "f77178c7-67b0-4e9c-a1b3-b5d421e7ea5f" # Pivot Point test after Crossover/Index fixes
# JOB_ID_TO_CHECK = "a631d655-80ce-4f99-9e7e-576155a26d62" # Latest run after formatter fix
# JOB_ID_TO_CHECK = "350ac4bd-7dc5-45cd-b6cb-fc8f20512f4b" # Run after restarting services 5/5 PM
# JOB_ID_TO_CHECK = "81d4dcdc-3c4a-40ea-8339-0bbcb566a0ef" # Job ID seen in worker log after last restart
JOB_ID_TO_CHECK = "debd58c7-77df-4e4f-9be4-9b4e4036e26d" # One-week test run

def query_job_status(supabase: Client, job_id: str):
    '''Queries the status and error message for a specific job ID.'''
    try:
        print(f"Querying {TABLE_NAME} for job_id: {job_id}")
        response = supabase.table(TABLE_NAME)\
                         .select("job_id, strategy_id, status, error_message, created_at, updated_at")\
                         .eq("job_id", job_id)\
                         .execute()

        print(f"Response status code: {response.status_code if hasattr(response, 'status_code') else 'N/A'}")
        print(f"Raw response data: {response.data}")

        if response.data:
            record = response.data[0]
            print("\n--- Job Status ---")
            print(f"Job ID:        {record.get('job_id')}")
            print(f"Strategy ID:   {record.get('strategy_id')}")
            print(f"Status:        {record.get('status')}")
            print(f"Error Message: {record.get('error_message')}")
            print(f"Created At:    {record.get('created_at')}")
            print(f"Updated At:    {record.get('updated_at')}")
            print("------------------")
        else:
            print(f"No record found for job_id: {job_id}")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        print("Error: SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in the .env file.")
        sys.exit(1)

    try:
        supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        print("Supabase client created successfully.")
        query_job_status(supabase_client, JOB_ID_TO_CHECK)
    except Exception as e:
        print(f"Failed to initialize Supabase client or run query: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 
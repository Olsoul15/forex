import os
import logging
# from supabase_py_async import create_client, AsyncClient # Old incorrect import
from supabase import create_client, Client as AsyncClient # Use main package
from supabase.lib.client_options import ClientOptions
from dotenv import load_dotenv
from typing import Optional, Dict, Any
from forex_ai.config.settings import get_settings
import uuid

# Configure logging
logger = logging.getLogger(__name__)

class SupabaseBacktestResultRepository:
    """
    Repository class for interacting with the backtest_results table in Supabase.
    """
    TABLE_NAME = "backtest_results"

    def __init__(self, settings: Optional[dict] = None, db_client: Optional[AsyncClient] = None):
        if db_client:
            self.client: AsyncClient = db_client
            logger.info(f"SupabaseBacktestResultRepository initialized with provided async db_client for {self.TABLE_NAME}.")
        else:
            logger.warning(
                f"SupabaseBacktestResultRepository for {self.TABLE_NAME} is creating its own synchronous Supabase client. "
                "This is deprecated and may cause issues. Please pass an async client via db_client parameter."
            )
            if settings is None:
                settings = get_settings()

            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_SERVICE_KEY")

            if not supabase_url or not supabase_key:
                logger.error("SUPABASE_URL or SUPABASE_SERVICE_KEY not found.")
                raise ValueError("Supabase credentials missing.")

            logger.info(f"Initializing (deprecated) synchronous Supabase client for {self.TABLE_NAME} repository.")
            try:
                self.client: AsyncClient = create_client(supabase_url, supabase_key)
                logger.info("(Deprecated) Synchronous Supabase client initialized.")
            except Exception as e:
                logger.exception(f"Failed to initialize (deprecated) synchronous Supabase client: {e}")
                raise

    async def create_or_update_status(
        self,
        job_id: str,
        strategy_id: str,
        status: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        initial_balance: Optional[float] = None,
    ) -> bool:
        """
        Creates a new record or updates status using upsert. Async version.
        """
        # Use the client initialized in __init__
        client = self.client
        record = {
            "job_id": job_id,
            "strategy_id": strategy_id,
            "status": status,
            "start_date": start_date,
            "end_date": end_date,
            "initial_balance": initial_balance,
        }
        try:
            logger.info(f"Attempting async upsert for job_id: {job_id} with status: {status}")
            # Execute the upsert - remove await as execute() might be synchronous
            response = client.table(self.TABLE_NAME).upsert(record).execute()

            # Response handling logic remains similar for sync/async client v1+
            if response and hasattr(response, 'data'):
                 logger.info(f"Async upsert successful for job_id: {job_id}. Response data: {response.data}")
                 # Check if data is non-empty, as upsert might return empty data on success
                 return True # Assuming success if no error
            elif response:
                logger.info(f"Async upsert likely successful for job_id: {job_id} (no data in response object or attribute missing)")
                return True # Assuming success if no error
            else:
                # This case might indicate an issue before execution raised an exception
                logger.error(f"Async upsert failed for job_id: {job_id}. Response object was falsey: {response}")
                return False
        except Exception as e:
            logger.exception(f"Error during async upsert for job_id {job_id}: {e}")
            return False

    async def update_status(
        self,
        job_id: str,
        status: str,
        error_message: Optional[str] = None,
        results: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Updates status, error message, and results for an existing job. Async version.
        """
        # Use the client initialized in __init__
        client = self.client
        update_data = {
            "status": status,
        }
        if error_message is not None:
            update_data["error_message"] = error_message
        if results is not None:
            update_data["results"] = results # Assuming 'results' is a JSONB column

        try:
            logger.info(f"Updating status async for job_id: {job_id} to {status}. Error: {error_message is not None}. Results: {results is not None}")
            # REMOVED await and parentheses from execute() call
            response = client.table(self.TABLE_NAME)\
                .update(update_data)\
                .eq("job_id", job_id)\
                .execute()

            # Response handling logic remains similar for sync/async client v1+
            if response and hasattr(response, 'data'):
                 logger.info(f"Async update successful for job_id: {job_id}. Response data: {response.data}")
                 # Check if data is non-empty, indicating rows were matched and potentially updated
                 return len(response.data) > 0
            elif response:
                logger.info(f"Async update likely completed for job_id: {job_id} (no data in response object or attribute missing, might mean no rows matched filter)")
                # Decide if no rows matching should be True or False. False seems safer.
                return False
            else:
                logger.error(f"Async update failed for job_id: {job_id}. Response object was falsey: {response}")
                return False
        except Exception as e:
            logger.exception(f"Error updating status async for job_id {job_id}: {e}")
            return False

    async def get_multiple_results(self, job_ids: list[str]) -> list[Dict[str, Any]]:
        """
        Fetches multiple backtest result records by their job_ids.
        """
        client = self.client
        try:
            logger.info(f"Fetching backtest results for job_ids: {job_ids}")
            response = (
                client.table(self.TABLE_NAME)
                .select("*")
                .in_("job_id", job_ids)
                .execute()
            )

            if response and hasattr(response, 'data'):
                logger.info(f"Successfully fetched {len(response.data)} results for job_ids: {job_ids}")
                return response.data
            elif response:
                 logger.warning(f"Fetched results for job_ids {job_ids}, but response has no data attribute or it's empty.")
                 return []
            else:
                logger.error(f"Failed to fetch results for job_ids: {job_ids}. Response: {response}")
                return []
        except Exception as e:
            logger.exception(f"Error fetching multiple results for job_ids {job_ids}: {e}")
            return []

# Example usage (for testing purposes, usually called from elsewhere like tasks.py)
# def main(): # Needs to be sync if methods are sync
#     repo = SupabaseBacktestResultRepository()
#     job_id = "your-test-job-id"
#     strategy_id = "your-strategy-id"
    
#     # Test create/initial update
#     success = repo.create_or_update_status(
#         job_id=job_id,
#         strategy_id=strategy_id,
#         status="RUNNING",
#         start_date="2023-01-01T00:00:00Z",
#         end_date="2023-02-01T00:00:00Z",
#         initial_balance=10000.0
#     )
#     print(f"Initial upsert success: {success}")

#     # Test update (e.g., on completion)
#     results_data = {"metric1": 1.23, "trades": 50}
#     success = repo.update_status(job_id=job_id, status="COMPLETED", results=results_data)
#     print(f"Completion update success: {success}")
    
#     # Test update (e.g., on failure)
#     # success = await repo.update_status(job_id=job_id, status="FAILED", error_message="Something went wrong")
#     # print(f"Failure update success: {success}")

# if __name__ == "__main__":
#     # Requires an event loop to run async functions
#     # asyncio.run(main())
#     pass 
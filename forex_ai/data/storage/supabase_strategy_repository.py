"""
Supabase repository for LLM-enhanced Forex AI strategies.

This module provides a Supabase-based implementation of the strategy repository,
allowing storage and retrieval of strategy definitions with LLM-enhanced capabilities.
"""

import os
import json
import logging
import uuid
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import asyncio

from supabase import create_client, Client
from dotenv import load_dotenv
from pydantic import ValidationError
import pandas as pd

from forex_ai.config import settings
from forex_ai.custom_types import Strategy, CandlestickStrategy, IndicatorStrategy, PineScriptStrategy
from forex_ai.exceptions import DatabaseError, StrategyError, StrategyNotFoundError, StrategyRepositoryError

logger = logging.getLogger(__name__)

class SupabaseStrategyRepository:
    """
    Supabase implementation of the strategy repository.
    
    This class handles the storage and retrieval of strategy definitions
    in the Supabase database, including LLM-enhanced strategy components.
    """
    
    TABLE_NAME = "strategy_configurations" # Define table name constant
    OPTIMIZATION_VERSIONS_TABLE_NAME = "strategy_optimization_versions" # ADDED

    def __init__(self):
        """Initialize the Supabase repository."""
        try:
            # Load environment variables if needed
            load_dotenv()
            
            # Get Supabase credentials from environment
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
            
            if not supabase_url or not supabase_key:
                raise ValueError("Missing Supabase credentials. Please set SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables.")
            
            # Create Supabase client with updated initialization
            self.supabase = create_client(supabase_url, supabase_key)
            logger.info("Supabase strategy repository initialized")
            
        except Exception as e:
            logger.error(f"Error initializing Supabase repository: {str(e)}")
            raise DatabaseError(f"Supabase repository initialization failed: {str(e)}")

    async def create_strategy(self, strategy_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Creates the first version of a new strategy.

        Args:
            strategy_data: Dictionary containing the strategy configuration fields
                           (excluding id, version, is_latest, created_at).

        Returns:
            The created strategy dictionary including id, version=1, etc.

        Raises:
            DatabaseError: If the database operation fails.
            StrategyRepositoryError: For other repository-related errors.
        """
        try:
            # Generate a new UUID for the strategy group
            strategy_id_value = str(uuid.uuid4())
            
            # Prepare data for the first version
            db_strategy = self._prepare_strategy_for_storage(strategy_data)
            db_strategy['strategy_id'] = strategy_id_value
            db_strategy['version'] = 1
            db_strategy['is_latest'] = True
            db_strategy['created_at'] = datetime.now().isoformat()
            # 'updated_at' might not be needed if we rely on version created_at
            # Remove potentially conflicting keys if provided by caller
            db_strategy.pop('updated_at', None) 

            logger.info(f"Creating new strategy (ID: {strategy_id_value}, Version: 1)")

            response = self.supabase.table(self.TABLE_NAME).insert(db_strategy).execute()

            # Check response
            if not response.data:
                 error_detail = response.error.message if hasattr(response, 'error') and response.error else "Unknown error"
                 logger.error(f"Failed to insert new strategy {strategy_id_value}: {error_detail}")
                 raise DatabaseError(f"Strategy creation failed: {error_detail}")

            created_strategy = response.data[0]
            logger.info(f"Successfully created strategy ID: {strategy_id_value}, Version: 1")
            
            # Convert back to domain format before returning
            return self._convert_to_strategy_type(created_strategy)

        except DatabaseError as db_err:
             # Re-raise DatabaseError specifically
             raise db_err
        except Exception as e:
            logger.error(f"Unexpected error creating strategy: {e}", exc_info=True)
            # Wrap other exceptions
            raise StrategyRepositoryError(f"Failed to create strategy: {e}")
    
    async def get_all_strategies(self) -> List[Dict[str, Any]]:
        """
        Get the latest version of all strategies from the database.
        
        Returns:
            List of the latest strategy version dictionaries.
        """
        try:
            logger.info("Fetching latest versions of all strategies.")
            # Use the constant table name and filter for latest versions
            # Select specific fields needed by the frontend + origin
            response = self.supabase.table(self.TABLE_NAME)\
                .select("id, name, description, strategy_origin")\
                .eq("is_latest", True)\
                .execute()
            
            # Parse response
            strategies = response.data
            logger.info(f"Retrieved {len(strategies)} latest strategy versions.")
            
            # Convert database format to domain format (e.g., JSON parsing)
            # Note: _convert_to_strategy_type might do more than needed here.
            # If only basic fields are selected, direct return might be okay,
            # but using the converter ensures consistency if it handles nulls/types.
            return [self._convert_to_strategy_type(s) for s in strategies]
            
        except Exception as e:
            logger.error(f"Error getting all latest strategies from {self.TABLE_NAME}: {e}", exc_info=True)
            raise StrategyRepositoryError(f"Failed to get latest strategies: {e}")
    
    async def get_strategy(self, strategy_id: str, version: Optional[int] = None, exclude_columns: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """
        Get a specific version of a strategy, or the latest version if unspecified.

        Args:
            strategy_id: The ID of the strategy group to retrieve.
            version: Optional specific version number to retrieve. If None, retrieves 
                     the latest version marked with is_latest=True.
            exclude_columns: Optional list of column names to exclude from the SELECT statement.
            
        Returns:
            The strategy dictionary for the requested version or None if not found.
        """
        try:
            logger.info(f"Attempting to fetch strategy ID: {strategy_id}, Version: {version or 'latest'}, Excluding: {exclude_columns}")
            
            # --- DETAILED LOGGING ADDED ---
            logger.debug(f"[QueryBuild] Step 1: Getting table '{self.TABLE_NAME}'...")
            table_query = self.supabase.table(self.TABLE_NAME)
            logger.debug(f"[QueryBuild] Step 2: Adding explicit select for key columns...")
            # MODIFIED: Explicitly select columns instead of "*"
            DEFAULT_COLUMNS_TO_SELECT = [
                "id", "name", "description", "strategy_type", 
                "config_data", # Corrected from "parameters"
                "version", "is_latest", "created_at", "user_id", 
                "strategy_origin"
            ]
            
            columns_to_fetch = DEFAULT_COLUMNS_TO_SELECT
            if exclude_columns:
                columns_to_fetch = [col for col in DEFAULT_COLUMNS_TO_SELECT if col not in exclude_columns]
                logger.info(f"[QueryBuild] Excluding columns: {exclude_columns}. Actual columns to fetch: {columns_to_fetch}")

            # select_query = table_query.select(",".join(COLUMNS_TO_SELECT)) # Old way
            select_query = table_query.select(*columns_to_fetch) # New way: pass as varargs
            logger.debug(f"[QueryBuild] Step 3: Adding eq('id', {strategy_id})...")
            id_query = select_query.eq("id", strategy_id)
            logger.debug(f"[QueryBuild] Step 4: Checking version...")
            # --- END DETAILED LOGGING ---

            # Original query building logic
            # query = self.supabase.table(self.TABLE_NAME).select("*").eq("id", strategy_id) # Replaced by steps above

            final_query = id_query # Start with the query after filtering by ID

            if version is not None:
                # Fetch specific version
                logger.debug(f"[QueryBuild] Step 5a: Adding eq('version', {version})...") # ADDED LOG
                final_query = final_query.eq("version", version)
                logger.info(f"Querying for specific version: {version}")
            else:
                # Fetch latest version
                logger.debug(f"[QueryBuild] Step 5b: Adding eq('is_latest', True)...") # ADDED LOG
                final_query = final_query.eq("is_latest", True)
                logger.info("Querying for latest version (is_latest=True)")
            
            logger.debug("[QueryBuild] Step 6: Adding limit(1)...") # ADDED LOG
            limited_query = final_query.limit(1)

            logger.debug("[QueryBuild] Step 7: Executing query...") # ADDED LOG
            response = limited_query.execute() # Should only be one matching record
            logger.debug("[QueryBuild] Step 8: Query execution finished.") # ADDED LOG
            
            # Parse response
            strategies = response.data
            
            # --- Add Logging Here ---\
            logger.debug(f"Raw data fetched from Supabase for ID {strategy_id}, Version {version or 'latest'}: {strategies}")\
            # -----------------------\

            if not strategies:
                logger.warning(f"Strategy not found for ID: {strategy_id}, Version: {version or 'latest'}")
                return None # Changed from raising StrategyNotFoundError to returning None as per original logic
                
            # Convert database format to domain format and return the first result
            converted_strategy = self._convert_to_strategy_type(strategies[0])
            logger.info(f"Successfully retrieved strategy ID: {strategy_id}, Version: {version or 'latest'}")
            logger.debug(f"Converted strategy data being returned for ID {strategy_id}, Version {version or 'latest'}: {converted_strategy}")
            # -------------------------
            return converted_strategy
            
        except Exception as e:
            logger.error(f"Error getting strategy {strategy_id} (Version: {version or 'latest'}) from {self.TABLE_NAME}: {e}", exc_info=True)
            # Wrap exception
            raise StrategyRepositoryError(f"Failed to get strategy {strategy_id} (Version: {version or 'latest'}): {e}")
    
    async def update_strategy(self, strategy_id: str, strategy_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Creates a new version of an existing strategy.

        Args:
            strategy_id: The ID of the strategy group to update.
            strategy_data: Dictionary containing the fields to update. Fields not 
                           present will be carried over from the previous latest version.

        Returns:
            The newly created strategy version dictionary.

        Raises:
            StrategyNotFoundError: If the strategy_id is not found.
            DatabaseError: If a database operation fails.
            StrategyRepositoryError: For other repository-related errors.
        """
        try:
            # 1. Find the current latest version
            logger.info(f"Attempting to update strategy ID {strategy_id} (create new version)")
            latest_version_response = self.supabase.table(self.TABLE_NAME)\
                .select("*")\
                .eq("id", strategy_id)\
                .eq("is_latest", True)\
                .limit(1)\
                .execute()

            if not latest_version_response.data:
                logger.warning(f"Strategy {strategy_id} not found for update (no latest version).")
                raise StrategyNotFoundError(f"Strategy with ID {strategy_id} not found.")

            current_latest_version_data = latest_version_response.data[0]
            current_version_number = current_latest_version_data.get('version', 0) # Should exist
            new_version_number = current_version_number + 1
            logger.info(f"Found latest version {current_version_number} for strategy {strategy_id}.")

            # 2. Mark the old version as not latest
            update_old_response = self.supabase.table(self.TABLE_NAME)\
                .update({"is_latest": False})\
                .eq("id", strategy_id)\
                .eq("version", current_version_number)\
                .execute()

            # Check if the update was successful (optional, but good practice)
            if not update_old_response.data:
                 error_detail = update_old_response.error.message if hasattr(update_old_response, 'error') and update_old_response.error else "Unknown error updating old version"
                 logger.error(f"Failed to mark version {current_version_number} as not latest: {error_detail}")
                 # Decide whether to proceed or raise - potentially raise to avoid inconsistent state
                 raise DatabaseError(f"Failed to update latest flag on previous version {current_version_number}: {error_detail}")

            logger.info(f"Marked version {current_version_number} of strategy {strategy_id} as not latest.")
            
            # 3. Prepare new version data
            # Start with the data from the previous version
            new_version_data = current_latest_version_data.copy()
            # Update with the provided data
            new_version_data.update(strategy_data) 
            
            # Set new version metadata
            new_version_data['version'] = new_version_number
            new_version_data['is_latest'] = True
            new_version_data['created_at'] = datetime.now().isoformat()
            # Remove fields that shouldn't be copied or are set anew
            new_version_data.pop('updated_at', None) # Rely on created_at for version timestamp

            # Prepare for storage (JSON dumps etc.)
            db_insert_data = self._prepare_strategy_for_storage(new_version_data)
            # Ensure crucial fields aren't lost during preparation if _prepare... modifies inplace or returns subset
            db_insert_data['id'] = strategy_id 
            db_insert_data['version'] = new_version_number
            db_insert_data['is_latest'] = True
            db_insert_data['created_at'] = new_version_data['created_at'] # Ensure timestamp is preserved

            logger.info(f"Inserting new version {new_version_number} for strategy {strategy_id}.")

            # 4. Insert the new version
            insert_new_response = self.supabase.table(self.TABLE_NAME).insert(db_insert_data).execute()

            if not insert_new_response.data:
                 error_detail = insert_new_response.error.message if hasattr(insert_new_response, 'error') and insert_new_response.error else "Unknown error inserting new version"
                 logger.error(f"Failed to insert new version {new_version_number} for strategy {strategy_id}: {error_detail}")
                 # TODO: Consider rolling back the 'is_latest=False' update on the previous version if possible/needed
                 raise DatabaseError(f"Failed to insert new strategy version {new_version_number}: {error_detail}")

            created_version = insert_new_response.data[0]
            logger.info(f"Successfully created version {new_version_number} for strategy {strategy_id}.")
            
            # Convert back to domain format
            return self._convert_to_strategy_type(created_version)

        except StrategyNotFoundError:
            # Re-raise specifically
            raise
        except DatabaseError as db_err:
            # Re-raise DatabaseError specifically
             raise db_err
        except Exception as e:
            logger.error(f"Unexpected error updating strategy {strategy_id}: {e}", exc_info=True)
            raise StrategyRepositoryError(f"Failed to update strategy {strategy_id}: {e}")
    
    async def delete_strategy(self, strategy_id: str) -> bool:
        """
        Delete ALL versions of a strategy from the database.
        
        Args:
            strategy_id: The ID of the strategy group to delete.
            
        Returns:
            True if at least one version was deleted, False otherwise.
        """
        try:
            logger.info(f"Attempting to delete all versions for strategy ID: {strategy_id}")
            # Delete all rows matching the strategy ID, regardless of version
            response = self.supabase.table(self.TABLE_NAME).delete().eq("id", strategy_id).execute()
            
            # Parse response
            deleted_strategies = response.data # Response typically contains the deleted records
            
            if not deleted_strategies:
                 # Check if the strategy ID even existed
                 count_response = self.supabase.table(self.TABLE_NAME).select("id", count="exact").eq("id", strategy_id).execute()
                 if count_response.count == 0:
                     logger.warning(f"Strategy {strategy_id} not found for deletion.")
                     # Return False or raise StrategyNotFoundError depending on desired behavior
                     # Returning False aligns with original behavior if nothing was deleted
                     return False 
                 else:
                     # Strategy existed but delete failed for some reason?
                     error_detail = response.error.message if hasattr(response, 'error') and response.error else "Unknown delete error"
                     logger.error(f"Delete operation failed for strategy {strategy_id}, though it exists. Error: {error_detail}")
                     raise DatabaseError(f"Failed to delete strategy {strategy_id}: {error_detail}")
            
            num_deleted = len(deleted_strategies)
            logger.info(f"Successfully deleted {num_deleted} version(s) for strategy ID: {strategy_id}")
            # Return True if any strategies (versions) were deleted
            return num_deleted > 0
            
        except Exception as e:
            logger.error(f"Error deleting strategy {strategy_id} from {self.TABLE_NAME}: {e}", exc_info=True)
            raise StrategyRepositoryError(f"Failed to delete strategy {strategy_id}: {e}")
    
    async def update_status(self, strategy_id: str, status: str) -> Dict[str, Any]:
        """
        Update the status of a strategy.
        
        Args:
            strategy_id: The ID of the strategy to update
            status: The new status
            
        Returns:
            The updated strategy
        """
        try:
            # Use the constant table name
            response = self.supabase.table(self.TABLE_NAME).update({"status": status, "updated_at": datetime.now().isoformat()}).eq("id", strategy_id).execute()
            
            # Parse response
            updated_strategies = response.data
            
            if not updated_strategies:
                raise StrategyNotFoundError(f"Strategy with ID {strategy_id} not found")
                
            # Convert database format to domain format and return the first result
            return self._convert_to_strategy_type(updated_strategies[0])
            
        except Exception as e:
            logger.error(f"Error updating strategy status {strategy_id} in {self.TABLE_NAME}: {str(e)}")
            raise DatabaseError(f"Failed to update strategy status: {str(e)}")
    
    async def get_strategy_versions(self, strategy_id: str) -> List[Dict[str, Any]]:
        """
        Lists available versions for a given strategy ID.

        Args:
            strategy_id: The ID of the strategy group.

        Returns:
            A list of dictionaries, each containing 'version' and 'created_at'.
            Returns an empty list if the strategy_id is not found.

        Raises:
            StrategyRepositoryError: If the database operation fails.
        """
        try:
            logger.info(f"Listing versions for strategy ID: {strategy_id}")
            # Select only version and created_at, order by version descending
            response = self.supabase.table(self.TABLE_NAME)\
                .select("version, created_at")\
                .eq("id", strategy_id)\
                .order("version", desc=True)\
                .execute()
            
            versions = response.data
            
            if versions is None:
                # Handle cases where Supabase might return None on error (though usually raises)
                logger.warning(f"Received None response when listing versions for {strategy_id}")
                return [] 
            
            logger.info(f"Found {len(versions)} versions for strategy ID: {strategy_id}")
            # Format the output if needed, though selecting specific columns already helps
            # Example formatting (if needed):
            # formatted_versions = [
            #     {"version": v["version"], "created_at": v["created_at"]}
            #     for v in versions
            # ]
            # return formatted_versions
            return versions # Return the list directly

        except Exception as e:
            logger.error(f"Error listing versions for strategy {strategy_id}: {e}", exc_info=True)
            raise StrategyRepositoryError(f"Failed to list versions for strategy {strategy_id}: {e}")
    
    async def get_strategies_by_type(self, strategy_type: str) -> List[Dict[str, Any]]:
        """
        Get strategies by type.
        
        Args:
            strategy_type: The type of strategies to retrieve
            
        Returns:
            List of strategy dictionaries
        """
        try:
            # Use the constant table name
            response = self.supabase.table(self.TABLE_NAME).select("*").eq("strategy_type", strategy_type).execute()
            
            # Parse response
            strategies = response.data
            
            # Convert database format to domain format
            return [self._convert_to_strategy_type(s) for s in strategies]
            
        except Exception as e:
            logger.error(f"Error getting strategies by type {strategy_type} from {self.TABLE_NAME}: {str(e)}")
            raise DatabaseError(f"Failed to get strategies by type: {str(e)}")
    
    async def get_active_strategies(self) -> List[Dict[str, Any]]:
        """
        Get all active strategies.
        
        Returns:
            List of active strategy dictionaries
        """
        try:
            # Use the constant table name
            response = self.supabase.table(self.TABLE_NAME).select("*").eq("status", "active").execute()
            
            # Parse response
            strategies = response.data
            
            # Convert database format to domain format
            return [self._convert_to_strategy_type(s) for s in strategies]
            
        except Exception as e:
            logger.error(f"Error getting active strategies from {self.TABLE_NAME}: {str(e)}")
            raise DatabaseError(f"Failed to get active strategies: {str(e)}")
    
    def _prepare_strategy_for_storage(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare a strategy for storage in the database.
        
        This method handles serialization of complex fields and ensures
        all required fields are present.
        
        Args:
            strategy: The strategy to prepare
            
        Returns:
            Dictionary ready for database storage
        """
        db_strategy = strategy.copy()

        # Explicitly remove top-level keys that should only exist within config_data 
        # or are handled by specific columns, to prevent schema cache conflicts.
        keys_to_remove_if_top_level = [
            "entry_conditions", "exit_conditions", 
            "parameters", "risk_management", "indicators",
            "trade_rules",
            "configuration",
            "success"  # Added to prevent schema error if it appears at top level
        ]
        for key_to_remove in keys_to_remove_if_top_level:
            if key_to_remove in db_strategy:
                logger.debug(f"Removing top-level key '{key_to_remove}' from strategy data before DB storage to avoid schema conflict.")
                del db_strategy[key_to_remove]
        
        # Serialize complex fields that are actual columns or within config_data (original logic)
        # Note: The original list here might be for fields *within* config_data that need serialization, 
        # but if they are actual column names that expect JSON strings, this is fine.
        # For now, assuming these are columns or handled correctly by Supabase if they are part of config_data.
        json_fields_for_direct_columns = [] # If 'parameters', 'risk_management' etc. are direct JSON columns, add them here.
        
        # If 'config_data' itself is a JSONB column and contains these fields, 
        # ensure 'config_data' is properly dumped to string if it's a dict.
        if 'config_data' in db_strategy and isinstance(db_strategy['config_data'], dict):
            logger.debug("Serializing 'config_data' field to JSON string for storage.")
            # Before dumping config_data, ensure its nested complex fields are also strings if necessary
            # This depends on how your LLM/agent generates these nested structures.
            # For now, assuming config_data is prepared correctly by the caller if it's a dict.
            db_strategy['config_data'] = json.dumps(db_strategy['config_data'])
        elif 'config_data' in db_strategy and db_strategy['config_data'] is None:
            # Ensure NULL is passed correctly if config_data is explicitly None
            pass # Supabase client should handle None as NULL

        # Serialize arrays as comma-separated strings (original logic - for specific direct columns)
        for field in ["timeframes", "pairs"]:
            if field in db_strategy and db_strategy[field] is not None and isinstance(db_strategy[field], list):
                db_strategy[field] = ",".join(db_strategy[field])
        
        return db_strategy
    
    def _convert_to_strategy_type(self, db_strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a database strategy to the appropriate strategy type.
        
        Args:
            db_strategy: The database strategy record
            
        Returns:
            Strategy dictionary with proper types
        """
        strategy = db_strategy.copy()
        
        # Deserialize JSON fields
        for field in ["entry_conditions", "exit_conditions", "parameters", "risk_management", "config_data"]:
            if field in strategy and strategy[field] is not None and isinstance(strategy[field], str):
                try:
                    strategy[field] = json.loads(strategy[field])
                except json.JSONDecodeError:
                    strategy[field] = {}
        
        # Deserialize array fields
        for field in ["timeframes", "pairs"]:
            if field in strategy and strategy[field] is not None and isinstance(strategy[field], str):
                strategy[field] = strategy[field].split(",")
        
        return strategy 

    # --- Backtest Log Methods ---
    
    async def add_backtest_log(
        self,
        job_id: str,
        level: str,
        message: str,
        source: Optional[str] = None
    ) -> bool:
        """
        Adds a log entry for a specific backtest job. (DB INSERTION DISABLED)

        Args:
            job_id: The UUID of the backtest job.
            level: Log level (e.g., INFO, DEBUG, TRADE).
            message: The log message.
            source: Optional source identifier.

        Returns:
            True always, as DB insertion is skipped.
        Raises:
            DatabaseError: If the database operation fails.
        """
        # --- DB INSERTION DISABLED FOR DEBUGGING --- 
        log_msg_short = message[:100] + ('...' if len(message) > 100 else '')
        logger.debug(f"[DB Log Suppressed] Job: {job_id}, Level: {level.upper()}, Source: {source}, Msg: {log_msg_short}")
        return True # Always return True to avoid breaking callers
        # --- Original Code Below ---
        # try:
        #     log_entry = {
        #         "job_id": job_id,
        #         "level": level.upper(),
        #         "message": message,
        #         "timestamp": datetime.now().isoformat(),
        #         "source": source
        #     }
        #     # Remove source if None
        #     if source is None:
        #         del log_entry["source"]

        #     response = self.supabase.table("backtest_logs").insert(log_entry).execute()
            
        #     if not response.data:
        #         error_detail = response.error.message if hasattr(response, 'error') and response.error else "Unknown log insert error"
        #         logger.error(f"Failed to insert backtest log for job {job_id}: {error_detail}")
        #         # Return False on failure, task might continue with warnings
        #         return False 
        #         # Optionally raise DatabaseError(f"Failed to insert log: {error_detail}")
            
        #     # logger.debug(f"Successfully added log for job {job_id}: {level} - {message[:50]}...")
        #     return True
        # except Exception as e:
        #     # Log error but potentially allow task to continue
        #     logger.error(f"Error adding backtest log for job {job_id}: {e}", exc_info=True)
        #     # Depending on severity, you might re-raise as DatabaseError or just return False
        #     # raise DatabaseError(f"Failed to add backtest log: {e}")
        #     return False

    async def get_backtest_logs(self, job_id: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Retrieves log entries for a specific backtest job.

        Args:
            job_id: The UUID of the backtest job.
            limit: Maximum number of log entries to retrieve.

        Returns:
            A list of log entry dictionaries, ordered by timestamp ascending.
            Returns an empty list if no logs are found.
        Raises:
            StrategyRepositoryError: If the database operation fails.
        """
        try:
            logger.info(f"Fetching logs for backtest job ID: {job_id} (limit: {limit})")
            response = self.supabase.table("backtest_logs")\
                .select("*")\
                .eq("job_id", job_id)\
                .order("timestamp", desc=False) \
                .limit(limit) \
                .execute()
            
            logs = response.data
            if logs is None:
                logger.warning(f"Received None response when fetching logs for job {job_id}")
                return []
                
            logger.info(f"Retrieved {len(logs)} log entries for job {job_id}.")
            return logs
        except Exception as e:
            logger.error(f"Error fetching backtest logs for job {job_id}: {e}", exc_info=True)
            raise StrategyRepositoryError(f"Failed to fetch backtest logs for job {job_id}: {e}")

    # --- End Backtest Log Methods ---

    # --- Backtest Configuration Methods ---
    CONFIG_TABLE_NAME = "backtest_configurations" # Constant for the new table

    async def save_backtest_config(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Saves a new backtest configuration setup.

        Args:
            config_data: Dictionary containing the configuration details 
                         (name, strategy_id, strategy_version, instrument, 
                         timeframe, start_date, end_date, initial_balance, 
                         parameter_overrides, created_by).

        Returns:
            The saved configuration dictionary including its generated config_id.

        Raises:
            DatabaseError: If the database operation fails.
            StrategyRepositoryError: For other repository-related errors.
        """
        try:
            # Basic validation (can be enhanced)
            required_fields = ["name", "strategy_id", "strategy_version", "instrument", 
                               "timeframe", "start_date", "end_date", "initial_balance"]
            if not all(field in config_data for field in required_fields):
                missing = [f for f in required_fields if f not in config_data]
                raise ValueError(f"Missing required fields for backtest configuration: {missing}")

            # Ensure dates are in correct format (isoformat strings expected)
            config_data['start_date'] = pd.to_datetime(config_data['start_date']).isoformat()
            config_data['end_date'] = pd.to_datetime(config_data['end_date']).isoformat()
            # Ensure balance is float/numeric
            config_data['initial_balance'] = float(config_data['initial_balance'])

            # Serialize parameter_overrides if it's not None
            if 'parameter_overrides' in config_data and config_data['parameter_overrides'] is not None:
                 config_data['parameter_overrides'] = json.dumps(config_data['parameter_overrides'])
            else:
                 config_data['parameter_overrides'] = None # Ensure it's null in DB if not provided

            # created_at is handled by DB default
            # config_id is handled by DB default
            config_data.pop('config_id', None) # Remove if accidentally provided
            config_data.pop('created_at', None) # Remove if accidentally provided

            logger.info(f"Saving new backtest configuration: {config_data.get('name')}")
            response = self.supabase.table(self.CONFIG_TABLE_NAME).insert(config_data).execute()

            if not response.data:
                error_detail = response.error.message if hasattr(response, 'error') and response.error else "Unknown insert error"
                logger.error(f"Failed to save backtest configuration: {error_detail}")
                raise DatabaseError(f"Backtest configuration save failed: {error_detail}")

            saved_config = response.data[0]
            logger.info(f"Successfully saved backtest configuration ID: {saved_config.get('config_id')}")
            
            # Deserialize JSONB before returning if needed
            if saved_config.get('parameter_overrides') and isinstance(saved_config['parameter_overrides'], str):
                 try:
                     saved_config['parameter_overrides'] = json.loads(saved_config['parameter_overrides'])
                 except json.JSONDecodeError:
                      logger.warning(f"Failed to deserialize parameter_overrides for config {saved_config.get('config_id')}")
                      saved_config['parameter_overrides'] = {}
            
            return saved_config

        except (ValueError, TypeError) as val_err:
            logger.error(f"Validation error saving backtest config: {val_err}", exc_info=True)
            raise StrategyRepositoryError(f"Invalid data for backtest configuration: {val_err}")
        except DatabaseError as db_err:
            raise db_err
        except Exception as e:
            logger.error(f"Unexpected error saving backtest config: {e}", exc_info=True)
            raise StrategyRepositoryError(f"Failed to save backtest configuration: {e}")

    async def get_backtest_config(self, config_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a specific backtest configuration by its ID.

        Args:
            config_id: The UUID of the configuration to retrieve.

        Returns:
            The configuration dictionary or None if not found.
        Raises:
            StrategyRepositoryError: If the database operation fails.
        """
        try:
            logger.info(f"Fetching backtest configuration ID: {config_id}")
            response = self.supabase.table(self.CONFIG_TABLE_NAME)\
                .select("*")\
                .eq("config_id", config_id)\
                .limit(1)\
                .execute()
            
            config = response.data
            if not config:
                logger.warning(f"Backtest configuration not found: {config_id}")
                return None
                
            retrieved_config = config[0]
             # Deserialize JSONB before returning
            if retrieved_config.get('parameter_overrides') and isinstance(retrieved_config['parameter_overrides'], str):
                 try:
                     retrieved_config['parameter_overrides'] = json.loads(retrieved_config['parameter_overrides'])
                 except json.JSONDecodeError:
                      logger.warning(f"Failed to deserialize parameter_overrides for config {config_id}")
                      retrieved_config['parameter_overrides'] = {}
            
            logger.info(f"Successfully retrieved backtest configuration: {config_id}")
            return retrieved_config
        except Exception as e:
            logger.error(f"Error fetching backtest configuration {config_id}: {e}", exc_info=True)
            raise StrategyRepositoryError(f"Failed to fetch backtest configuration {config_id}: {e}")

    async def list_backtest_configs(
        self, 
        user_id: Optional[str] = None, 
        strategy_id: Optional[str] = None, 
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Lists saved backtest configurations, optionally filtering by user or strategy.

        Args:
            user_id: Optional user UUID to filter by.
            strategy_id: Optional strategy group UUID to filter by.
            limit: Maximum number of configurations to return.

        Returns:
            A list of backtest configuration dictionaries, ordered by creation date descending.
        Raises:
            StrategyRepositoryError: If the database operation fails.
        """
        try:
            query = self.supabase.table(self.CONFIG_TABLE_NAME)\
                .select("*")\
                .order("created_at", desc=True)\
                .limit(limit)
            
            if user_id:
                query = query.eq("created_by", user_id)
                logger.info(f"Listing backtest configs for user {user_id} (limit {limit})")
            elif strategy_id:
                query = query.eq("strategy_id", strategy_id)
                logger.info(f"Listing backtest configs for strategy {strategy_id} (limit {limit})")
            else:
                 logger.info(f"Listing all backtest configs (limit {limit})")

            response = query.execute()
            configs = response.data
            
            if configs is None:
                 logger.warning("Received None response when listing backtest configs.")
                 return []
            
            # Deserialize JSONB for each config
            for config in configs:
                if config.get('parameter_overrides') and isinstance(config['parameter_overrides'], str):
                    try:
                        config['parameter_overrides'] = json.loads(config['parameter_overrides'])
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to deserialize parameter_overrides for config {config.get('config_id')}")
                        config['parameter_overrides'] = {}
                        
            logger.info(f"Retrieved {len(configs)} backtest configurations.")
            return configs
            
        except Exception as e:
            logger.error(f"Error listing backtest configurations: {e}", exc_info=True)
            raise StrategyRepositoryError(f"Failed to list backtest configurations: {e}")

    async def delete_backtest_config(self, config_id: str) -> bool:
        """
        Deletes a specific backtest configuration.

        Args:
            config_id: The UUID of the configuration to delete.

        Returns:
            True if deletion was successful, False if not found.
        Raises:
            StrategyRepositoryError: If the database operation fails.
        """
        try:
            logger.info(f"Attempting to delete backtest configuration ID: {config_id}")
            response = self.supabase.table(self.CONFIG_TABLE_NAME)\
                .delete()\
                .eq("config_id", config_id)\
                .execute()
                
            deleted_configs = response.data
            if not deleted_configs:
                 # Check if it existed
                 count_response = self.supabase.table(self.CONFIG_TABLE_NAME).select("config_id", count="exact").eq("config_id", config_id).execute()
                 if count_response.count == 0:
                     logger.warning(f"Backtest configuration {config_id} not found for deletion.")
                     return False
                 else:
                     error_detail = response.error.message if hasattr(response, 'error') and response.error else "Unknown delete error"
                     logger.error(f"Delete operation failed for backtest config {config_id}: {error_detail}")
                     raise DatabaseError(f"Failed to delete backtest config {config_id}: {error_detail}")

            logger.info(f"Successfully deleted backtest configuration ID: {config_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting backtest configuration {config_id}: {e}", exc_info=True)
            raise StrategyRepositoryError(f"Failed to delete backtest configuration {config_id}: {e}")
            
    # --- End Backtest Configuration Methods ---

    async def insert_optimization_version(
        self,
        strategy_id: str,
        version_number: int,
        name: str,
        strategy_type: str,
        description: Optional[str],
        config_data: Dict[str, Any],
        comment: Optional[str] = None,
        created_by: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Saves a new optimization version for a strategy to the strategy_optimization_versions table.
        Includes name, strategy_type, and description.
        """
        try:
            logger.info(
                f"Inserting optimization version {version_number} for strategy {strategy_id}. "
                f"Name: {name}, Type: {strategy_type}, Created by: {created_by}"
            )
            
            db_record = {
                "strategy_id": strategy_id,
                "version": version_number,
                "name": name,
                "strategy_type": strategy_type,
                "description": description,
                "config_data": json.dumps(config_data) if isinstance(config_data, dict) else config_data,
                "comment": comment,
                "created_by": created_by,
                "created_at": datetime.now().isoformat()
            }

            logger.debug(f"Record to insert into {self.OPTIMIZATION_VERSIONS_TABLE_NAME}: {db_record}")

            response = self.supabase.table(self.OPTIMIZATION_VERSIONS_TABLE_NAME)\
                .insert(db_record)\
                .execute()

            if response.data:
                saved_data = response.data[0]
                logger.info(f"Successfully inserted optimization version. ID: {saved_data.get('id')}, Strategy ID: {strategy_id}, Version: {version_number}")
                return saved_data
            else:
                error_message = "No data returned from insert operation."
                if hasattr(response, 'error') and response.error:
                    error_message = response.error.message
                logger.error(f"Failed to insert optimization version for strategy {strategy_id}, version {version_number}: {error_message}")
                logger.error(f"Request was: table='{self.OPTIMIZATION_VERSIONS_TABLE_NAME}', data={db_record}")
                if hasattr(response, 'status_code'):
                     logger.error(f"Response status code: {response.status_code}")
                if hasattr(response, 'json') and callable(response.json):
                    try:
                        logger.error(f"Full response JSON: {response.json()}")
                    except Exception as e_json:
                        logger.error(f"Could not parse response JSON: {e_json}")
                elif hasattr(response, 'text'):
                    logger.error(f"Full response text: {response.text}")
                
                raise StrategyRepositoryError(f"Failed to insert optimization version: {error_message}")

        except StrategyRepositoryError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error inserting optimization version for strategy {strategy_id}, version {version_number}: {e}", exc_info=True)
            raise StrategyRepositoryError(f"Unexpected error during optimization version insert: {e}")

    logger.info("DEBUG_MARKER_V2: SupabaseStrategyRepository class definition includes get_strategy_optimization_config.") # Adding a unique marker

    async def get_strategy_optimization_config(
        self,
        base_strategy_id: str,
        optimization_version_number: int
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieves a specific optimization version's configuration directly.
        This method now assumes the strategy_id is the base_strategy_id
        and optimization_version_number is the specific integer version for that optimization.
        COLUMN IN DB IS NOW 'version'.
        """
        try:
            logger.info(
                f"Attempting to fetch optimization config for base_strategy_id: {base_strategy_id}, "
                f"optimization_version_number (maps to 'version' column): {optimization_version_number}"
            )
            
            # Log the exact table name and parameters just before the query
            logger.debug(f"Executing Supabase query on table: '{self.OPTIMIZATION_VERSIONS_TABLE_NAME}' with params: strategy_id='{base_strategy_id}', version='{optimization_version_number}'")

            query = (
                self.supabase.table(self.OPTIMIZATION_VERSIONS_TABLE_NAME)
                .select("*")\
                .eq("strategy_id", base_strategy_id)\
                .eq("version", optimization_version_number)\
                .limit(1)\
                .execute()
            )

            if query.data:
                config_record = query.data[0]
                # Deserialize config_data if it's a string
                if 'config_data' in config_record and isinstance(config_record['config_data'], str):
                    try:
                        config_record['config_data'] = json.loads(config_record['config_data'])
                    except json.JSONDecodeError as e:
                        logger.error(f"Error deserializing config_data for strategy {base_strategy_id} version {optimization_version_number}: {e}")
                        # Return raw string or handle as error
                
                # The column is now 'version', so no remapping needed like:
                # if "version_number" in config_record:
                #     config_record["version"] = config_record["version_number"]

                logger.info(f"Successfully retrieved optimization config for strategy ID: {base_strategy_id}, version: {optimization_version_number}")
                return config_record
            else:
                logger.warning(f"Optimization config not found for strategy ID: {base_strategy_id}, version: {optimization_version_number}")
                return None
        except Exception as e:
            logger.error(f"Error fetching optimization config for strategy {base_strategy_id} version {optimization_version_number}: {e}", exc_info=True)
            raise StrategyRepositoryError(f"Failed to get optimization config for strategy {base_strategy_id} version {optimization_version_number}: {e}")

    async def get_latest_optimization_version(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the latest optimization version for a given strategy_id.
        Latest is determined by the highest 'version'.
        The returned dictionary will have a 'version' key.
        """
        try:
            logger.info(f"Fetching latest optimization version for strategy_id: {strategy_id} from {self.OPTIMIZATION_VERSIONS_TABLE_NAME}")
            response = self.supabase.table(self.OPTIMIZATION_VERSIONS_TABLE_NAME)\
                .select("*")\
                .eq("strategy_id", strategy_id)\
                .order("version", desc=True)\
                .limit(1)\
                .execute()

            if response.data:
                latest_version_db_record = response.data[0]
                # The column is already 'version'.
                latest_version_domain_record = latest_version_db_record.copy()
                # if "version_number" in latest_version_domain_record: # REMOVED
                #     latest_version_domain_record["version"] = latest_version_domain_record["version_number"] # REMOVED
                
                logger.info(f"Found latest optimization version {latest_version_domain_record.get('version')} for strategy {strategy_id}.")
                
                # Deserialize config_data if it's a string
                if 'config_data' in latest_version_domain_record and isinstance(latest_version_domain_record['config_data'], str):
                    try:
                        latest_version_domain_record['config_data'] = json.loads(latest_version_domain_record['config_data'])
                    except json.JSONDecodeError as e:
                        logger.error(f"Error deserializing config_data for latest opt version of strategy {strategy_id}: {e}")
                        # Retain raw string if deserialization fails
                return latest_version_domain_record
            else:
                logger.warning(f"No optimization versions found for strategy_id: {strategy_id} in {self.OPTIMIZATION_VERSIONS_TABLE_NAME}")
                return None
        except Exception as e:
            logger.error(f"Error fetching latest optimization version for strategy {strategy_id}: {e}", exc_info=True)
            raise StrategyRepositoryError(f"Failed to get latest optimization version for strategy {strategy_id}: {e}")

    # --- ADDED BY ASSISTANT: Method to upsert a strategy into the main table ---
    async def upsert_strategy(self, strategy_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Upserts a strategy into the main strategy table.

        Args:
            strategy_data: The strategy data to upsert.

        Returns:
            The upserted strategy dictionary.

        Raises:
            DatabaseError: If the database operation fails.
            StrategyRepositoryError: For other repository-related errors.
        """
        try:
            # Generate a new UUID for the strategy group
            strategy_id_value = str(uuid.uuid4())
            
            # Prepare data for the first version
            db_strategy = self._prepare_strategy_for_storage(strategy_data)
            db_strategy['strategy_id'] = strategy_id_value
            db_strategy['version'] = 1
            db_strategy['is_latest'] = True
            db_strategy['created_at'] = datetime.now().isoformat()
            # 'updated_at' might not be needed if we rely on version created_at
            # Remove potentially conflicting keys if provided by caller
            db_strategy.pop('updated_at', None) 

            logger.info(f"Upserting strategy (ID: {strategy_id_value}, Version: 1)")

            response = self.supabase.table(self.TABLE_NAME).upsert(db_strategy).execute()

            # Check response
            if not response.data:
                 error_detail = response.error.message if hasattr(response, 'error') and response.error else "Unknown error"
                 logger.error(f"Failed to upsert strategy {strategy_id_value}: {error_detail}")
                 raise DatabaseError(f"Strategy upsert failed: {error_detail}")

            upserted_strategy = response.data[0]
            logger.info(f"Successfully upserted strategy ID: {strategy_id_value}, Version: 1")
            
            # Convert back to domain format before returning
            return self._convert_to_strategy_type(upserted_strategy)

        except DatabaseError as db_err:
             # Re-raise DatabaseError specifically
             raise db_err
        except Exception as e:
            logger.error(f"Unexpected error upserting strategy: {e}", exc_info=True)
            raise StrategyRepositoryError(f"Failed to upsert strategy: {e}") 

    async def get_latest_optimization_version_number(self, strategy_id: str) -> Optional[int]:
        logger.debug(f"Attempting to get latest optimization version identifier (column 'version') for strategy_id: {strategy_id} from {self.OPTIMIZATION_VERSIONS_TABLE_NAME}")
        try:
            query = (
                self.supabase.table(self.OPTIMIZATION_VERSIONS_TABLE_NAME)
                .select("version, strategy_id, id, created_at")
                .eq("strategy_id", strategy_id)
                .order("version", desc=True)
                .limit(1)
            )
            # Attempt to log query URL - this is illustrative, supabase-py might not directly expose URL like this
            # logger.debug(f"Constructed query for latest opt version: {query.url}") # This line might error if .url is not an attribute

            response = query.execute() # Assuming execute() is awaitable if client is async, or handled by supabase-py

            # Log the full Supabase response object
            try:
                response_dict = {
                    "data": response.data,
                    "status_code": response.status_code if hasattr(response, 'status_code') else 'N/A',
                    "error": response.error.message if hasattr(response, 'error') and response.error and hasattr(response.error, 'message') else (response.error if hasattr(response, 'error') else 'N/A'),
                    "count": response.count if hasattr(response, 'count') else 'N/A'
                }
                logger.debug(f"Full Supabase response for get_latest_optimization_version_identifier (strategy_id: {strategy_id}): {response_dict}")
            except Exception as log_e:
                logger.error(f"Error logging full Supabase response for get_latest_opt_version: {log_e}")

            if response.data:
                logger.debug(f"Found data for latest opt version (strategy_id: {strategy_id}): {response.data[0]}")
                return response.data[0]["version"]
            else:
                error_message = response.error.message if hasattr(response, 'error') and response.error and hasattr(response.error, 'message') else (response.error if hasattr(response, 'error') else "No error object")
                logger.warning(f"No optimization versions found for strategy_id: {strategy_id} in {self.OPTIMIZATION_VERSIONS_TABLE_NAME}. Response error (if any): {error_message}. Full response logged at DEBUG.")
                return None
        except Exception as e:
            logger.error(f"Exception fetching latest optimization version for {strategy_id}: {e}", exc_info=True)
            return None 
"""
Supabase client for the Forex AI Trading System.

This module provides database connectivity and operations using Supabase.
"""

import logging
import json
from typing import Dict, List, Optional, Union, Any, Tuple
from functools import lru_cache
from contextlib import contextmanager

from supabase import create_client, Client

from forex_ai.config.settings import get_settings
from forex_ai.exceptions import DatabaseError, DatabaseConnectionError
from forex_ai.auth.supabase import get_supabase_client

logger = logging.getLogger(__name__)

class SupabaseClient:
    """
    Client for Supabase database operations.
    
    This class provides methods for performing various data operations
    using the Supabase API.
    """
    
    def __init__(self):
        """
        Initialize the Supabase client.
        """
        try:
            self.client = get_supabase_client()
            logger.info("Supabase client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {str(e)}")
            raise DatabaseConnectionError(f"Failed to connect to Supabase: {str(e)}")
    
    def _handle_error(self, e: Exception, operation: str) -> None:
        """
        Handle database errors.
        
        Args:
            e: Exception that was raised.
            operation: Operation that caused the error.
            
        Raises:
            DatabaseError: Always raised with appropriate context.
        """
        error_msg = f"Supabase {operation} error: {str(e)}"
        logger.error(error_msg)
        raise DatabaseError(error_msg)
    
    def execute_rpc(
        self,
        function_name: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a remote procedure call (RPC) function in Supabase.
        
        Args:
            function_name: Name of the stored procedure to call.
            params: Parameters to pass to the function.
            
        Returns:
            Function result.
            
        Raises:
            DatabaseError: If executing the RPC fails.
        """
        try:
            result = self.client.rpc(function_name, params or {}).execute()
            return result.data
        except Exception as e:
            self._handle_error(e, f"RPC '{function_name}'")
    
    def fetch_all(
        self,
        table: str,
        columns: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch all results from a table.
        
        Args:
            table: Table name.
            columns: Columns to select.
            where: Filter conditions.
            order_by: Order by clause.
            limit: Maximum number of results.
            offset: Number of results to skip.
            
        Returns:
            Query results.
            
        Raises:
            DatabaseError: If fetching fails.
        """
        try:
            query = self.client.table(table).select(",".join(columns) if columns else "*")
            
            # Apply filters if provided
            if where:
                for key, value in where.items():
                    if isinstance(value, dict) and "operator" in value:
                        query = query.filter(key, value["operator"], value["value"])
                    else:
                        query = query.eq(key, value)
            
            # Apply sorting if provided
            if order_by:
                # Handle both ascending and descending order
                if order_by.startswith("-"):
                    query = query.order(order_by[1:], desc=True)
                else:
                    query = query.order(order_by)
                    
            # Apply pagination if provided
            if limit is not None:
                query = query.limit(limit)
                
            if offset is not None:
                query = query.offset(offset)
                
            result = query.execute()
            return result.data
        except Exception as e:
            self._handle_error(e, f"fetch_all from '{table}'")
    
    def fetch_one(
        self,
        table: str,
        where: Dict[str, Any],
        columns: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch a single result from a table.
        
        Args:
            table: Table name.
            where: Filter conditions.
            columns: Columns to select.
            
        Returns:
            Query result or None if no result was found.
            
        Raises:
            DatabaseError: If fetching fails.
        """
        try:
            results = self.fetch_all(table, columns, where, limit=1)
            return results[0] if results else None
        except Exception as e:
            self._handle_error(e, f"fetch_one from '{table}'")
    
    def insert_one(
        self,
        table: str,
        data: Dict[str, Any],
        return_id: bool = True,
    ) -> Optional[Any]:
        """
        Insert a single row into a table.
        
        Args:
            table: Table name.
            data: Row data.
            return_id: Whether to return the inserted row's ID.
            
        Returns:
            The inserted row or ID if return_id is True, otherwise None.
            
        Raises:
            DatabaseError: If inserting fails.
        """
        try:
            result = self.client.table(table).insert(data).execute()
            if return_id and result.data:
                return result.data[0].get("id")
            return result.data[0] if result.data else None
        except Exception as e:
            self._handle_error(e, f"insert_one into '{table}'")
    
    def insert_many(
        self,
        table: str,
        data: List[Dict[str, Any]],
        return_ids: bool = False,
    ) -> Optional[List[Any]]:
        """
        Insert multiple rows into a table.
        
        Args:
            table: Table name.
            data: Row data.
            return_ids: Whether to return the inserted rows' IDs.
            
        Returns:
            The inserted rows or IDs if return_ids is True, otherwise None.
            
        Raises:
            DatabaseError: If inserting fails.
        """
        if not data:
            return []
            
        try:
            result = self.client.table(table).insert(data).execute()
            if return_ids and result.data:
                return [row.get("id") for row in result.data]
            return result.data if result.data else []
        except Exception as e:
            self._handle_error(e, f"insert_many into '{table}'")
    
    def upsert(
        self,
        table: str,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        on_conflict: Optional[str] = None, # Column name(s) for conflict resolution
        ignore_duplicates: bool = False,
        returning: str = 'representation' # 'minimal' or 'representation'
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Insert or update rows in a table (upsert).
        Now Synchronous.

        Args:
            table: Table name.
            data: Row data or list of row data dictionaries.
            on_conflict: Comma-separated string of column names to specify the conflict target.
                         Required for true upsert behavior (update on conflict).
                         If None, it behaves like insert with potential duplicate errors unless ignore_duplicates=True.
            ignore_duplicates: If True and on_conflict is None, duplicate rows are ignored.
            returning: 'minimal' returns nothing, 'representation' returns the upserted rows.

        Returns:
            List of upserted rows if returning='representation', otherwise None.

        Raises:
            DatabaseError: If upserting fails.
        """
        try:
            # Build and execute the upsert query synchronously
            result = self.client.table(table).upsert(
                data,
                on_conflict=on_conflict,
                ignore_duplicates=ignore_duplicates,
                returning=returning
            ).execute() # ADDED execute()
            logger.debug(f"Upsert successful on table '{table}'. Result data: {result.data}")
            return result.data if returning == 'representation' else None
        except Exception as e:
            # Log the specific data that caused the error if possible (be cautious with sensitive data)
            # truncated_data = str(data)[:200] # Example truncation
            # logger.error(f"Error during upsert on table '{table}' with data (truncated): {truncated_data}")
            self._handle_error(e, f"upsert into '{table}'")
    
    def update(
        self,
        table: str,
        data: Dict[str, Any],
        where: Dict[str, Any],
    ) -> Optional[List[Dict[str, Any]]]: # Return list of updated rows or None
        """
        Update rows in a table.

        Args:
            table: Table name.
            data: Data to update.
            where: Filter conditions.

        Returns:
            List of updated rows if successful, otherwise None.

        Raises:
            DatabaseError: If updating fails.
        """
        try:
            query = self.client.table(table).update(data)
            # Apply filters
            for key, value in where.items():
                if isinstance(value, dict) and "operator" in value:
                    query = query.filter(key, value["operator"], value["value"])
                else:
                    query = query.eq(key, value)
            
            # Execute synchronously
            result = query.execute() # Use sync execute
            logger.debug(f"Update successful on table '{table}'. Result data: {result.data}")
            return result.data # Return the list of updated rows
        except Exception as e:
            self._handle_error(e, f"update on '{table}'")
    
    def delete(
        self,
        table: str,
        where: Dict[str, Any],
    ) -> Optional[List[Dict[str, Any]]]: # Return list of deleted rows or None
        """
        Delete rows from a table.

        Args:
            table: Table name.
            where: Filter conditions.

        Returns:
           List of deleted rows if successful, otherwise None.

        Raises:
            DatabaseError: If deleting fails.
        """
        try:
            query = self.client.table(table).delete()
            # Apply filters
            for key, value in where.items():
                if isinstance(value, dict) and "operator" in value:
                    query = query.filter(key, value["operator"], value["value"])
                else:
                    query = query.eq(key, value)

            # Execute synchronously
            result = query.execute() # Use sync execute
            logger.debug(f"Delete successful on table '{table}'. Result data: {result.data}")
            return result.data # Return the list of deleted rows
        except Exception as e:
            self._handle_error(e, f"delete from '{table}'")
    
    def count(
        self,
        table: str,
        where: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Count rows in a table.

        Args:
            table: Table name.
            where: Filter conditions.

        Returns:
            Number of rows.

        Raises:
            DatabaseError: If counting fails.
        """
        try:
            query = self.client.table(table).select("*", count="exact") # Use exact count
            
            # Apply filters if provided
            if where:
                for key, value in where.items():
                    if isinstance(value, dict) and "operator" in value:
                        query = query.filter(key, value["operator"], value["value"])
                    else:
                        query = query.eq(key, value)

            # Execute synchronously
            result = query.execute()
            return result.count if hasattr(result, 'count') else 0
        except Exception as e:
            self._handle_error(e, f"count on '{table}'")
    
    def execute_sql(self, sql: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute raw SQL via RPC function.
        
        Note: This requires setting up a database function in Supabase that can execute SQL.
        
        Args:
            sql: SQL query to execute.
            params: Query parameters.
            
        Returns:
            Query results.
            
        Raises:
            DatabaseError: If execution fails.
            NotImplementedError: If the RPC function is not set up.
        """
        # This requires a SQL execution function to be created in Supabase
        # Example:
        # CREATE OR REPLACE FUNCTION execute_sql(query text, params jsonb DEFAULT '{}'::jsonb)
        # RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER AS $$
        # DECLARE
        #   result jsonb;
        # BEGIN
        #   EXECUTE query INTO result USING params;
        #   RETURN result;
        # END;
        # $$;
        
        raise NotImplementedError("Raw SQL execution requires a database function setup in Supabase")

    # select needs to be sync to match fetch_all/fetch_one
    def select(
        self,
        table: str,
        columns: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """ Wrapper for fetch_all for consistency if needed """
        # This just calls the existing synchronous fetch_all
        return self.fetch_all(table, columns, where, order_by, limit, offset)

@lru_cache
def get_supabase_db_client() -> SupabaseClient:
    """
    Get a cached instance of the SupabaseClient.
    
    Returns:
        A SupabaseClient instance.
    """
    return SupabaseClient() 
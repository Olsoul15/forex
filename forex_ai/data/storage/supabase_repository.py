"""
Supabase repository base class for Forex AI Trading System.

This module provides a base repository class for interacting with Supabase.
"""

import logging
from typing import Dict, List, Any, Optional, TypeVar, Generic, Type
from pydantic import BaseModel

from forex_ai.auth.supabase import get_supabase_client

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class SupabaseRepository(Generic[T]):
    """Base repository class for Supabase integration."""

    def __init__(self, table_name: str, model_class: Type[T]):
        """
        Initialize the repository.

        Args:
            table_name: Name of the Supabase table
            model_class: Pydantic model class for the entity
        """
        self.table_name = table_name
        self.model_class = model_class
        self.client = get_supabase_client()

    async def get_by_id(self, entity_id: str) -> Optional[T]:
        """
        Get an entity by ID.

        Args:
            entity_id: Entity ID

        Returns:
            Entity if found, None otherwise
        """
        try:
            response = self.client.table(self.table_name).select("*").eq("id", entity_id).execute()
            
            # Handle both object with .data attribute and dict with 'data' key
            data = getattr(response, "data", None) if hasattr(response, "data") else response.get("data")
            
            if not data:
                return None
                
            return self.model_class(**data[0])
        except Exception as e:
            logger.error(f"Error getting {self.table_name} by ID: {str(e)}", exc_info=True)
            return None

    async def get_by_user_id(self, user_id: str) -> List[T]:
        """
        Get entities by user ID.

        Args:
            user_id: User ID

        Returns:
            List of entities
        """
        try:
            response = self.client.table(self.table_name).select("*").eq("user_id", user_id).execute()
            
            # Handle both object with .data attribute and dict with 'data' key
            data = getattr(response, "data", None) if hasattr(response, "data") else response.get("data")
            
            if not data:
                return []
                
            return [self.model_class(**item) for item in data]
        except Exception as e:
            logger.error(f"Error getting {self.table_name} by user ID: {str(e)}", exc_info=True)
            return []

    async def create(self, entity_data: Dict[str, Any]) -> Optional[T]:
        """
        Create a new entity.

        Args:
            entity_data: Entity data

        Returns:
            Created entity if successful, None otherwise
        """
        try:
            # Handle both direct execution and object with execute method
            query = self.client.table(self.table_name).insert(entity_data)
            response = query.execute() if hasattr(query, "execute") else query
            
            # Handle both object with .data attribute and dict with 'data' key
            data = getattr(response, "data", None) if hasattr(response, "data") else response.get("data")
            
            if not data:
                return None
                
            return self.model_class(**data[0])
        except Exception as e:
            logger.error(f"Error creating {self.table_name}: {str(e)}", exc_info=True)
            return None

    async def update(self, entity_id: str, entity_data: Dict[str, Any]) -> Optional[T]:
        """
        Update an entity.

        Args:
            entity_id: Entity ID
            entity_data: Entity data

        Returns:
            Updated entity if successful, None otherwise
        """
        try:
            # Handle both direct execution and object with execute method
            query = self.client.table(self.table_name).update(entity_data).eq("id", entity_id)
            response = query.execute() if hasattr(query, "execute") else query
            
            # Handle both object with .data attribute and dict with 'data' key
            data = getattr(response, "data", None) if hasattr(response, "data") else response.get("data")
            
            if not data:
                return None
                
            return self.model_class(**data[0])
        except Exception as e:
            logger.error(f"Error updating {self.table_name}: {str(e)}", exc_info=True)
            return None

    async def delete(self, entity_id: str) -> bool:
        """
        Delete an entity.

        Args:
            entity_id: Entity ID

        Returns:
            True if successful, False otherwise
        """
        try:
            # Handle both direct execution and object with execute method
            query = self.client.table(self.table_name).delete().eq("id", entity_id)
            response = query.execute() if hasattr(query, "execute") else query
            
            # Handle both object with .data attribute and dict with 'data' key
            data = getattr(response, "data", None) if hasattr(response, "data") else response.get("data")
            
            return data is not None and len(data) > 0
        except Exception as e:
            logger.error(f"Error deleting {self.table_name}: {str(e)}", exc_info=True)
            return False

    async def list(self, limit: int = 100, offset: int = 0) -> List[T]:
        """
        List entities with pagination.

        Args:
            limit: Maximum number of entities to return
            offset: Number of entities to skip

        Returns:
            List of entities
        """
        try:
            # Handle both direct execution and object with execute method
            query = self.client.table(self.table_name).select("*").limit(limit).offset(offset)
            response = query.execute() if hasattr(query, "execute") else query
            
            # Handle both object with .data attribute and dict with 'data' key
            data = getattr(response, "data", None) if hasattr(response, "data") else response.get("data")
            
            if not data:
                return []
                
            return [self.model_class(**item) for item in data]
        except Exception as e:
            logger.error(f"Error listing {self.table_name}: {str(e)}", exc_info=True)
            return []

    async def count(self) -> int:
        """
        Count entities.

        Returns:
            Number of entities
        """
        try:
            # Handle both direct execution and object with execute method
            query = self.client.table(self.table_name).select("count", count="exact")
            response = query.execute() if hasattr(query, "execute") else query
            
            # Handle both object with .count attribute and dict with 'count' key
            count_value = getattr(response, "count", None) if hasattr(response, "count") else response.get("count")
            
            return count_value or 0
        except Exception as e:
            logger.error(f"Error counting {self.table_name}: {str(e)}", exc_info=True)
            return 0

    async def query(self, query_func) -> List[T]:
        """
        Execute a custom query.

        Args:
            query_func: Function that takes a table reference and returns a query

        Returns:
            List of entities
        """
        try:
            table_ref = self.client.table(self.table_name)
            query = query_func(table_ref)
            
            # Handle both direct execution and object with execute method
            response = query.execute() if hasattr(query, "execute") else query
            
            # Handle both object with .data attribute and dict with 'data' key
            data = getattr(response, "data", None) if hasattr(response, "data") else response.get("data")
            
            if not data:
                return []
                
            return [self.model_class(**item) for item in data]
        except Exception as e:
            logger.error(f"Error querying {self.table_name}: {str(e)}", exc_info=True)
            return [] 
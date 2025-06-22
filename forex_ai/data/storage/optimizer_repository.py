"""
Forex optimizer repository for Forex AI Trading System.

This module provides a repository for forex optimizer jobs using Supabase.
"""

import logging
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from forex_ai.data.storage.supabase_repository import SupabaseRepository

logger = logging.getLogger(__name__)


class OptimizerJob(BaseModel):
    """Forex optimizer job model."""
    
    id: str
    user_id: str
    status: str
    progress: float = 0.0
    parameters: Dict[str, Any]
    results: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class OptimizerRepository(SupabaseRepository[OptimizerJob]):
    """Repository for forex optimizer jobs."""
    
    def __init__(self):
        """Initialize the repository."""
        super().__init__("forex_optimizer_jobs", OptimizerJob)
    
    async def get_jobs(self, 
                      user_id: str,
                      status: Optional[str] = None,
                      limit: int = 10,
                      offset: int = 0) -> List[OptimizerJob]:
        """
        Get optimizer jobs for a user.
        
        Args:
            user_id: User ID
            status: Filter by status
            limit: Maximum number of jobs to return
            offset: Number of jobs to skip
            
        Returns:
            List of optimizer jobs
        """
        try:
            def build_query(table_ref):
                query = table_ref.select("*").eq("user_id", user_id)
                
                if status:
                    query = query.eq("status", status)
                    
                query = query.order("created_at", desc=True).limit(limit).offset(offset)
                
                return query
            
            return await self.query(build_query)
        except Exception as e:
            logger.error(f"Error getting optimizer jobs: {str(e)}", exc_info=True)
            return []
    
    async def get_job_by_id(self, job_id: str) -> Optional[OptimizerJob]:
        """
        Get optimizer job by ID.
        
        Args:
            job_id: Job ID
            
        Returns:
            Optimizer job if found, None otherwise
        """
        return await self.get_by_id(job_id)
    
    async def create_job(self,
                        user_id: str,
                        parameters: Dict[str, Any]) -> Optional[OptimizerJob]:
        """
        Create a new optimizer job.
        
        Args:
            user_id: User ID
            parameters: Job parameters
            
        Returns:
            Created job if successful, None otherwise
        """
        try:
            job_data = {
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "status": "pending",
                "progress": 0.0,
                "parameters": parameters,
                "message": "Job created, waiting to start",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }
            
            return await self.create(job_data)
        except Exception as e:
            logger.error(f"Error creating optimizer job: {str(e)}", exc_info=True)
            return None
    
    async def update_job_status(self,
                               job_id: str,
                               status: str,
                               progress: Optional[float] = None,
                               message: Optional[str] = None,
                               results: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update optimizer job status.
        
        Args:
            job_id: Job ID
            status: New status
            progress: Job progress (0-100)
            message: Status message
            results: Job results
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get current job
            job = await self.get_by_id(job_id)
            
            if not job:
                logger.error(f"Job {job_id} not found")
                return False
            
            # Update job
            update_data = {
                "status": status,
                "updated_at": datetime.now().isoformat(),
            }
            
            if progress is not None:
                update_data["progress"] = progress
                
            if message:
                update_data["message"] = message
                
            if results:
                update_data["results"] = results
                
            result = await self.update(job_id, update_data)
            return result is not None
        except Exception as e:
            logger.error(f"Error updating optimizer job status: {str(e)}", exc_info=True)
            return False
    
    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel optimizer job.
        
        Args:
            job_id: Job ID
            
        Returns:
            True if successful, False otherwise
        """
        return await self.update_job_status(
            job_id=job_id,
            status="cancelled",
            message="Job cancelled by user",
        )

"""
Strategy API Endpoints.

This module provides FastAPI endpoints for strategy management.
"""

from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status
from datetime import datetime

# Create router
router = APIRouter(prefix="/strategies", tags=["strategies"])

# Simple in-memory storage for testing
strategies_db = {}


@router.get("/test")
async def test_connection():
    """Test endpoint to verify API connectivity."""
    return {
        "status": "success",
        "message": "Strategy API is connected",
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/")
async def get_strategies():
    """Get all strategies (test endpoint)."""
    return {"success": True, "strategies": [], "message": "This is a test endpoint"}

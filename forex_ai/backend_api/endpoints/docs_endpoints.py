"""
Documentation API Endpoints.

This module contains FastAPI endpoints for API documentation.
"""

import logging
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Path
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi import status

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["documentation"])


@router.get("/docs")
async def get_docs():
    """
    Redirect to the Swagger UI documentation.
    """
    logger.info("Redirecting to Swagger UI documentation")
    return RedirectResponse(url="/docs/")


@router.get("/redoc")
async def get_redoc():
    """
    Redirect to the ReDoc documentation.
    """
    logger.info("Redirecting to ReDoc documentation")
    return RedirectResponse(url="/redoc/")


@router.get("/api/docs")
async def mock_api_docs():
    """
    Mock implementation for /api/docs.
    """
    logger.info("Processing mock request for /api/docs")
    return RedirectResponse(url="/docs/")


@router.get("/api/redoc")
async def mock_api_redoc():
    """
    Mock implementation for /api/redoc.
    """
    logger.info("Processing mock request for /api/redoc")
    return RedirectResponse(url="/redoc/") 
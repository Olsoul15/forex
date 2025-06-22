"""
Main entry point for the Forex AI Trading System.

This module initializes the FastAPI application and sets up routes.
"""

import logging
import os
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from forex_ai.config.settings import get_settings
from forex_ai.utils.logging import setup_logging
from forex_ai.api.main import app as api_app
from forex_ai.automation.engine import get_workflow_engine

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=get_settings().API_TITLE,
    description=get_settings().API_DESCRIPTION,
    version=get_settings().API_VERSION,
    docs_url=get_settings().API_DOCS_URL,
    redoc_url=get_settings().API_REDOC_URL,
    openapi_url=get_settings().API_OPENAPI_URL,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API app
app.mount("/api", api_app)

# Initialize workflow engine if enabled
if get_settings().WORKFLOW_ENABLED:
    try:
        # Import and register workflows
        from forex_ai.automation.workflows.data_collection import register_workflows
        
        # Start workflow engine
        workflow_engine = get_workflow_engine()
        register_workflows()
        workflow_engine.start()
        
        logger.info("Workflow engine started successfully")
    except Exception as e:
        logger.error(f"Failed to start workflow engine: {str(e)}")

@app.on_event("startup")
async def startup():
    """Run startup tasks."""
    logger.info("Starting Forex AI Trading System")

@app.on_event("shutdown")
async def shutdown():
    """Run shutdown tasks."""
    logger.info("Shutting down Forex AI Trading System")
    
    # Stop workflow engine if enabled
    if get_settings().WORKFLOW_ENABLED:
        try:
            workflow_engine = get_workflow_engine()
            workflow_engine.stop()
            logger.info("Workflow engine stopped successfully")
        except Exception as e:
            logger.error(f"Failed to stop workflow engine: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Forex AI Trading System",
        "version": get_settings().API_VERSION,
        "status": "running",
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}

def run():
    """Run the application."""
    settings = get_settings()
    
    uvicorn.run(
        "forex_ai.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        workers=settings.API_WORKERS,
        log_level=settings.LOG_LEVEL.lower(),
    )

if __name__ == "__main__":
    run()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Development example for OANDA API.

This script is for DEVELOPMENT USE ONLY. It demonstrates how to use the OANDA API
with credentials from environment variables.

In production, credentials should always come from the database/API.

Required environment variables:
- OANDA_ACCESS_TOKEN or OANDA_API_KEY
- OANDA_ACCOUNT_ID
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from forex_ai.models.broker_models import OandaCredentials
from forex_ai.execution.oanda_api import OandaAPI
from forex_ai.data.connectors.oanda_handler import OandaDataHandler
from forex_ai.utils.logging import setup_logging, get_logger

# Set up logging
setup_logging()
logger = get_logger(__name__)


async def main():
    """Main entry point."""
    logger.warning("DEVELOPMENT MODE: Using environment variables for OANDA credentials")
    
    # Check for required environment variables
    access_token = os.environ.get("OANDA_ACCESS_TOKEN") or os.environ.get("OANDA_API_KEY")
    account_id = os.environ.get("OANDA_ACCOUNT_ID")
    
    if not access_token:
        logger.error("Neither OANDA_ACCESS_TOKEN nor OANDA_API_KEY environment variable is set")
        sys.exit(1)
    if not account_id:
        logger.error("OANDA_ACCOUNT_ID environment variable not set")
        sys.exit(1)
    
    try:
        # Create credentials from environment variables
        credentials = OandaCredentials.from_env()
        
        # Create OANDA API instance
        oanda_api = OandaAPI(credentials=credentials)
        
        # Create OANDA data handler
        oanda_handler = OandaDataHandler(
            access_token=credentials.access_token,
            account_id=credentials.default_account_id
        )
        
        # Get account info
        account_info = await oanda_api.get_account_info()
        logger.info(f"Account info: {account_info}")
        
        # Get historical data
        df = oanda_handler.fetch_historical_data(
            pair="EUR_USD",
            timeframe="M5",
            count=10
        )
        logger.info(f"Historical data:\n{df}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 
# test_supabase_provider.py
import asyncio
import logging
from datetime import datetime, timedelta

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Make sure the provider can be imported (adjust path if necessary)
try:
    from backend_api.strategy_service.app.dependencies import HistoricalSupabaseDataProvider
    from forex_ai.exceptions import DatabaseError
except ImportError as e:
    logger.error(f"Failed to import HistoricalSupabaseDataProvider. Ensure forex_ai is in the Python path or installed. Error: {e}")
    # You might need to adjust PYTHONPATH or run this script from the project root
    # For example: export PYTHONPATH=/path/to/your/project/aiforex:$PYTHONPATH
    # Or, if running from the root: python test_supabase_provider.py
    exit(1)


async def main():
    logger.info("Starting Supabase provider test...")
    try:
        provider = HistoricalSupabaseDataProvider()
    except DatabaseError as e:
        logger.error(f"Failed to initialize provider: {e}")
        return

    # Define test parameters
    test_pair = "EUR_USD"
    test_timeframe = "H1"
    # These times will be ignored currently as the filters are commented out
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=1) # Example: last 1 day

    logger.info(f"Calling get_historical_candles for {test_pair} ({test_timeframe})...")
    logger.warning("Timestamp filters in the provider are currently commented out - fetching all data for pair/timeframe.")

    try:
        result = await provider.get_historical_candles(
            pair=test_pair,
            timeframe=test_timeframe,
            start_time=start_time,
            end_time=end_time
        )

        if 'candles' in result and not result['candles'].empty:
            df = result['candles']
            logger.info(f"Successfully received {len(df)} candles.")
            logger.info(f"First 5 rows:\n{df.head()}")
            logger.info(f"Last 5 rows:\n{df.tail()}")
        elif 'candles' in result and result['candles'].empty:
            logger.warning("Received an empty DataFrame. No data found or error occurred during fetch.")
        else:
             logger.error(f"Unexpected result format: {result}")


    except Exception as e:
        logger.error(f"An error occurred during get_historical_candles call: {e}", exc_info=True)

if __name__ == "__main__":
    # Note: Depending on your environment setup, you might need to ensure
    # the forex_ai package is discoverable (e.g., by setting PYTHONPATH
    # or running from the project root directory)
    logger.info("Running test script...")
    asyncio.run(main())
    logger.info("Test script finished.") 
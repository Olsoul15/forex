# test_redis_connection.py

import asyncio
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RedisTest")

try:
    from forex_ai.data.storage.redis_client import get_redis_client, RedisClient
    import redis.exceptions
except ImportError as e:
    logger.error(f"Failed to import necessary modules: {e}")
    logger.error("Ensure forex_ai package and redis library are installed correctly.")
    exit(1)

async def test_connection():
    logger.info("Attempting to get Redis client...")
    redis_client = None # Initialize to None
    try:
        redis_client: RedisClient = await get_redis_client()
        logger.info(f"Successfully obtained Redis client instance: {type(redis_client)}")
    except Exception as e:
        logger.error(f"Failed to get Redis client instance: {e}", exc_info=True)
        return

    # Ensure redis_client is not None before proceeding
    if redis_client is None:
        logger.error("Redis client instance is None after initialization attempt. Cannot proceed.")
        return

    logger.info("Attempting to PING Redis server...")
    try:
        # Check if the client has an async ping method
        # No need to check for iscoroutinefunction, await will work or raise TypeError
        if hasattr(redis_client, 'ping'):
            response = await redis_client.ping()
            if response:
                logger.info(f"Redis PING successful! Response: {response}")
            else:
                logger.error("Redis PING failed (returned false).")
        else:
            logger.error("Redis client instance does not have a PING method.")

    except redis.exceptions.ConnectionError as e:
        logger.error(f"Redis Connection Error: {e}. Check Redis server status and connection settings (host, port, password) used by get_redis_client().", exc_info=True)
    except redis.exceptions.AuthenticationError as e:
        logger.error(f"Redis Authentication Error: {e}. Check Redis password.", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred during Redis PING: {e}", exc_info=True)
    finally:
        # Attempt to close the connection if redis_client was successfully obtained
        if redis_client is not None:
            logger.info("Attempting to close Redis connection...")
            try:
                # Check for async close/aclose methods
                 if hasattr(redis_client, 'close') and asyncio.iscoroutinefunction(redis_client.close):
                     await redis_client.close()
                     logger.info("Redis connection closed (via close).")
                 # The global close_redis_client function might be preferred if available
                 # elif 'close_redis_client' in globals() and asyncio.iscoroutinefunction(globals()['close_redis_client']):
                 #     await close_redis_client()
                 #     logger.info("Global Redis connection closed.")
                 else:
                      logger.warning("Redis client has no suitable async close method found.")
            except Exception as e:
                logger.error(f"Error closing Redis connection: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(test_connection()) 
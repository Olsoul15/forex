import os
import sys
import logging
import requests
from urllib.parse import urljoin
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('healthcheck')

def check_health():
    """Check if the application is healthy."""
    start_time = datetime.now()
    try:
        port = os.environ.get("PORT", "8000")
        base_url = f"http://localhost:{port}"
        logger.info(f"Starting health check on {base_url}")
        
        # Try the health endpoint first
        health_url = urljoin(base_url, "/api/health")
        logger.info(f"Checking health endpoint: {health_url}")
        
        try:
            response = requests.get(health_url, timeout=5)
            logger.info(f"Health endpoint response status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    health_data = response.json()
                    logger.info(f"Health check data: {health_data}")
                    if health_data.get('status') == 'healthy':
                        logger.info("Health check successful via /api/health")
                        sys.exit(0)
                    else:
                        logger.error(f"Unhealthy status reported: {health_data}")
                        sys.exit(1)
                except ValueError as e:
                    logger.error(f"Failed to parse health check response: {e}")
                    sys.exit(1)
            else:
                logger.warning(f"Health endpoint returned status {response.status_code}")
                
                # Try root endpoint as fallback
                logger.info(f"Trying root endpoint: {base_url}")
                response = requests.get(base_url, timeout=5)
                logger.info(f"Root endpoint response status: {response.status_code}")
                
                if response.status_code in [200, 404]:  # 404 is ok as root might not be defined
                    logger.info("Health check successful via root endpoint")
                    sys.exit(0)
                else:
                    logger.error(f"Root endpoint returned status {response.status_code}")
                    sys.exit(1)
                    
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error during health check: {str(e)}")
            sys.exit(1)
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout during health check: {str(e)}")
            sys.exit(1)
                
    except Exception as e:
        logger.error(f"Unexpected error during health check: {str(e)}")
        sys.exit(1)
    finally:
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Health check completed in {duration:.2f} seconds")

if __name__ == "__main__":
    check_health() 
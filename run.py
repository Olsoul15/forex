import os
import pathlib
import uvicorn

# Create necessary log directories
os.makedirs('/app/logs', exist_ok=True)
os.chmod('/app/logs', 0o777)

# Import app after log directory is created
from forex_ai.api.main import app

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    # Configure uvicorn with proper settings for Cloud Run
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True,
        timeout_keep_alive=30,
        workers=1
    ) 
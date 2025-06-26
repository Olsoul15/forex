import os
import subprocess

def main():
    """
    Main function to run the application using Gunicorn.
    Configures Gunicorn with appropriate settings for Cloud Run.
    """
    # Ensure the log directory exists and has the correct permissions.
    # This path must match the one created in the Dockerfile.
    log_dir = "/app/forex_ai/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    port = os.getenv("PORT", "8000")
    host = os.getenv("HOST", "0.0.0.0")
    
    # Production command using Gunicorn
    # See Gunicorn docs for more info: https://docs.gunicorn.org/en/stable/settings.html
    command = [
        "gunicorn",
        "forex_ai.api.main:app",
        "--workers", "4",  # A common starting point for worker processes
        "--worker-class", "uvicorn.workers.UvicornWorker",
        "--bind", f"{host}:{port}",
        "--log-level", "info",
        "--access-logfile", os.path.join(log_dir, "gunicorn-access.log"),
        "--error-logfile", os.path.join(log_dir, "gunicorn-error.log"),
        "--timeout", "120",
        "--keep-alive", "5",
    ]
    
    print(f"Starting server with command: {' '.join(command)}")
    
    # Use subprocess to run the Gunicorn command
    subprocess.run(command)

if __name__ == "__main__":
    main() 
import os
import sys

def main():
    """
    Main function to run the application using Gunicorn.
    This script will be replaced by the Gunicorn process using exec.
    """
    log_dir = "/app/forex_ai/logs"
    port = os.getenv("PORT", "8000")
    host = os.getenv("HOST", "0.0.0.0")

    # Ensure the log directory exists.
    # Note: Permissions should be handled in the Dockerfile.
    os.makedirs(log_dir, exist_ok=True)

    # Command arguments for Gunicorn
    command = [
        "gunicorn",
        "forex_ai.api.main:app",
        "--workers", "4",
        "--worker-class", "uvicorn.workers.UvicornWorker",
        "--bind", f"{host}:{port}",
        "--log-level", "info",
        "--access-logfile", os.path.join(log_dir, "gunicorn-access.log"),
        "--error-logfile", "-",  # Log errors to stdout to be captured by Cloud Run
        "--timeout", "120",
        "--keep-alive", "5",
    ]

    print(f"Executing command: {' '.join(command)}")
    sys.stdout.flush() # Ensure the print statement is flushed before exec

    # Use os.execvp to replace the current process with Gunicorn
    # This is the standard way to run an application server in a container
    os.execvp(command[0], command)

if __name__ == "__main__":
    main() 
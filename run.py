import os
import sys
import traceback

def main():
    """
    Main function to run the application using Gunicorn.
    If Gunicorn fails to start, the error is logged to a file.
    """
    log_dir = "/app/forex_ai/logs"
    error_log_path = os.path.join(log_dir, "startup_error.log")
    
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
        "--access-logfile", "-",
        "--error-logfile", "-",
        "--timeout", "120",
        "--keep-alive", "5",
    ]

    try:
        print(f"Executing command: {' '.join(command)}")
        sys.stdout.flush() # Ensure the print statement is flushed before exec
        os.execvp(command[0], command)
    except Exception as e:
        with open(error_log_path, "w") as f:
            f.write("Failed to start Gunicorn:\n")
            f.write(traceback.format_exc())
        # The CMD in Dockerfile will tail this file, so we just exit
        sys.exit(1)

if __name__ == "__main__":
    main() 
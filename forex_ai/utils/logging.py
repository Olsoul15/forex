"""
Logging utilities for Forex AI.

This module provides logging functions and configuration for the Forex AI system.
"""
import logging
import os
import sys
from datetime import datetime

# Define a project-relative base directory for logs
LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'logs')
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)


# Configure root logger
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def get_logger(name, log_level=None):
    """
    Get a configured logger instance.

    Args:
        name: Name of the logger
        log_level: Optional log level to override default

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    # Override log level if specified
    if log_level:
        logger.setLevel(log_level)
        
    # Ensure a file handler is attached for this logger
    log_file = os.path.join(LOGS_DIR, f"{name.replace('.', '_')}.log")
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    
    # Avoid adding duplicate handlers
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == file_handler.baseFilename for h in logger.handlers):
        logger.addHandler(file_handler)


    return logger


def setup_file_logging(log_dir=None):
    """
    Set up file logging in addition to console logging.

    Args:
        log_dir: Directory to store log files. Defaults to the project's logs directory.
    """
    if log_dir is None:
        log_dir = LOGS_DIR
        
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create a unique log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"forex_ai_{timestamp}.log")

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    # Add file handler to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

    return log_file

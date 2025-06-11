"""
Logging utilities for Forex AI.

This module provides logging functions and configuration for the Forex AI system.
"""
import logging
import os
import sys
from datetime import datetime
from pydantic import BaseModel

LOG_LEVELS = ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]
LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")

# Ensure the logs directory exists
os.makedirs(LOGS_DIR, exist_ok=True)

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

    # Ensure a file handler is attached for this logger if none exists on the root logger
    if not any(isinstance(h, logging.FileHandler) for h in logging.getLogger().handlers):
        setup_file_logging()

    return logger


_file_logging_configured = False

def setup_file_logging():
    """
    Create a timestamped log file and add a file handler to the root logger.
    """
    global _file_logging_configured
    if _file_logging_configured:
        return

    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)

    log_filename = f"forex_ai_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_filepath = os.path.join(LOGS_DIR, log_filename)
    file_handler = logging.FileHandler(log_filepath)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    # Add file handler to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    _file_logging_configured = True

    return log_filepath

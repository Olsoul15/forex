"""
Logging utilities for Forex AI.

This module provides logging functions and configuration for the Forex AI system.
"""
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional
from datetime import datetime
from pydantic import BaseModel

from forex_ai.config.settings import get_settings

LOG_LEVELS = ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]
LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")

# Ensure the logs directory exists
os.makedirs(LOGS_DIR, exist_ok=True)

def setup_logging(log_level: Optional[str] = None) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_level: Optional override for log level
    """
    settings = get_settings()
    
    # Determine log level
    level = log_level or settings.LOG_LEVEL
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create logs directory if it doesn't exist
    log_dir = Path(settings.LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            RotatingFileHandler(
                log_dir / 'forex_ai.log',
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
        ]
    )
    
    # Set log levels for specific loggers
    logging.getLogger('uvicorn').setLevel(logging.INFO)
    logging.getLogger('fastapi').setLevel(logging.INFO)
    
    # Log startup information
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized at level: {level}")
    logger.info(f"Log directory: {log_dir}")

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

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

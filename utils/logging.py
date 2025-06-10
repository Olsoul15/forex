"""
Logging utilities for Forex AI.

This module provides logging functions and configuration for the Forex AI system.
"""

with open(
    r"C:\Users\desha\aiforex\forex_ai\utils\utils_module_entry.log", "a"
) as f_entry:
    import datetime

    f_entry.write(f"{datetime.datetime.now()}: forex_ai.utils.logging module PARSED.\n")

import logging
import os
import sys
from datetime import datetime

# Configure root logger
# MODIFIED: Only configure basicConfig if no handlers are already set for the root logger

_utils_logging_applied_basicConfig = False
_utils_logging_skipped_basicConfig = False

if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _utils_logging_applied_basicConfig = True
else:
    _utils_logging_skipped_basicConfig = True

# Temporary log to check behavior of the guard
_temp_log_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "utils_logging_guard_check.log"
)
with open(_temp_log_path, "a") as f_guard:
    _now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if _utils_logging_applied_basicConfig:
        f_guard.write(f"{_now}: forex_ai.utils.logging APPLIED basicConfig.\n")
    elif _utils_logging_skipped_basicConfig:
        f_guard.write(
            f"{_now}: forex_ai.utils.logging SKIPPED basicConfig (root had handlers).\n"
        )
    else:
        f_guard.write(
            f"{_now}: forex_ai.utils.logging guard logic ran, but neither applied nor skipped flag was true (unexpected).\n"
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
    # TEST LOG INSIDE GET_LOGGER
    with open(
        r"C:\Users\desha\aiforex\forex_ai\utils\get_logger_called.log", "a"
    ) as f_get_logger:
        _now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f_get_logger.write(
            f"{_now}: forex_ai.utils.logging.get_logger() CALLED for name: {name}\n"
        )

    logger = logging.getLogger(name)

    # Override log level if specified
    if log_level:
        logger.setLevel(log_level)

    return logger


def setup_file_logging(log_dir="logs"):
    """
    Set up file logging in addition to console logging.

    Args:
        log_dir: Directory to store log files
    """
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

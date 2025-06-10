import logging
import os
import shutil
from unittest.mock import patch, MagicMock

import pytest

from forex_ai.utils.logging import get_logger, setup_file_logging, LOGS_DIR

def close_log_handlers(logger):
    """Close all handlers associated with a logger."""
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

@pytest.fixture(autouse=True)
def cleanup_logs():
    """A fixture to clean up the logs directory before and after each test."""
    logging.shutdown()
    if os.path.exists(LOGS_DIR):
        shutil.rmtree(LOGS_DIR, ignore_errors=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    yield
    logging.shutdown()
    if os.path.exists(LOGS_DIR):
        shutil.rmtree(LOGS_DIR, ignore_errors=True)

def test_get_logger():
    """
    Tests that get_logger returns a logger with the correct name and file handler.
    """
    logger_name = "test_logger"
    logger = get_logger(logger_name)

    assert isinstance(logger, logging.Logger)
    assert logger.name == logger_name
    
    log_file_path = os.path.join(LOGS_DIR, f"{logger_name}.log")
    assert os.path.exists(log_file_path)

    initial_handler_count = len(logger.handlers)
    get_logger(logger_name) # Call again
    assert len(logger.handlers) == initial_handler_count
    
    close_log_handlers(logger)

def test_get_logger_with_level():
    """
    Tests that get_logger correctly sets the log level when provided.
    """
    logger_name = "test_logger_with_level"
    level = logging.DEBUG
    logger = get_logger(logger_name, log_level=level)

    assert logger.getEffectiveLevel() == level
    close_log_handlers(logger)

def test_setup_file_logging():
    """
    Tests that setup_file_logging adds a file handler to the root logger.
    """
    root_logger = logging.getLogger()
    initial_handler_count = len(root_logger.handlers)
    
    log_file = setup_file_logging()

    assert os.path.exists(log_file)
    assert len(root_logger.handlers) > initial_handler_count
    
    assert any(isinstance(h, logging.FileHandler) for h in root_logger.handlers)
    
    logging.shutdown()

@patch('forex_ai.utils.logging.os.path.exists')
def test_setup_file_logging_creates_dir(mock_path_exists):
    """
    Tests that setup_file_logging creates the log directory if it doesn't exist.
    """
    mock_path_exists.return_value = False
    with patch('os.makedirs') as mock_makedirs:
        setup_file_logging()
        mock_makedirs.assert_called_with(LOGS_DIR)
    logging.shutdown()

@patch('forex_ai.utils.logging.datetime')
def test_setup_file_logging_filename(mock_datetime):
    """
    Tests that setup_file_logging creates a log file with the correct timestamp format.
    """
    mock_now = MagicMock()
    mock_now.strftime.return_value = "20240101_120000"
    mock_datetime.now.return_value = mock_now
    
    log_file = setup_file_logging()
    
    expected_filename = os.path.join(LOGS_DIR, "forex_ai_20240101_120000.log")
    assert log_file == expected_filename
    
    logging.shutdown()

import logging
import os
from forex_ai.utils.logging import get_logger, setup_file_logging

def test_get_logger():
    """
    Tests that a logger instance can be retrieved and has the correct name.
    """
    logger = get_logger(__name__)
    assert isinstance(logger, logging.Logger)
    assert logger.name == __name__

def test_setup_file_logging_creates_file():
    """
    Tests that setup_file_logging creates a log file and that we can clean it up.
    """
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers[:]
    
    # Ensure all handlers are closed before removing them
    for handler in original_handlers:
        handler.close()
        root_logger.removeHandler(handler)

    # Reset the internal flag to allow setup to run again
    from forex_ai.utils import logging as logging_utils
    logging_utils._file_logging_configured = False

    log_filepath = None
    try:
        log_filepath = setup_file_logging()
        assert log_filepath is not None, "setup_file_logging did not return a path"
        assert os.path.exists(log_filepath)
    finally:
        # Clean up all handlers that were added during the test
        for handler in root_logger.handlers[:]:
            handler.close()
            root_logger.removeHandler(handler)
        
        # Clean up the created file
        if log_filepath and os.path.exists(log_filepath):
            os.remove(log_filepath)
            
        # Restore original handlers
        for handler in original_handlers:
            root_logger.addHandler(handler) 
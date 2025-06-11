# root conftest 
import os
import sys
import pytest
from unittest.mock import MagicMock, AsyncMock

def pytest_sessionstart(session):
    """
    Called after the Session object has been created and
    before performing test collection and entering the run test loop.
    """
    # This path needs to be calculated manually because importing forex_ai.utils.logging
    # here would trigger the same logging initialization problem.
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.abspath(os.path.join(project_root, '..')))
    logs_dir = os.path.join(project_root, '..', 'forex_ai', 'logs')
    os.makedirs(logs_dir, exist_ok=True)

def pytest_configure(config):
    """
    Allows plugins and conftest files to perform initial configuration.
    This hook is called for every plugin and initial conftest file
    after command line options have been parsed.
    """
    if not os.path.exists("logs"):
        os.makedirs("logs") 
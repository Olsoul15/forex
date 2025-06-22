import pytest
import asyncio
from unittest.mock import patch, MagicMock
import os

@pytest.fixture
def mock_env():
    """Mock environment variables for testing."""
    with patch.dict(os.environ, {
        "POSTGRES_USER": "testuser",
        "POSTGRES_PASSWORD": "testpassword",
        "POSTGRES_DB": "testdb",
        "POSTGRES_HOST": "localhost",
        "POSTGRES_PORT": "5432",
        "REDIS_HOST": "localhost",
        "REDIS_PORT": "6379",
        "REDIS_DB": "0",
        "REDIS_PASSWORD": "",
    }) as patched_dict:
        yield patched_dict

@pytest.mark.asyncio
async def test_application_starts_successfully(mock_env):
    """
    Test that the main application starts up without raising any exceptions.
    """
    from forex_ai import main as app_main
    try:
        # We only need to patch sleep now
        with patch('forex_ai.main.asyncio.sleep', return_value=None):
             await app_main.main()
    except Exception as e:
        pytest.fail(f"Application startup failed with an exception: {e}") 
"""
Pytest configuration for the Forex AI Trading System API tests.

This file contains fixtures that are available to all tests.
"""

import pytest
import asyncio
from httpx import AsyncClient
from fastapi import FastAPI

from forex_ai.api.main import app
from forex_ai.config.settings import get_settings


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def client():
    """Create a test client for the FastAPI application."""
    settings = get_settings()
    settings.TESTING = True
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def test_app():
    """Return the FastAPI application for testing."""
    settings = get_settings()
    settings.TESTING = True
    return app


@pytest.fixture
def mock_token():
    """Return a mock JWT token for testing."""
    return "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0QGV4YW1wbGUuY29tIiwiZXhwIjoxNzE5MDg4MzA3LCJpYXQiOjE3MTkwMDE5MDcsInJvbGUiOiJ0ZXN0X3VzZXIifQ.YlWbvHmQnK9uYEJRYNw7kLTjx9Ld0JMFpxkXXLRQ1TY"


@pytest.fixture
def mock_headers(mock_token):
    """Return mock headers with authorization token for testing."""
    return {"Authorization": f"Bearer {mock_token}"}

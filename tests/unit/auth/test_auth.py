"""
Unit tests for authentication functionality.

Tests JWT token generation, validation, and mock token handling.
"""

import pytest
import jwt
from datetime import datetime, timedelta
from fastapi import HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials

from forex_ai.api.auth import router as auth_router
from forex_ai.auth.supabase import get_current_user
from forex_ai.config.settings import get_settings


class MockHTTPAuthorizationCredentials:
    """Mock HTTP Authorization credentials for testing."""
    def __init__(self, token):
        self.credentials = token
        self.scheme = "Bearer"


@pytest.fixture
def settings():
    """Get application settings."""
    return get_settings()


@pytest.fixture
def valid_token(settings):
    """Generate a valid JWT token for testing."""
    expiration = int(datetime.now().timestamp()) + 3600  # 1 hour from now
    payload = {
        "sub": "test@example.com",
        "exp": expiration,
        "iat": int(datetime.now().timestamp()),
        "role": "test_user"
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


@pytest.fixture
def expired_token(settings):
    """Generate an expired JWT token for testing."""
    expiration = int(datetime.now().timestamp()) - 3600  # 1 hour ago
    payload = {
        "sub": "test@example.com",
        "exp": expiration,
        "iat": int(datetime.now().timestamp()) - 7200,
        "role": "test_user"
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


@pytest.fixture
def invalid_token(settings):
    """Generate a token with invalid signature."""
    expiration = int(datetime.now().timestamp()) + 3600  # 1 hour from now
    payload = {
        "sub": "test@example.com",
        "exp": expiration,
        "iat": int(datetime.now().timestamp()),
        "role": "test_user"
    }
    return jwt.encode(payload, "invalid_secret", algorithm=settings.ALGORITHM)


class TestTokenGeneration:
    """Tests for token generation."""

    # Skip this test for now as it requires a running server
    @pytest.mark.skip(reason="Requires a running server")
    @pytest.mark.asyncio
    async def test_mock_login_generates_token(self, client):
        """Test that mock login endpoint generates a valid JWT token."""
        response = await client.post(
            "/api/v1/auth/mock-login",
            json={"email": "test@example.com", "password": "test_password"}
        )
        assert response.status_code == 200
        assert "access_token" in response.json()
        token = response.json()["access_token"]
        
        # Verify token is valid JWT
        settings = get_settings()
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        assert payload["sub"] == "test@example.com"
        assert payload["role"] == "test_user"
        assert payload["exp"] > int(datetime.now().timestamp())


class TestTokenValidation:
    """Tests for token validation."""

    @pytest.mark.asyncio
    async def test_valid_token_authentication(self, valid_token):
        """Test that a valid token passes authentication."""
        credentials = MockHTTPAuthorizationCredentials(valid_token)
        user = await get_current_user(credentials)
        assert user is not None
        assert user["id"] == "test-user-id"
        assert user["email"] == "test@example.com"
        assert user["role"] == "test_user"

    @pytest.mark.asyncio
    async def test_expired_token_authentication(self, expired_token):
        """Test that an expired token fails authentication."""
        credentials = MockHTTPAuthorizationCredentials(expired_token)
        with pytest.raises(HTTPException) as excinfo:
            await get_current_user(credentials)
        assert excinfo.value.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_invalid_token_authentication(self, invalid_token):
        """Test that a token with invalid signature fails authentication."""
        credentials = MockHTTPAuthorizationCredentials(invalid_token)
        with pytest.raises(HTTPException) as excinfo:
            await get_current_user(credentials)
        assert excinfo.value.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_missing_token_authentication(self):
        """Test that missing token fails authentication."""
        with pytest.raises(HTTPException) as excinfo:
            await get_current_user(None)
        assert excinfo.value.status_code == status.HTTP_401_UNAUTHORIZED


class TestMockTokenHandling:
    """Tests for mock token handling."""

    @pytest.mark.asyncio
    async def test_mock_token_recognition(self, settings):
        """Test that mock tokens are recognized and handled correctly."""
        # Create a mock token
        expiration = int(datetime.now().timestamp()) + 86400  # 24 hours
        payload = {
            "sub": "test@example.com",
            "exp": expiration,
            "iat": int(datetime.now().timestamp()),
            "role": "test_user"
        }
        token = jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
        
        # Test that the token is recognized as a mock token
        credentials = MockHTTPAuthorizationCredentials(token)
        user = await get_current_user(credentials)
        assert user is not None
        assert user["id"] == "test-user-id"
        assert user["role"] == "test_user"

"""
Supabase authentication client for Forex AI Trading System.
"""

import os
import logging
from typing import Dict, Any, Optional
from functools import lru_cache

from supabase import create_client, Client
from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader, HTTPBearer
from pydantic import BaseModel
from dotenv import load_dotenv

from forex_ai.config.settings import get_settings

logger = logging.getLogger(__name__)

# HTTP security scheme for bearer token
security = HTTPBearer()


# Pydantic models for authentication
class UserCredentials(BaseModel):
    """User credentials for login."""

    email: str
    password: str


class UserRegistration(UserCredentials):
    """User registration data."""

    password_confirmation: str


class AuthResponse(BaseModel):
    """Authentication response with access token."""

    access_token: str
    token_type: str = "bearer"
    user: Dict[str, Any]


@lru_cache()
def get_supabase_client() -> Client:
    """
    Get a Supabase client instance.

    Returns:
        A Supabase client instance
    """
    try:
        # Load environment variables
        load_dotenv()

        # Get Supabase credentials
        supabase_url = os.getenv("SUPABASE_URL", "https://your-project-url.supabase.co")
        supabase_key = os.getenv("SUPABASE_SERVICE_KEY", "your-supabase-anon-key")

        logger.info(f"Initializing Supabase client with URL: {supabase_url}")

        # Only attempt to create a real client with valid-looking credentials
        if (
            supabase_url != "https://your-project-url.supabase.co"
            and supabase_key != "your-supabase-anon-key"
            and "supabase.co" in supabase_url
        ):
            return create_client(supabase_url, supabase_key)
        else:
            logger.warning(
                "Using mock Supabase client due to default or invalid credentials"
            )
            return _create_mock_client()
    except Exception as e:
        logger.error(f"Error creating Supabase client: {str(e)}")
        logger.warning("Falling back to mock Supabase client")
        return _create_mock_client()


def _create_mock_client():
    """
    Create a mock Supabase client for development when real credentials aren't available.

    This allows the application to run without a real Supabase instance.
    """

    # Create a basic mock object that won't throw errors
    class MockSupabaseClient:
        def __init__(self):
            self.auth = MockSupabaseAuth()
            self.table = lambda name: MockTable(name)

        def from_(self, table_name):
            return MockTable(table_name)

    class MockSupabaseAuth:
        def sign_up(self, credentials):
            return {
                "user": {
                    "id": "mock-user-id",
                    "email": credentials.get("email", "mock@example.com"),
                }
            }

        def sign_in(self, credentials):
            return {
                "user": {
                    "id": "mock-user-id",
                    "email": credentials.get("email", "mock@example.com"),
                }
            }

        def sign_out(self):
            return {"success": True}

    class MockTable:
        def __init__(self, name):
            self.name = name

        def select(self, *args, **kwargs):
            return self

        def insert(self, data, **kwargs):
            return {"id": "mock-id", **data}

        def update(self, data, **kwargs):
            return {"id": "mock-id", **data}

        def delete(self, **kwargs):
            return {"success": True}

        def eq(self, *args, **kwargs):
            return self

        def execute(self):
            return {"data": [], "error": None}

    return MockSupabaseClient()


async def get_current_user(token: str = Depends(security)) -> Dict[str, Any]:
    """
    Verify authentication token and return current user.

    Args:
        token: JWT token from HTTP Authorization header

    Returns:
        Dict: User data

    Raises:
        HTTPException: If authentication fails
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        # Get Supabase client
        supabase = get_supabase_client()

        # The token.credentials value contains the actual token string
        # Verify token and get user data
        user = supabase.auth.get_user(token.credentials)

        if user is None:
            raise credentials_exception

        return user.model_dump()
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        raise credentials_exception


class SupabaseAuth:
    """
    Supabase authentication service.
    """

    def __init__(self):
        """Initialize Supabase auth service."""
        self.client = get_supabase_client()

    async def register(self, user_data: UserRegistration) -> AuthResponse:
        """
        Register a new user.

        Args:
            user_data: User registration data

        Returns:
            AuthResponse: Authentication response with access token

        Raises:
            HTTPException: If registration fails
        """
        try:
            if user_data.password != user_data.password_confirmation:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Passwords do not match",
                )

            # Register user with Supabase
            response = self.client.auth.sign_up(
                {
                    "email": user_data.email,
                    "password": user_data.password,
                }
            )

            # Create response
            return AuthResponse(
                access_token=response.session.access_token,
                user=response.user.model_dump(),
            )
        except Exception as e:
            logger.error(f"Registration error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Registration failed: {str(e)}",
            )

    async def login(self, credentials: UserCredentials) -> AuthResponse:
        """
        Login a user.

        Args:
            credentials: User credentials

        Returns:
            AuthResponse: Authentication response with access token

        Raises:
            HTTPException: If login fails
        """
        try:
            # Login with Supabase
            response = self.client.auth.sign_in_with_password(
                {
                    "email": credentials.email,
                    "password": credentials.password,
                }
            )

            # Create response
            return AuthResponse(
                access_token=response.session.access_token,
                user=response.user.model_dump(),
            )
        except Exception as e:
            logger.error(f"Login error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Authentication failed: {str(e)}",
            )

    async def logout(self, token: str) -> Dict[str, str]:
        """
        Logout a user.

        Args:
            token: JWT token

        Returns:
            Dict: Message confirming logout

        Raises:
            HTTPException: If logout fails
        """
        try:
            # Logout from Supabase
            self.client.auth.sign_out(token)

            return {"message": "Successfully logged out"}
        except Exception as e:
            logger.error(f"Logout error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Logout failed: {str(e)}",
            )


# Create a singleton instance
auth_service = SupabaseAuth()

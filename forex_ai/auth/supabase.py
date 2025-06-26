"""
Supabase authentication client for Forex AI Trading System.
"""

import os
import logging
import time
from typing import Dict, Any, Optional
from functools import lru_cache

from supabase import Client
from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader, HTTPBearer
from pydantic import BaseModel
from dotenv import load_dotenv
import jwt
import requests

from forex_ai.config.settings import get_settings
from forex_ai.data.storage.supabase_base import get_base_supabase_client

logger = logging.getLogger(__name__)

# HTTP security scheme for bearer token
security = HTTPBearer()

# Global client instance
_supabase_client = None

def initialize_supabase() -> Optional[Client]:
    """
    Initialize the Supabase client.

    Returns:
        Optional[Client]: The Supabase client instance or None if initialization fails
    """
    global _supabase_client
    
    if _supabase_client is not None:
        return _supabase_client

    try:
        settings = get_settings()
        
        # Check if required credentials are available
        if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
            logger.warning("Supabase credentials not provided. Authentication will be disabled.")
            return None

        _supabase_client = get_base_supabase_client()
        logger.info("Supabase client initialized successfully")
        return _supabase_client
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {str(e)}")
        return None

def get_supabase_client() -> Optional[Client]:
    """
    Get the Supabase client instance.

    Returns:
        Optional[Client]: The Supabase client instance or None if not initialized
    """
    global _supabase_client
    return _supabase_client

async def get_current_user(token: str = Depends(security)) -> Dict[str, Any]:
    """
    Get the current user from a JWT token by validating it with Supabase.

    Args:
        token: JWT token from the Authorization header

    Returns:
        User data from Supabase

    Raises:
        HTTPException: If the token is invalid or authentication is disabled
    """
    client = get_supabase_client()
    if client is None:
        logger.warning("Authentication is disabled - no Supabase client available")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service is not available",
        )

    try:
        # Ask Supabase to validate the token and get the user
        # The supabase-py library handles the JWT verification against Supabase's auth service
        user_response = client.auth.get_user(token.credentials)
        
        # The user object is available in user_response.user
        if not user_response or not user_response.user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token",
            )
        
        # Pydantic models in Supabase are not directly dict-serializable
        # We need to convert it to a dict to be used in the application
        user_dict = user_response.user.dict()
        return user_dict

    except Exception as e:
        logger.error(f"Error getting current user from Supabase: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication token: {e}",
        )


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


class SupabaseAuth:
    """Supabase authentication service."""

    def __init__(self):
        """Initialize the Supabase auth service."""
        self.client = None
        self._initialized = False

    async def ensure_initialized(self):
        """Ensure the auth service is initialized."""
        if not self._initialized:
            try:
                self.client = get_supabase_client()
                self._initialized = True
                logger.info("Supabase auth service initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Supabase auth service: {str(e)}")
                raise ValueError(f"Failed to initialize Supabase auth service: {str(e)}")

    async def register(self, user_data: UserRegistration) -> AuthResponse:
        """
        Register a new user.

        Args:
            user_data: User registration data

        Returns:
            Authentication response with access token

        Raises:
            HTTPException: If registration fails
        """
        try:
            await self.ensure_initialized()

            # Check if passwords match
            if user_data.password != user_data.password_confirmation:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Passwords do not match",
                )

            # Register user with Supabase
            response = self.client.auth.sign_up({
                "email": user_data.email,
                "password": user_data.password,
            })

            # Check for errors in the response
            if response.user is None or response.session is None:
                error_message = "Registration failed"
                if response.user is None and response.session is None:
                    # Likely email already exists and is confirmed
                     error_message = "User with this email already exists."

                logger.warning(f"Supabase registration failed for {user_data.email}. Message: {error_message}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=error_message,
                )

            # Return the direct Supabase authentication response
            return AuthResponse(
                access_token=response.session.access_token,
                user=response.user.dict(),
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error registering user: {str(e)}")
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
            Authentication response with access token

        Raises:
            HTTPException: If login fails
        """
        try:
            await self.ensure_initialized()

            # Sign in user with Supabase
            response = self.client.auth.sign_in_with_password({
                "email": credentials.email,
                "password": credentials.password,
            })

            # Check if login was successful
            if not response.session or not response.user:
                logger.warning(f"Supabase login failed for {credentials.email}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid credentials",
                )

            # Return the direct Supabase authentication response
            return AuthResponse(
                access_token=response.session.access_token,
                user=response.user.dict(),
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error logging in user: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Login failed: {str(e)}",
            )

    async def logout(self, token: str) -> Dict[str, str]:
        """
        Logout a user by invalidating the token with Supabase.

        Args:
            token: The user's JWT

        Returns:
            A confirmation message

        Raises:
            HTTPException: If logout fails
        """
        try:
            await self.ensure_initialized()
            
            # Use the provided token to sign out
            self.client.auth.sign_out(token)
            
            return {"message": "Successfully logged out"}
        except Exception as e:
            logger.error(f"Error logging out user: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Logout failed: {str(e)}",
            )


# Create auth service instance
auth_service = SupabaseAuth()

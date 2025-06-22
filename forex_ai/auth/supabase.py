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
    Get the current user from a JWT token.

    Args:
        token: JWT token

    Returns:
        User data

    Raises:
        HTTPException: If the token is invalid or authentication is disabled
    """
    try:
        # Get settings
        settings = get_settings()

        # Check if authentication is enabled
        client = get_supabase_client()
        if client is None:
            logger.warning("Authentication is disabled - no Supabase client available")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication service is not available",
            )

        # Decode the token
        payload = jwt.decode(
            token.credentials,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )

        # Get user ID from token
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token",
            )

        # Get user data from Supabase
        response = client.table("users").select("*").eq("id", user_id).execute()

        # Check if user exists
        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
            )

        # Return user data
        return response.data[0]
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
        )
    except Exception as e:
        logger.error(f"Error getting current user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication error",
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
                self.client = await get_supabase_client()
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

            # Check if registration was successful
            if not response.get("user"):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Registration failed",
                )

            # Get user data
            user = response.get("user")

            # Generate JWT token
            settings = get_settings()
            access_token = jwt.encode(
                {
                    "sub": user.get("id"),
                    "email": user.get("email"),
                    "exp": time.time() + settings.JWT_EXPIRE_MINUTES * 60,
                },
                settings.JWT_SECRET_KEY,
                algorithm=settings.JWT_ALGORITHM,
            )

            # Return authentication response
            return AuthResponse(
                access_token=access_token,
                user=user,
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

            # Login user with Supabase
            response = self.client.auth.sign_in({
                "email": credentials.email,
                "password": credentials.password,
            })

            # Check if login was successful
            if not response.get("user"):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid credentials",
                )

            # Get user data
            user = response.get("user")

            # Generate JWT token
            settings = get_settings()
            access_token = jwt.encode(
                {
                    "sub": user.get("id"),
                    "email": user.get("email"),
                    "exp": time.time() + settings.JWT_EXPIRE_MINUTES * 60,
                },
                settings.JWT_SECRET_KEY,
                algorithm=settings.JWT_ALGORITHM,
            )

            # Return authentication response
            return AuthResponse(
                access_token=access_token,
                user=user,
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
        Logout a user.

        Args:
            token: JWT token

        Returns:
            Success message

        Raises:
            HTTPException: If logout fails
        """
        try:
            await self.ensure_initialized()

            # Logout user with Supabase
            self.client.auth.sign_out()

            # Return success message
            return {"message": "Successfully logged out"}
        except Exception as e:
            logger.error(f"Error logging out user: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Logout failed: {str(e)}",
            )


# Create auth service instance
auth_service = SupabaseAuth()

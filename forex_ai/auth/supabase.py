"""
Supabase authentication client for Forex AI Trading System.
"""

import os
import logging
import time
from typing import Dict, Any, Optional
from functools import lru_cache

from supabase import create_client, Client
from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader, HTTPBearer
from pydantic import BaseModel
from dotenv import load_dotenv
import jwt
import requests

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
        supabase_key = os.getenv("SUPABASE_SERVICE_KEY", os.getenv("SUPABASE_KEY", "your-supabase-anon-key"))
        
        logger.info(f"Initializing Supabase client with URL: {supabase_url}")

        # Validate credentials
        if (supabase_url == "https://your-project-url.supabase.co" or 
            supabase_key == "your-supabase-anon-key" or 
            "supabase.co" not in supabase_url):
            logger.critical("Invalid Supabase credentials provided!")
            raise ValueError("Invalid Supabase credentials. Please set valid SUPABASE_URL and SUPABASE_KEY environment variables.")

        # Create the real client
        client = create_client(supabase_url, supabase_key)
        # Just create the client without testing a specific table
        logger.info("Successfully created Supabase client")
        return client
    except Exception as e:
        logger.critical(f"Failed to create Supabase client: {str(e)}")
        raise ValueError(f"Failed to connect to Supabase: {str(e)}. Please check your credentials and connection.")


async def get_current_user(token: str = Depends(security)) -> Dict[str, Any]:
    """
    Get the current user from a JWT token.

    Args:
        token: JWT token

    Returns:
        User data

    Raises:
        HTTPException: If the token is invalid
    """
    try:
        # Get settings
        settings = get_settings()

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
        supabase_client = get_supabase_client()
        response = supabase_client.table("users").select("*").eq("id", user_id).execute()

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


class SupabaseAuth:
    """Supabase authentication service."""

    def __init__(self):
        """Initialize the Supabase auth service."""
        self.client = get_supabase_client()

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
            # Logout user with Supabase
            response = self.client.auth.sign_out()

            # Return success message
            return {"message": "Logged out successfully"}
        except Exception as e:
            logger.error(f"Error logging out user: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Logout failed: {str(e)}",
            )


# Create auth service instance
auth_service = SupabaseAuth()

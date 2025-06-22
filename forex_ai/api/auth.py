"""
Authentication API routes for the Forex AI Trading System.
"""

import logging
import time
import jwt
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status

from forex_ai.auth.supabase import (
    auth_service,
    UserCredentials,
    UserRegistration,
    AuthResponse,
    get_current_user,
)
from forex_ai.config.settings import get_settings

# Configure logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=AuthResponse)
async def register(user_data: UserRegistration) -> AuthResponse:
    """
    Register a new user.

    Args:
        user_data: User registration data

    Returns:
        AuthResponse: Authentication response with access token
    """
    logger.info(f"Registering new user with email: {user_data.email}")
    return await auth_service.register(user_data)


@router.post("/login", response_model=AuthResponse)
async def login(credentials: UserCredentials) -> AuthResponse:
    """
    Login a user.

    Args:
        credentials: User credentials

    Returns:
        AuthResponse: Authentication response with access token
    """
    logger.info(f"Login attempt for user: {credentials.email}")
    return await auth_service.login(credentials)


@router.post("/mock-login", response_model=AuthResponse)
async def mock_login(credentials: UserCredentials) -> AuthResponse:
    """
    Mock login endpoint for testing purposes.
    
    This endpoint always succeeds and returns a hardcoded JWT token.

    Args:
        credentials: User credentials

    Returns:
        AuthResponse: Authentication response with mock access token
    """
    logger.info(f"Mock login for testing with email: {credentials.email}")
    
    # Create a mock JWT token that expires in 24 hours
    settings = get_settings()
    secret_key = settings.SECRET_KEY
    expiration = int(time.time()) + 86400  # 24 hours
    
    payload = {
        "sub": credentials.email,
        "exp": expiration,
        "iat": int(time.time()),
        "role": "test_user"
    }
    
    # Generate token
    token = jwt.encode(payload, secret_key, algorithm=settings.ALGORITHM)
    
    # Create response
    return AuthResponse(
        access_token=token,
        user={
            "id": "test-user-id",
            "email": credentials.email,
            "role": "test_user",
            "created_at": "2025-01-01T00:00:00Z"
        }
    )


@router.post("/logout")
async def logout(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, str]:
    """
    Logout a user.

    Args:
        current_user: Current user data from token

    Returns:
        Dict: Message confirming logout
    """
    logger.info(f"Logout for user: {current_user.get('email')}")
    # Get the token from the current user session
    token = current_user.get("session", {}).get("access_token", "")
    return await auth_service.logout(token)


@router.get("/me", response_model=Dict[str, Any])
async def get_me(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get current user information.

    Args:
        current_user: Current user data from token

    Returns:
        Dict: User data
    """
    logger.info(f"Get current user: {current_user.get('email')}")
    return current_user

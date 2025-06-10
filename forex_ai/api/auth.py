"""
Authentication API routes for the Forex AI Trading System.
"""

import logging
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status

from forex_ai.auth.supabase import (
    auth_service,
    UserCredentials,
    UserRegistration,
    AuthResponse,
    get_current_user,
)

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

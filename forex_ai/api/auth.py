"""
Authentication API routes for the Forex AI Trading System.
"""

import logging
import time
import jwt
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer

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

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")


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
async def logout(token: str = Depends(oauth2_scheme)) -> Dict[str, str]:
    """
    Logout a user by invalidating their JWT.

    Args:
        token: The bearer token from the Authorization header.

    Returns:
        Dict: Message confirming logout
    """
    logger.info("Logout attempt.")
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

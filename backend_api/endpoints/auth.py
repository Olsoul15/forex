"""
Authentication API Endpoints.

This module provides FastAPI endpoints for user authentication.
"""

import logging
from typing import Dict, Any, Optional
import secrets
import time
import hashlib
import os
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, status, Header, Request, Cookie
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, EmailStr

# Setup logging
logger = logging.getLogger(__name__)

# JWT settings
JWT_SECRET = os.getenv("JWT_SECRET", secrets.token_hex(32))
JWT_EXPIRATION = 3600  # 1 hour

# Create router
router = APIRouter(prefix="/auth", tags=["auth"])


# Define models
class UserCredentials(BaseModel):
    """User login credentials."""

    email: EmailStr
    password: str


class UserRegistration(BaseModel):
    """User registration data."""

    email: EmailStr
    password: str
    name: str = Field(min_length=1)


class AuthResponse(BaseModel):
    """Authentication response."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int = 3600
    user: Dict[str, Any]


# Simple in-memory user database (replace with a real database in production)
users_db = {}
active_tokens = {}

# Security
security = HTTPBasic()


def get_password_hash(password: str, salt: Optional[str] = None) -> tuple:
    """
    Get a secure password hash.

    Note: In production, use a proper hashing algorithm like bcrypt.
    """
    if not salt:
        salt = secrets.token_hex(16)

    # Create a simple hash with the salt
    password_hash = hashlib.sha256((password + salt).encode()).hexdigest()

    return (password_hash, salt)


def generate_token(user_email: str) -> str:
    """Generate a random token with expiration."""
    token = secrets.token_hex(32)

    # Store token with expiration
    active_tokens[token] = {
        "user_email": user_email,
        "expires": time.time() + JWT_EXPIRATION,
    }

    return token


async def get_current_user(
    authorization: Optional[str] = Header(None),
    session_token: Optional[str] = Cookie(None),
) -> Dict[str, Any]:
    """
    Get the current user from token.

    Returns:
        Dict with user data

    Raises:
        HTTPException: If authentication fails
    """
    token = None

    # Check authorization header first
    if authorization:
        parts = authorization.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1]

    # Check cookie if no token from authorization header
    if not token and session_token:
        token = session_token

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if token not in active_tokens:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token or token expired",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token_data = active_tokens[token]

    # Check if token is expired
    if token_data["expires"] < time.time():
        active_tokens.pop(token, None)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Get user from database
    user_email = token_data["user_email"]
    if user_email not in users_db:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create a copy of the user without password hash
    user = users_db[user_email].copy()
    if "password_hash" in user:
        del user["password_hash"]
    if "salt" in user:
        del user["salt"]

    return user


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

    # Check if user already exists
    if user_data.email in users_db:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email already exists",
        )

    # Hash password with salt
    password_hash, salt = get_password_hash(user_data.password)

    # Create new user
    user = {
        "email": user_data.email,
        "password_hash": password_hash,
        "salt": salt,
        "name": user_data.name,
        "created_at": datetime.now().isoformat(),
    }

    # Save user
    users_db[user_data.email] = user

    # Generate token
    token = generate_token(user_data.email)

    # User info without sensitive data
    user_info = {"email": user["email"], "name": user["name"]}

    return AuthResponse(access_token=token, user=user_info)


@router.post("/login", response_model=AuthResponse)
async def login(credentials: UserCredentials) -> AuthResponse:
    """
    Login with credentials.

    Args:
        credentials: User login credentials

    Returns:
        AuthResponse: Authentication response with access token
    """
    # Check if user exists
    if credentials.email not in users_db:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials"
        )

    user = users_db[credentials.email]

    # Verify password
    password_hash, salt = get_password_hash(credentials.password, user["salt"])

    if password_hash != user["password_hash"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials"
        )

    # Generate token
    token = generate_token(credentials.email)

    # User info without sensitive data
    user_info = {"email": user["email"], "name": user["name"]}

    return AuthResponse(access_token=token, user=user_info)


@router.post("/logout")
async def logout(
    current_user: Dict[str, Any] = Depends(get_current_user),
    authorization: Optional[str] = Header(None),
) -> Dict[str, str]:
    """
    Logout and invalidate current token.

    Args:
        current_user: Current user data
        authorization: Authorization header

    Returns:
        Dict with success message
    """
    if authorization:
        parts = authorization.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1]
            active_tokens.pop(token, None)

    return {"message": "Logged out successfully"}


@router.get("/me", response_model=Dict[str, Any])
async def get_me(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get current user data.

    Args:
        current_user: Current user data (from dependency)

    Returns:
        Dict with user data
    """
    return current_user

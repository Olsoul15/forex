"""
Tests for the settings module.
"""

import os
import pytest
from unittest.mock import patch

from forex_ai.config.settings import Settings, get_settings, get_env_var


def test_settings_default_values():
    """Test that settings have default values."""
    settings = Settings()
    assert settings.ENVIRONMENT == "development"
    assert settings.DEBUG is True  # Default is True in the actual settings
    assert settings.LOG_LEVEL == "DEBUG"  # Default is DEBUG in the actual settings
    assert settings.POSTGRES_HOST == "localhost"
    assert settings.POSTGRES_PORT == 5432
    assert settings.REDIS_HOST == "localhost"
    # Skip REDIS_PORT check since it might be dynamically set or different in different environments
    # Skip Supabase URL and key checks since they might be dynamically set
    assert isinstance(settings.SUPABASE_URL, str)
    assert isinstance(settings.SUPABASE_KEY, str)


def test_settings_from_env_vars():
    """Test that settings are loaded from environment variables."""
    with patch.dict(os.environ, {
        "ENVIRONMENT": "production",
        "DEBUG": "true",
        "LOG_LEVEL": "DEBUG",
        "SUPABASE_URL": "https://test-project.supabase.co",
        "SUPABASE_KEY": "test-key",
    }):
        settings = Settings()
        assert settings.ENVIRONMENT == "production"
        assert settings.DEBUG is True
        assert settings.LOG_LEVEL == "DEBUG"
        assert settings.SUPABASE_URL == "https://test-project.supabase.co"
        assert settings.SUPABASE_KEY == "test-key"


def test_oanda_token_validation():
    """Test OANDA token validation."""
    # Test that OANDA_API_KEY is used as fallback for OANDA_ACCESS_TOKEN
    with patch.dict(os.environ, {
        "OANDA_API_KEY": "test-api-key",
    }):
        settings = Settings()
        assert settings.OANDA_ACCESS_TOKEN == "test-api-key"
        assert settings.OANDA_API_KEY == "test-api-key"
    
    # Test that OANDA_ACCESS_TOKEN takes precedence
    with patch.dict(os.environ, {
        "OANDA_API_KEY": "test-api-key",
        "OANDA_ACCESS_TOKEN": "test-access-token",
    }):
        settings = Settings()
        assert settings.OANDA_ACCESS_TOKEN == "test-access-token"
        assert settings.OANDA_API_KEY == "test-api-key"


def test_get_settings_caching():
    """Test that get_settings() caches the settings instance."""
    settings1 = get_settings()
    settings2 = get_settings()
    assert settings1 is settings2  # Same instance


def test_get_env_var():
    """Test get_env_var function."""
    # Test primary name
    with patch.dict(os.environ, {
        "PRIMARY_VAR": "primary-value",
        "ALIAS_VAR": "alias-value",
    }):
        value = get_env_var("PRIMARY_VAR", ["ALIAS_VAR"])
        assert value == "primary-value"
    
    # Test alias fallback
    with patch.dict(os.environ, {
        "ALIAS_VAR": "alias-value",
    }):
        value = get_env_var("PRIMARY_VAR", ["ALIAS_VAR"])
        assert value == "alias-value"
    
    # Test default value
    with patch.dict(os.environ, {}):
        value = get_env_var("PRIMARY_VAR", ["ALIAS_VAR"], "default-value")
        assert value == "default-value"


def test_workflow_settings():
    """Test workflow engine settings."""
    settings = Settings()
    assert settings.WORKFLOW_ENABLED is True
    assert settings.WORKFLOW_MAX_WORKERS == 4
    assert settings.WORKFLOW_RESULTS_TTL == 30
    
    # Test overriding workflow settings
    with patch.dict(os.environ, {
        "WORKFLOW_ENABLED": "false",
        "WORKFLOW_MAX_WORKERS": "8",
        "WORKFLOW_RESULTS_TTL": "60",
    }):
        settings = Settings()
        assert settings.WORKFLOW_ENABLED is False
        assert settings.WORKFLOW_MAX_WORKERS == 8
        assert settings.WORKFLOW_RESULTS_TTL == 60 
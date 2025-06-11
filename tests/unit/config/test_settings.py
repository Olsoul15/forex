import os
import pytest
from unittest.mock import patch

from pydantic import ValidationError

from forex_ai.config.settings import Settings, get_settings


@pytest.fixture
def mock_redis_password(monkeypatch):
    monkeypatch.setenv("REDIS_PASSWORD", "testpassword")


@pytest.fixture(autouse=True)
def clear_lru_cache():
    get_settings.cache_clear()


def test_default_settings(mock_redis_password):
    """Test that default settings are loaded correctly."""
    settings = Settings()
    assert settings.ENVIRONMENT == "development"
    assert settings.DEBUG is False
    assert settings.LOG_LEVEL == "INFO"
    assert settings.POSTGRES_HOST == "localhost"
    assert settings.REDIS_HOST == "localhost"
    assert settings.TRADING_ENABLED is False


def test_environment_variable_override(monkeypatch, mock_redis_password):
    """Test that environment variables override default settings."""
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("DEBUG", "True")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("TRADING_ENABLED", "True")

    settings = Settings()
    assert settings.ENVIRONMENT == "production"
    assert settings.DEBUG is True
    assert settings.LOG_LEVEL == "DEBUG"
    assert settings.TRADING_ENABLED is True


def test_database_url_assembly(mock_redis_password):
    """Test that the DATABASE_URL is correctly assembled."""
    settings = Settings(
        POSTGRES_USER="testuser",
        POSTGRES_PASSWORD="testpassword",
        POSTGRES_HOST="db",
        POSTGRES_PORT=5433,
        POSTGRES_DB="testdb",
    )
    expected_url = "postgresql://testuser:testpassword@db:5433/testdb"
    assert str(settings.DATABASE_URL) == expected_url


def test_database_url_provided(monkeypatch, mock_redis_password):
    """Test that a provided DATABASE_URL is used."""
    db_url = "postgresql://user:pass@host:5432/db"
    monkeypatch.setenv("DATABASE_URL", db_url)
    settings = Settings()
    assert str(settings.DATABASE_URL) == db_url


def test_missing_db_config(mock_redis_password):
    """Test that a ValueError is raised if required DB config is missing."""
    with pytest.raises(ValidationError) as excinfo:
        Settings(POSTGRES_USER=None)
    assert "Missing required database configuration" in str(excinfo.value)


def test_redis_url_assembly():
    """Test that the REDIS_URL is correctly assembled."""
    settings = Settings(
        REDIS_HOST="redis_host",
        REDIS_PORT=6380,
        REDIS_PASSWORD="redispassword",
    )
    expected_url = "redis://:redispassword@redis_host:6380/0"
    assert str(settings.REDIS_URL) == expected_url


def test_redis_url_provided(monkeypatch):
    """Test that a provided REDIS_URL is used."""
    redis_url = "redis://user:pass@host:6379/1"
    monkeypatch.setenv("REDIS_URL", redis_url)
    settings = Settings()
    assert str(settings.REDIS_URL) == redis_url


def test_missing_redis_config():
    """Test that a ValueError is raised if required Redis config is missing."""
    with pytest.raises(ValidationError) as excinfo:
        Settings(REDIS_HOST=None)
    assert "Missing required Redis configuration" in str(excinfo.value)


def test_get_settings_caching(mock_redis_password):
    """Test that get_settings caches the Settings object."""
    first_call = get_settings()
    second_call = get_settings()
    assert first_call is second_call


@patch("forex_ai.config.settings.Settings")
def test_get_settings_called_once(mock_settings, mock_redis_password):
    """Test that the Settings object is only instantiated once."""
    get_settings()
    get_settings()
    mock_settings.assert_called_once() 
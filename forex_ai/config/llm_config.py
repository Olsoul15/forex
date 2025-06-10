"""
Configuration for LLM-powered features in Forex AI.

This module contains configuration settings for the LLM integration,
including model selection, API settings, and feature toggles.
"""

import os
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """LLM configuration settings"""

    # LLM provider settings
    provider: str = Field(
        default="azure",
        description="LLM provider to use ('azure', 'openai', 'groq', or 'openrouter')",
    )

    model_name: str = Field(
        default="gpt4",
        description="Model ID to use (e.g., 'gpt4', 'mixtral', 'claude3')",
    )

    # Provider-specific settings
    provider_settings: Dict[str, Dict[str, Any]] = Field(
        default={
            "azure": {
                "api_version": "2024-02-15-preview",
                "deployments": {
                    "gpt4": "gpt-4",
                    "gpt35turbo": "gpt-35-turbo",
                    "embedding": "text-embedding-3-small",
                },
            },
            "openai": {"organization_id": None, "beta": True},
            "groq": {"timeout": 30},
            "openrouter": {
                "route_prefix": None,
                "fallback_models": ["mistralai/mixtral-8x7b", "meta/llama2-70b"],
            },
        },
        description="Provider-specific configuration settings",
    )

    # Feature toggles
    features_enabled: Dict[str, bool] = Field(
        default={
            "natural_language_strategy": True,
            "code_generation": True,
            "strategy_optimization": True,
            "intelligent_validation": True,
            "market_context_integration": True,
            "documentation_generation": True,
            "performance_analysis": True,
            "risk_management": True,
            "collaborative_features": True,
        },
        description="Toggle individual LLM features on/off",
    )

    # Rate limiting and quota settings
    rate_limit: Dict[str, Any] = Field(
        default={
            "max_requests_per_minute": {
                "azure": 10,
                "openai": 10,
                "groq": 20,
                "openrouter": 10,
            },
            "max_tokens_per_day": {
                "azure": 100000,
                "openai": 100000,
                "groq": 200000,
                "openrouter": 100000,
            },
        },
        description="Rate limiting settings for LLM API calls",
    )

    # Caching settings
    caching: Dict[str, Any] = Field(
        default={
            "enabled": True,
            "ttl_seconds": 3600,  # 1 hour
            "max_cache_size": 1000,  # Maximum number of cached responses
            "cache_key_fields": ["provider", "model", "prompt", "temperature"],
        },
        description="Caching settings for LLM API calls",
    )

    # Advanced model settings
    model_settings: Dict[str, Any] = Field(
        default={
            "temperature": 0.2,
            "top_p": 1.0,
            "max_tokens": 4000,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "response_format": {"type": "text"},
        },
        description="Advanced settings for LLM model behavior",
    )

    # Fallback settings
    fallbacks: Dict[str, Any] = Field(
        default={
            "max_retries": 3,
            "retry_delay_seconds": 1,
            "fallback_providers": ["openai", "groq", "openrouter"],
            "fallback_models": {
                "gpt4": ["gpt4openai", "mixtral", "claude3"],
                "gpt35turbo": ["llama2", "mixtral"],
                "embedding": ["text-embedding-3-small"],
            },
        },
        description="Fallback settings for handling API failures",
    )

    # Debug settings
    debug: Dict[str, Any] = Field(
        default={
            "log_prompts": False,
            "log_responses": False,
            "save_interactions": False,
            "trace_tokens": False,
        },
        description="Debug settings for LLM interactions",
    )


def get_default_llm_config() -> Dict[str, Any]:
    """
    Get the default LLM configuration.

    Returns:
        Dictionary with LLM configuration
    """
    config = LLMConfig()
    return config.dict()


def get_llm_config_from_env() -> Dict[str, Any]:
    """
    Get LLM configuration from environment variables.

    Returns:
        Dictionary with LLM configuration
    """
    # Start with default config
    config = get_default_llm_config()

    # Override with environment variables if present
    if os.getenv("FOREX_AI_LLM_PROVIDER"):
        config["provider"] = os.getenv("FOREX_AI_LLM_PROVIDER")

    if os.getenv("FOREX_AI_LLM_MODEL"):
        config["model_name"] = os.getenv("FOREX_AI_LLM_MODEL")

    # Provider-specific settings
    for provider in ["azure", "openai", "groq", "openrouter"]:
        env_prefix = f"FOREX_AI_LLM_{provider.upper()}_"
        for key in config["provider_settings"][provider].keys():
            env_var = f"{env_prefix}{key.upper()}"
            if os.getenv(env_var) is not None:
                config["provider_settings"][provider][key] = os.getenv(env_var)

    # Toggle features based on environment variables
    for feature in config["features_enabled"].keys():
        env_var = f"FOREX_AI_LLM_FEATURE_{feature.upper()}"
        if os.getenv(env_var) is not None:
            config["features_enabled"][feature] = os.getenv(env_var).lower() in (
                "true",
                "1",
                "yes",
            )

    return config

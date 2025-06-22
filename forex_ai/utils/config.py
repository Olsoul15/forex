"""
Configuration utilities for the AI Forex Trading System.

This module provides functions for loading and managing configuration
settings across the system.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


def get_env_var(primary_key: str, fallback_keys: List[str] = None, default: Any = None) -> Any:
    """
    Get environment variable with fallback keys.
    
    Args:
        primary_key: Primary environment variable key
        fallback_keys: List of fallback keys to try if primary key is not set
        default: Default value if no keys are set
        
    Returns:
        Environment variable value or default
    """
    value = os.environ.get(primary_key)
    if value is not None:
        return value
        
    if fallback_keys:
        for key in fallback_keys:
            value = os.environ.get(key)
            if value is not None:
                return value
                
    return default


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from a JSON file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Dictionary containing configuration settings
    """
    config = {}

    try:
        if not os.path.exists(config_path):
            logger.warning(f"Config file not found: {config_path}")
            return config

        file_ext = os.path.splitext(config_path)[1].lower()

        if file_ext == ".json":
            with open(config_path, "r") as file:
                config = json.load(file)
        else:
            logger.warning(f"Only JSON format is supported without PyYAML: {file_ext}")

        logger.info(f"Loaded configuration from {config_path}")

    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {str(e)}")

    return config


def get_env_config() -> Dict[str, Any]:
    """
    Load configuration from environment variables.

    Returns:
        Dictionary containing configuration from environment variables
    """
    env_config = {}

    # Common environment variable prefixes for the system
    prefixes = ["FOREX_AI_", "OANDA_", "ML_"]

    for key, value in os.environ.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                # Convert to lowercase for consistency
                config_key = key.lower().replace(prefix.lower(), "")

                # Try to parse boolean values
                if value.lower() in ["true", "yes", "1"]:
                    env_config[config_key] = True
                elif value.lower() in ["false", "no", "0"]:
                    env_config[config_key] = False
                else:
                    # Try to parse as number
                    try:
                        if "." in value:
                            env_config[config_key] = float(value)
                        else:
                            env_config[config_key] = int(value)
                    except ValueError:
                        # Keep as string
                        env_config[config_key] = value

                break

    return env_config


def merge_configs(*configs) -> Dict[str, Any]:
    """
    Merge multiple configurations, with later configs taking precedence.

    Args:
        *configs: Variable number of configuration dictionaries

    Returns:
        Merged configuration dictionary
    """
    result = {}
    for config in configs:
        if config:
            result.update(config)
    return result


def get_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get full configuration by merging file config and environment variables.

    Args:
        config_path: Optional path to configuration file

    Returns:
        Complete configuration dictionary
    """
    # Load from file
    file_config = load_config(config_path) if config_path else {}

    # Load from environment
    env_config = get_env_config()

    # Merge with environment taking precedence
    return merge_configs(file_config, env_config)

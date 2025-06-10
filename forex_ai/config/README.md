# Configuration Module

This module provides centralized configuration management for the Forex AI Trading System.

## Overview

The configuration module offers a comprehensive system for managing configuration settings across all components of the Forex AI Trading System. It includes mechanisms for loading, validating, and accessing configuration from various sources, with support for environment-specific settings and runtime configuration updates.

## Key Components

### Configuration Manager

- **config_manager.py**: Core configuration management
  - Manages loading and merging configuration
  - Handles configuration validation
  - Provides access to configuration values
  - Implements configuration caching

### Configuration Sources

- **sources.py**: Configuration source providers
  - Implements file-based configuration loaders
  - Provides environment variable configuration
  - Supports database-backed configuration
  - Includes command-line argument handling

### Schema Validation

- **schemas.py**: Configuration schema definitions
  - Defines configuration validation schemas
  - Implements type checking and validation
  - Provides default configuration values
  - Includes configuration documentation

### Environment Management

- **environment.py**: Environment handling
  - Manages environment detection
  - Implements environment-specific configuration
  - Provides environment variable utilities
  - Includes environment switching

## Configuration Hierarchy

The configuration system uses a hierarchical approach with the following precedence (highest to lowest):

1. **Runtime Overrides**: Configuration set programmatically at runtime
2. **Command Line Arguments**: Configuration provided via command line
3. **Environment Variables**: Configuration from environment variables
4. **Environment-Specific Files**: Configuration for the current environment
5. **User Configuration**: User-specific configuration files
6. **Default Configuration**: System default configuration files

## Configuration Categories

The module organizes configuration into several categories:

### System Configuration

- **system**: Core system settings
  - Logging configuration
  - Performance settings
  - Feature flags
  - Environment settings

### Trading Configuration

- **trading**: Trading settings
  - Risk management parameters
  - Position sizing rules
  - Trading hours
  - Instrument settings

### Integration Configuration

- **integration**: External integration settings
  - Broker API credentials
  - Data provider settings
  - Third-party service configuration
  - Authentication parameters

### Model Configuration

- **model**: AI model settings
  - Model selection
  - Model parameters
  - Training configuration
  - Inference settings

## Usage Examples

### Basic Configuration Access

```python
from forex_ai.config import get_config

# Get the configuration manager
config = get_config()

# Access configuration values
log_level = config.get("system.logging.level")
max_positions = config.get("trading.risk.max_positions")
broker_api_key = config.get("integration.broker.api_key")

# Access with default values
timeout = config.get("system.network.timeout", 30)  # Default to 30 seconds
```

### Environment-Specific Configuration

```python
from forex_ai.config import get_config, get_environment

# Get current environment
env = get_environment()
print(f"Current environment: {env}")

# Get configuration for specific environment
config = get_config()
if env == "production":
    max_risk = config.get("trading.risk.max_risk_percent", 1.0)  # Conservative default
else:
    max_risk = config.get("trading.risk.max_risk_percent", 2.0)  # More aggressive for testing

print(f"Maximum risk per trade: {max_risk}%")
```

### Updating Configuration

```python
from forex_ai.config import get_config

# Get the configuration manager
config = get_config()

# Update configuration values
config.set("trading.risk.max_positions", 5)
config.set("system.logging.level", "DEBUG")

# Update nested configuration
config.set("model.parameters", {
    "lookback_period": 50,
    "prediction_horizon": 10,
    "confidence_threshold": 0.75
})

# Save configuration changes
config.save()
```

### Configuration Validation

```python
from forex_ai.config import validate_config, ConfigValidationError

try:
    # Validate the current configuration
    validate_config()
    print("Configuration is valid")
except ConfigValidationError as e:
    print(f"Configuration validation failed: {e}")
    print("Validation errors:")
    for error in e.errors:
        print(f"- {error.path}: {error.message}")
```

## Configuration Files

The config module supports several configuration file formats:

- **YAML**: Human-readable configuration format
- **JSON**: Standard data interchange format
- **TOML**: Easy to read configuration format
- **INI**: Simple key-value format
- **Python**: Dynamic configuration files

## Dependencies

- **Core Module**: For system infrastructure and basic utilities
- **PyYAML**: For YAML configuration file support
- **Pydantic**: For configuration validation
- **Python-dotenv**: For environment variable loading 
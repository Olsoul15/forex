# Models Module

This module provides AI model integration and management for the Forex AI Trading System, enabling interaction with various LLM and machine learning models.

## Overview

The models module serves as the central integration point for AI models used throughout the Forex AI Trading System. It provides a standardized interface for working with different model types, manages model loading and inference, and ensures efficient model usage.

## Key Components

### Model Controller

- **controller.py**: Centralized model controller
  - Manages model lifecycles (loading, unloading, etc.)
  - Provides a unified interface for all models
  - Handles model versioning and updates
  - Tracks model performance metrics

### LLM Integration

- **llm_controller.py**: Controller for large language models
  - Integrates with various LLM providers (OpenAI, Azure, Groq, etc.)
  - Manages prompt templates and generation parameters
  - Provides fallback mechanisms between providers
  - Implements caching for efficient operation

## Model Infrastructure

The model infrastructure provides several key capabilities:

1. **Model Abstraction**: Common interface for different model types
2. **Resource Management**: Efficient allocation of compute resources
3. **Metrics Collection**: Performance tracking and monitoring
4. **Caching**: Avoid redundant model calls
5. **Fallback Mechanisms**: Maintain availability when primary providers are unavailable

## Supported Model Types

The module supports various types of models:

- **Large Language Models (LLMs)**: For reasoning, analysis, and text generation
- **Embedding Models**: For text representation and semantic search
- **Vision Models**: For chart image analysis
- **Time-Series Models**: For market prediction and forecasting

## Usage Examples

### Using the Model Controller

```python
from forex_ai.models.controller import get_model_controller

# Get the model controller singleton
controller = get_model_controller()

# Get a model instance
model = controller.get_model("gpt4")

# Use the model
response = model.generate(
    prompt="Analyze the current market conditions for EUR/USD",
    max_tokens=500
)

print(response)

# Get model status and metrics
status = controller.get_model_status("gpt4")
print(f"Model calls: {status['metrics']['calls']}")
print(f"Average response time: {status['metrics']['avg_response_time']}s")
```

### Using the LLM Controller

```python
from forex_ai.models.llm_controller import LLMController

# Create an LLM controller
llm_controller = LLMController()

# Generate text with parameters
response = llm_controller.generate(
    model="gpt-4",
    prompt="What factors are currently affecting the EUR/USD pair?",
    temperature=0.7,
    max_tokens=300
)

print(response)

# Use a specific provider
response = llm_controller.generate_with_provider(
    provider="azure",
    model="gpt4",
    prompt="Analyze recent EUR/USD price action",
    temperature=0.5
)

print(response)
```

## Dependencies

- **Config Module**: For model configuration settings
- **Health Module**: For health monitoring integration
- **OpenAI API**: For OpenAI models
- **Azure OpenAI API**: For Azure-hosted models
- **Groq API**: For Groq models
- **Google AI API**: For Google models 
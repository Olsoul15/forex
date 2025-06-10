# Examples Module

This directory contains example scripts, notebooks, and tutorials for the Forex AI Trading System.

## Overview

The examples module provides working code examples, tutorials, and sample implementations to help users understand and utilize the Forex AI Trading System. It includes examples for different components of the system, various use cases, and common workflows.

## Key Components

### Getting Started Examples

The `getting_started` directory contains introductory examples for first-time users:

- **basic_setup.py**: Setting up the system for first use
- **configuration.py**: Working with the configuration system
- **hello_world.py**: A simple end-to-end example

### Component Examples

Examples of using specific system components:

- **data_examples**: Working with market data
  - **fetching_data.py**: Retrieving historical and real-time data
  - **data_processing.py**: Processing and normalizing data
  - **data_visualization.py**: Visualizing market data

- **model_examples**: Working with AI models
  - **using_llms.py**: Interacting with large language models
  - **prediction_models.py**: Using forecasting models
  - **custom_models.py**: Creating custom model implementations

- **strategy_examples**: Working with trading strategies
  - **simple_strategy.py**: Implementing a basic trading strategy
  - **multi_timeframe.py**: Working with multiple timeframes
  - **strategy_optimization.py**: Optimizing strategy parameters

- **execution_examples**: Order execution and management
  - **order_placement.py**: Placing different order types
  - **position_management.py**: Managing open positions
  - **broker_integration.py**: Connecting with different brokers

### Workflow Examples

Examples of end-to-end workflows:

- **backtesting_workflow.py**: Complete backtesting process
- **live_trading_workflow.py**: Setting up live trading
- **data_pipeline_workflow.py**: Building data pipelines
- **model_training_workflow.py**: Training and deploying models

### Use Case Examples

Examples of specific use cases:

- **trend_following.py**: Implementing trend following strategies
- **news_trading.py**: Trading based on news events
- **portfolio_management.py**: Managing a portfolio of strategies
- **risk_management.py**: Implementing advanced risk management

### Notebooks

Jupyter notebooks for interactive learning:

- **exploratory_analysis.ipynb**: Analyzing market data
- **strategy_development.ipynb**: Developing trading strategies
- **performance_evaluation.ipynb**: Evaluating trading performance
- **ai_forex_analysis.ipynb**: Using AI for market analysis

## Tutorial Scenarios

Each tutorial includes a complete scenario that demonstrates a practical application:

### Scenario 1: Building a Simple Trend Following System

Files in the `trend_following_tutorial` directory guide you through building a basic trend following system:

1. **01_data_preparation.py**: Fetching and preparing historical data
2. **02_indicator_calculation.py**: Calculating technical indicators
3. **03_strategy_implementation.py**: Implementing the strategy logic
4. **04_backtesting.py**: Backtesting the strategy
5. **05_optimization.py**: Optimizing strategy parameters
6. **06_live_trading.py**: Deploying the strategy for live trading

### Scenario 2: News-Enhanced Trading System

Files in the `news_trading_tutorial` directory demonstrate integrating news data:

1. **01_news_data_collection.py**: Collecting financial news
2. **02_sentiment_analysis.py**: Analyzing news sentiment
3. **03_event_detection.py**: Detecting significant market events
4. **04_strategy_integration.py**: Incorporating news into a strategy
5. **05_combined_backtesting.py**: Backtesting the news-enhanced strategy
6. **06_live_implementation.py**: Implementing the strategy live

## Usage Examples

### Running a Basic Example

```bash
# Navigate to the examples directory
cd forex_ai/examples

# Run a basic example
python getting_started/basic_setup.py
```

### Running a Jupyter Notebook

```bash
# Navigate to the examples directory
cd forex_ai/examples

# Start Jupyter
jupyter notebook

# Then open the desired notebook in your browser
```

### Following a Tutorial

Each tutorial contains a README file with step-by-step instructions:

```bash
# Navigate to the tutorial directory
cd forex_ai/examples/trend_following_tutorial

# Read the tutorial guide
cat README.md

# Run each step in sequence
python 01_data_preparation.py
python 02_indicator_calculation.py
# ... and so on
```

## Dependencies

Most examples require the core Forex AI Trading System to be installed. Some examples may have additional dependencies, which are noted in their respective directories.

## Contributing

We welcome contributions of new examples! If you have a useful example or tutorial, please submit a pull request following the project's contribution guidelines. 
# Agents Module

This module contains the AI agent architecture for the Forex AI Trading System, providing specialized agents for different aspects of forex market analysis and trading.

## Overview

The agents module implements a modular agent-based architecture where each agent specializes in a particular aspect of forex analysis or trading. Agents can work independently or collaborate through a workflow system to produce comprehensive market analysis and trading decisions.

## Key Components

### Base Components

- **base.py**: Defines the base agent class and core agent functionality
- **agent_manager.py**: Manages the lifecycle of agents and facilitates communication between them

### Analysis Agents

- **technical_analysis.py**: Agent for technical analysis of price charts and patterns
- **fundamental_analysis.py**: Agent for analyzing economic indicators and news impact
  - Recently enhanced with `EconomicCalendarAnalyzer` for detailed economic event analysis
- **news.py**: Specialized agent for news analysis and sentiment detection
- **sentiment_analysis.py**: Agent for market sentiment analysis from various sources

### Advanced Agents

- **market_state_analysis_agent.py**: Analyzes market states and regime detection
- **context_aware_analyzer.py**: Context-aware market analysis with enhanced memory

### Framework Components

The `framework/` directory contains core components for the agent architecture:

- **agent.py**: Base agent implementation with tools and memory management
- **agent_types.py**: Type definitions for agent communication
- **agent_implementations.py**: Higher-level implementations of specific agent types

### Tools and Utilities

- **tools.py**: Common tools that can be used by various agents
- **tools/**: Directory with specialized tool implementations

### Workflows

The `workflows/` directory contains predefined agent collaboration workflows:

- **forex_analysis_workflow.py**: Complete forex analysis workflow combining multiple agents

## Recent Enhancements

### Economic Calendar Analysis

The `EconomicCalendarAnalyzer` in `fundamental_analysis.py` provides:

- Economic event retrieval and importance assessment
- Impact evaluation of economic events on currency pairs
- Event-driven market sentiment analysis
- Comprehensive calendar analysis for trading decisions

### News Analysis

The newly added `news.py` implements a `NewsAnalysisAgent` that offers:

- Retrieval of relevant financial news for currency pairs
- News sentiment analysis and scoring
- Impact evaluation on specific currency pairs
- Market sentiment summarization based on news events

## Usage Examples

### Using the FundamentalAnalysisAgent

```python
from forex_ai.agents.fundamental_analysis import FundamentalAnalysisAgent

# Create an agent instance
agent = FundamentalAnalysisAgent(config={
    "news_api_key": "your_api_key"
})

# Perform fundamental analysis
result = agent.process({
    "symbol": "EUR/USD",
    "include_news": True
})

print(f"Analysis summary: {result['summary']}")
```

### Using the NewsAnalysisAgent

```python
from forex_ai.agents.news import NewsAnalysisAgent

# Create a news analysis agent
news_agent = NewsAnalysisAgent(config={})

# Get market sentiment summary for currency pairs
sentiment = news_agent.summarize_market_sentiment(
    currency_pairs=["EUR/USD", "GBP/USD", "USD/JPY"]
)

# Access the results
print(f"Overall market sentiment: {sentiment['overall_summary']}")
```

## Dependencies

- **LLM Models**: Language models for reasoning and analysis
- **Data Module**: For retrieving market data and news
- **Analysis Module**: For technical indicators and calculations 
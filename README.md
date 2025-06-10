# Forex AI Trading System

## What Is This?

The Forex AI Trading System is an intelligent platform that analyzes foreign exchange markets and automates trading decisions. Think of it as having a team of AI experts that constantly monitor currency markets, identify opportunities, and execute trades based on data-driven insights.

## How It Works

The system operates in a simple workflow:

1. **Gather Data** - Collects market prices, economic indicators, and news from various sources
2. **Analyze Markets** - Applies AI to detect patterns, calculate indicators, and identify potential trading setups
3. **Make Decisions** - Evaluates trading opportunities and generates signals with entry/exit points
4. **Execute & Monitor** - Places trades automatically (if enabled) and tracks performance in real-time

All these steps happen within a modern web interface where you can monitor everything and adjust settings as needed.

## Key Features

### Market Analysis
- **Technical Analysis Engine** - 50+ indicators and pattern detection algorithms
- **Fundamental Analysis** - Economic data integration and news sentiment analysis
- **Multi-timeframe Analysis** - Examines markets across different time horizons (1m to 1M)

### AI-Powered Trading
- **Agent Framework** - Specialized AI agents for different aspects of trading
- **Strategy Marketplace** - Library of pre-built trading strategies
- **Custom Strategy Builder** - Create and test your own strategies with Pine Script integration
- **Risk Management** - Automated position sizing and risk controls

### Backtesting & Optimization
- **Historical Testing** - Test strategies against years of historical data
- **Performance Analytics** - Comprehensive metrics and visualizations
- **Parameter Optimization** - Find the optimal settings for any strategy
- **Market Condition Analysis** - Understand when strategies perform best

### Modern Dashboard
- **Real-time Monitoring** - Live market data and position tracking
- **Mobile-friendly Interface** - Access from any device, anywhere
- **API Access** - Programmatic access to all features
- **Alerts & Notifications** - Stay informed of important events

## Try It Out

The easiest way to see how the system works is through the web dashboard:

1. Start the system:
   ```bash
   python -m forex_ai start
   ```

2. Open your browser to `http://localhost:8000`

3. Explore the interactive demo with pre-loaded historical data

4. Try running a backtest to see how strategies perform:
   ```bash
   python -m forex_ai backtest --strategy hammers_and_stars --pairs EUR/USD
   ```

## System Components

The system is organized into these main modules:

- **agents/** - AI components that analyze data and make decisions
- **analysis/** - Technical and fundamental analysis tools
- **backtesting/** - Strategy testing against historical data
- **data/** - Market data collection and management
- **api/** - REST API for system interaction
- **ui/** - Web dashboard and user interface
- **config/** - System configuration and settings
- **automation/** - Workflow automation and scheduling

## Installation Instructions

### Prerequisites
- Python 3.10+
- Docker and Docker Compose
- PostgreSQL 14+
- Redis 7+

### Setup Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/forex_ai.git
   cd forex_ai
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On Unix/MacOS
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env file with your configuration
   ```

5. Start required services using Docker:
   ```bash
   docker-compose up -d
   ```

6. Initialize the database:
   ```bash
   python -m forex_ai.scripts.init_db
   ```

## Launching the User Interface

There are several ways to launch the Forex AI Trading System interface:

### Quick Start

For most users, the simplest method is:

```bash
python -m forex_ai start
```

This will start the web server on the default host (localhost) and port (8000).

### Advanced Options

For more control over the server configuration:

```bash
python -m forex_ai start --host 0.0.0.0 --port 8000 --reload
```

- `--host 0.0.0.0`: Makes the server accessible from other devices on your network
- `--port 8000`: Sets the port number (change if 8000 is already in use)
- `--reload`: Enables auto-reloading for development

### Using Docker

If you're using Docker Compose:

```bash
docker-compose up web
```

This will start just the web interface container.

### Accessing the Interface

Once started, access the web interface by opening your browser and navigating to:

- Local access: http://localhost:8000
- Network access (if using --host 0.0.0.0): http://your-ip-address:8000

### Troubleshooting

If you encounter issues starting the interface:

1. Check that all required services are running (`docker-compose ps`)
2. Verify that port 8000 isn't being used by another application
3. Check the logs for errors: `python -m forex_ai logs`
4. Ensure your environment variables are properly set in `.env`

## Web Dashboard

The web dashboard provides a user-friendly interface for monitoring and controlling the system:

### Features

- **Dashboard Home**: Overview of system performance, active signals, and market charts
- **Backtesting Interface**: Run and analyze backtests of trading strategies
- **Strategy Management**: Configure and monitor trading strategies
- **Performance Analytics**: Detailed performance metrics and visualizations

### Starting the Dashboard

To start the web dashboard, use the CLI command:

```bash
python -m forex_ai start --host 0.0.0.0 --port 8000
```

The dashboard will be available at http://localhost:8000

### Accessing the API

The dashboard exposes a REST API for programmatic access:

- `/api/health` - Health check endpoint
- `/api/market-data/{currency_pair}/{timeframe}` - Get market data
- `/api/strategies` - Get available strategies
- `/api/signals` - Get trading signals
- `/api/performance` - Get performance metrics
- `/api/backtest` - Run a backtest

API documentation is available at http://localhost:8000/api/docs

## Further Documentation

- [Architecture Details](ARCHITECTURE.md) - System architecture and design principles
- [API Documentation](api/README.md) - API reference and examples
- [UI Documentation](ui/README.md) - Dashboard features and customization
- [Contributing Guide](CONTRIBUTING.md) - How to contribute to the project

## Recent Updates

### Code Structure Improvements

Recent updates to the codebase have addressed several structural issues:

- **Fixed Module Dependencies**: Corrected import paths and ensured proper module structure
- **Added Missing Components**: Implemented key modules that were referenced but missing
- **Standardized Database Access**: Unified database access through Supabase client
- **Enhanced Market State Analysis**: Improved market state detection and analysis framework

### Added Components

The following components were added or completed:

- **`data/db_connector.py`**: Unified database connectivity layer using Supabase
- **`data/market_data.py`**: Service layer for market data operations
- **`agents/news.py`**: News analysis agent for processing financial news
- **`agents/fundamental_analysis.py`**: Enhanced with `EconomicCalendarAnalyzer`
- **`features/market_states.py`**: Advanced market state identification

### Import Fixes

Fixed several import references to ensure proper module resolution:

- Updated import paths in `integration/tools/fundamental_tools.py`
- Corrected references in `integration/patterns/enhanced_pattern_recognition.py`
- Standardized imports in `integration/market_state_autoagent_integration.py`

### Database Standardization

The system now consistently uses Supabase as the primary database client:

- All data storage operations route through Supabase connector
- Unified database access patterns
- Enhanced caching and data retrieval efficiency

# Environment
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG

# Database Configuration
POSTGRES_USER=postgres
POSTGRES_PASSWORD=password
POSTGRES_DB=forex_ai
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
DATABASE_URL=postgresql://postgres:password@localhost:5432/forex_ai

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=password
REDIS_URL=redis://:password@localhost:6379/0

# N8N Configuration (Optional)
N8N_HOST=localhost
N8N_PORT=5678
N8N_ENCRYPTION_KEY=placeholder_for_development_only

# Broker API (Choose one)
OANDA_API_KEY=placeholder_for_development
OANDA_ACCOUNT_ID=placeholder_for_development
# Alternative brokers
# IBKR_API_KEY=placeholder_for_development
# ALPACA_API_KEY=placeholder_for_development

# External API Keys (Optional for development)
ALPHA_VANTAGE_API_KEY=placeholder_for_development
TRADING_VIEW_API_KEY=placeholder_for_development
NEWS_API_KEY=placeholder_for_development
YOUTUBE_API_KEY=placeholder_for_development

# AI Service Keys (Optional for development)
AZURE_OPENAI_KEY=placeholder_for_development
AZURE_OPENAI_ENDPOINT=placeholder_for_development
AZURE_OPENAI_API_VERSION=2023-05-15
OPENAI_API_KEY=placeholder_for_development
ANTHROPIC_API_KEY=placeholder_for_development
OPENROUTER_API_KEY=placeholder_for_development
GOOGLE_API_KEY=placeholder_for_development

# AI Model Configuration (Optional for development)
REASONING_MODEL=claude-3-sonnet-20240229
CHAT_MODEL=claude-haiku-20240307
VISION_MODEL=gemini-1.5-flash-001
EMBEDDING_MODEL=text-embedding-3-small

# WebApp Authentication
JWT_SECRET_KEY=placeholder_jwt_secret_key_for_development_only
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=60

# Trading Configuration
TRADING_ENABLED=false
MAX_POSITION_SIZE=0.01
MAX_OPEN_POSITIONS=3
DEFAULT_STOP_LOSS_PIPS=50
DEFAULT_TAKE_PROFIT_PIPS=100
RISK_PER_TRADE_PERCENT=1.0

# Web Dashboard Configuration
WEB_PORT=8000
ENABLE_DOCS=true

# Logging Configuration
SENTRY_DSN=
ENABLE_PERFORMANCE_LOGGING=true 
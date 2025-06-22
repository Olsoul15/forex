# Forex AI Trading System

## Overview

The Forex AI Trading System is an advanced platform that leverages AI and machine learning to analyze forex markets, identify trading opportunities, and execute trades. The system integrates with various data sources, provides technical and fundamental analysis, and offers automated trading capabilities.

## Key Features

- **Market Data Analysis**: Real-time and historical market data analysis
- **Technical Analysis**: Advanced technical indicators and pattern recognition
- **Fundamental Analysis**: News sentiment analysis and economic indicator tracking
- **AI-Powered Trading**: Machine learning models for trade signal generation
- **Automated Trading**: Execution of trades based on AI signals
- **Performance Tracking**: Detailed performance metrics and reporting

## Architecture

The system is built with a modular architecture:

- **Data Layer**: Supabase for persistent storage and Redis for caching
- **Analysis Layer**: Technical and fundamental analysis modules
- **AI Layer**: Machine learning models and LLM integrations
- **Execution Layer**: Trade execution and broker integration
- **Workflow Layer**: Custom workflow engine for task automation
- **API Layer**: RESTful API for client applications

## Technologies

- **Backend**: Python with FastAPI
- **Database**: Supabase (PostgreSQL)
- **Caching**: Redis
- **Machine Learning**: TensorFlow, PyTorch, scikit-learn
- **LLM Integration**: MCP Agent for AI-powered analysis
- **Workflow Automation**: Custom Python workflow engine
- **Broker Integration**: OANDA API

## Getting Started

### Prerequisites

- Python 3.9+
- Supabase account
- Redis
- OANDA demo account (for live trading)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/forex-ai-trading-system.git
   cd forex-ai-trading-system
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. Run the application:
   ```
   # For development with mock data
   python load_env_and_run_server.ps1  # Windows
   ./run_server.sh                     # Linux/macOS
   
   # For production with live Supabase
   ENVIRONMENT=production python -m forex_ai.main
   ```

### Docker Deployment

You can also run the application using Docker:

1. Build and start containers:
   ```
   # For development with mock data
   ./run_docker.sh dev    # Linux/macOS
   .\run_docker.ps1 -Mode dev  # Windows
   
   # For production with live Supabase
   ./run_docker.sh prod   # Linux/macOS
   .\run_docker.ps1 -Mode prod  # Windows
   ```

2. Access the application:
   ```
   http://localhost:8000
   ```

## Configuration

The system can be configured through environment variables or a configuration file. See `environment_variables.md` for details.

## API Documentation

API documentation is available at `/docs` when the server is running.

## Development

### Project Structure

```
forex_ai/
  ├── agents/           # AI agents for market analysis
  ├── analysis/         # Technical and fundamental analysis
  ├── api/              # API endpoints
  ├── automation/       # Workflow automation engine
  ├── common/           # Shared utilities and models
  ├── config/           # Configuration settings
  ├── core/             # Core business logic
  ├── data/             # Data management and storage
  ├── execution/        # Trade execution
  ├── integration/      # External integrations
  ├── models/           # Data models and MCP agent
  └── utils/            # Utility functions
```

### Testing

Run tests with pytest:
```
pytest
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Supabase Integration

The system can operate in two modes:

### Development Mode

In development mode, the system uses mock data and a mock Supabase client. This is useful for development and testing without requiring a live Supabase instance.

To enable development mode:
```
export FOREX_AI_DEV_MODE=true  # Linux/macOS
$env:FOREX_AI_DEV_MODE="true"  # Windows PowerShell
```

### Production Mode

In production mode, the system connects to a live Supabase instance. This requires valid Supabase credentials.

Required environment variables:
```
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
ENVIRONMENT=production
FOREX_AI_DEV_MODE=false
```

#### Supabase Schema Setup

Before running in production mode, ensure your Supabase instance has the following tables:

1. `accounts` - User trading accounts
2. `signals` - Trading signals
3. `auto_trading_preferences` - User auto-trading settings
4. `forex_optimizer_jobs` - Optimization job records
5. `system_status` - System status information

You can use the `seed_database.py` script to initialize your database with the required schema and sample data:
```
python seed_database.py --production
```
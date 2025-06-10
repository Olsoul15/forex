"""
Command-line interface for the Forex AI Trading System.

This module provides a CLI for interacting with the Forex AI system.
"""

import os
import sys
import logging
import click
import uvicorn
import webbrowser
import threading
import time
from pathlib import Path
from typing import Dict, Any

from forex_ai.config.settings import get_settings

# Configure logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("forex_ai")


@click.group()
@click.version_option(version="0.1.0")
def main():
    """Forex AI Trading System Command Line Interface."""
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.info(f"Forex AI CLI started (version: {settings.VERSION})")


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, type=int, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
@click.option("--no-browser", is_flag=True, help="Don't open browser automatically")
def start(host, port, reload, no_browser):
    """Start the Forex AI web application."""
    click.echo(f"Starting Forex AI web application on {host}:{port}...")

    # Function to open browser after a delay
    def open_browser():
        # Wait for the server to start
        time.sleep(2)
        # Use localhost instead of 0.0.0.0 for browser URL
        browser_host = "localhost" if host == "0.0.0.0" else host
        url = f"http://{browser_host}:{port}"
        click.echo(f"Opening browser at {url}")
        webbrowser.open(url)

    # Start browser in a separate thread if not disabled
    if not no_browser:
        threading.Thread(target=open_browser, daemon=True).start()

    # Start the web application using uvicorn
    try:
        uvicorn.run(
            "forex_ai.api.main:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info",
        )
    except Exception as e:
        click.echo(f"Error starting web application: {str(e)}")


@main.command()
@click.option("--debug", is_flag=True, help="Enable debug mode")
@click.option("--strategy", default=None, help="Strategy to use")
def agent(debug, strategy):
    """Start the Forex AI agent system."""
    level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(level)

    click.echo("Starting Forex AI agent system...")
    click.echo(f"Debug mode: {'enabled' if debug else 'disabled'}")
    click.echo(f"Using strategy: {strategy or 'default'}")

    # Placeholder for agent system startup
    click.echo("Agent system started")

    # This would be implemented in later phases
    click.echo("Agent system functionality will be implemented in later phases.")


@main.command()
@click.option("--strategy", help="Strategy ID")
@click.option("--pairs", multiple=True, default=["EUR/USD"], help="Currency pairs")
@click.option("--timeframes", multiple=True, default=["1h"], help="Timeframes")
@click.option("--start-date", help="Start date (YYYY-MM-DD)")
@click.option("--end-date", help="End date (YYYY-MM-DD)")
def backtest(strategy, pairs, timeframes, start_date, end_date):
    """Run backtesting for a strategy."""
    click.echo(f"Running backtest for strategy {strategy}...")
    click.echo(f"Pairs: {', '.join(pairs)}")
    click.echo(f"Timeframes: {', '.join(timeframes)}")
    click.echo(f"Date range: {start_date or 'latest'} to {end_date or 'now'}")

    # Start the backtesting process
    try:
        from forex_ai.ui.dashboard.app import ForexAiDashboard

        # Create and initialize dashboard
        dashboard = ForexAiDashboard(
            {
                "charts": {
                    "default_timeframe": timeframes[0],
                    "default_pair": pairs[0],
                },
                "strategies": {"selected_strategy": strategy},
                "backtesting": {
                    "start_date": start_date,
                    "end_date": end_date,
                    "pairs": pairs,
                    "timeframes": timeframes,
                },
            }
        )

        dashboard.initialize()

        click.echo("Backtesting system started")
        click.echo("Starting dashboard for backtesting visualization...")

        # Run dashboard in blocking mode
        dashboard.run(host="localhost", port=8050)
    except ImportError:
        click.echo("Error: Backtesting visualization requires additional dependencies.")
        click.echo("Please install the required packages: pip install plotly dash")
    except Exception as e:
        click.echo(f"Error running backtest: {str(e)}")


@main.command()
@click.option("--clear-cache", is_flag=True, help="Clear data cache")
def update_data(clear_cache):
    """Update market data."""
    click.echo("Updating market data...")

    if clear_cache:
        click.echo("Clearing data cache...")
        # Implementation would clear the cache

    # This would be implemented in later phases
    click.echo("Data update functionality will be implemented in later phases.")


@main.command()
@click.option(
    "--type",
    type=click.Choice(["market", "limit", "stop"]),
    default="market",
    help="Order type",
)
@click.option("--pair", required=True, help="Currency pair")
@click.option(
    "--direction",
    type=click.Choice(["buy", "sell"]),
    required=True,
    help="Trade direction",
)
@click.option("--amount", type=float, required=True, help="Trade amount")
@click.option(
    "--price", type=float, help="Limit/stop price (required for limit/stop orders)"
)
def trade(type, pair, direction, amount, price):
    """Execute a trade."""
    click.echo(f"Executing {direction} {type} order for {amount} {pair}...")

    if type in ["limit", "stop"] and price is None:
        click.echo("Error: Price is required for limit and stop orders.")
        sys.exit(1)

    # This would be implemented in later phases
    click.echo("Trade execution functionality will be implemented in later phases.")


if __name__ == "__main__":
    main()

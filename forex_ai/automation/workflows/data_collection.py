"""
Data Collection Workflow for Forex AI Trading System

This module provides workflows for collecting market data, news, and YouTube videos.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from forex_ai.automation.engine import WorkflowEngine, get_workflow_engine
from forex_ai.data.connectors.alpha_vantage import AlphaVantageConnector
from forex_ai.data.connectors.news_api import NewsApiConnector
from forex_ai.data.connectors.oanda_handler import OandaHandler
from forex_ai.data.storage.supabase_client import get_supabase_db_client

logger = logging.getLogger(__name__)

async def collect_market_data(
    symbols: List[str],
    timeframes: List[str],
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Collect market data for the specified symbols and timeframes.
    
    Args:
        symbols: List of symbols to collect data for.
        timeframes: List of timeframes to collect data for.
        start_date: Optional start date for historical data.
        end_date: Optional end date for historical data.
        
    Returns:
        Dictionary with collection results.
    """
    logger.info(f"Collecting market data for {len(symbols)} symbols and {len(timeframes)} timeframes")
    
    results = {}
    
    # Use Alpha Vantage for historical data
    alpha_vantage = AlphaVantageConnector()
    
    # Use Oanda for real-time data
    oanda = OandaHandler()
    
    # Get Supabase client for storage
    supabase = get_supabase_db_client()
    
    for symbol in symbols:
        symbol_results = {}
        for timeframe in timeframes:
            try:
                # Collect historical data
                if start_date and end_date:
                    historical_data = await alpha_vantage.get_historical_data(
                        symbol=symbol,
                        timeframe=timeframe,
                        start_date=start_date,
                        end_date=end_date,
                    )
                    
                    # Store in Supabase
                    if historical_data:
                        await supabase.store_market_data(
                            symbol=symbol,
                            timeframe=timeframe,
                            data=historical_data,
                        )
                
                # Collect real-time data
                realtime_data = await oanda.get_latest_candles(
                    symbol=symbol,
                    timeframe=timeframe,
                    count=100,  # Last 100 candles
                )
                
                # Store in Supabase
                if realtime_data:
                    await supabase.store_market_data(
                        symbol=symbol,
                        timeframe=timeframe,
                        data=realtime_data,
                    )
                
                symbol_results[timeframe] = {
                    "status": "success",
                    "historical_count": len(historical_data) if historical_data else 0,
                    "realtime_count": len(realtime_data) if realtime_data else 0,
                }
            except Exception as e:
                logger.error(f"Error collecting data for {symbol} {timeframe}: {str(e)}")
                symbol_results[timeframe] = {
                    "status": "error",
                    "error": str(e),
                }
        
        results[symbol] = symbol_results
    
    logger.info(f"Market data collection completed for {len(symbols)} symbols")
    return results

async def collect_news_data(
    keywords: List[str],
    days_back: int = 1,
) -> Dict[str, Any]:
    """
    Collect news data for the specified keywords.
    
    Args:
        keywords: List of keywords to search for.
        days_back: Number of days to look back.
        
    Returns:
        Dictionary with collection results.
    """
    logger.info(f"Collecting news data for {len(keywords)} keywords")
    
    results = {}
    
    # Use News API connector
    news_api = NewsApiConnector()
    
    # Get Supabase client for storage
    supabase = get_supabase_db_client()
    
    for keyword in keywords:
        try:
            # Collect news data
            news_data = await news_api.get_news(
                query=keyword,
                from_date=(datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d"),
                to_date=datetime.now().strftime("%Y-%m-%d"),
            )
            
            # Store in Supabase
            if news_data:
                await supabase.store_news_data(
                    keyword=keyword,
                    data=news_data,
                )
            
            results[keyword] = {
                "status": "success",
                "count": len(news_data) if news_data else 0,
            }
        except Exception as e:
            logger.error(f"Error collecting news for {keyword}: {str(e)}")
            results[keyword] = {
                "status": "error",
                "error": str(e),
            }
    
    logger.info(f"News data collection completed for {len(keywords)} keywords")
    return results

def setup_data_collection_workflows() -> None:
    """
    Set up data collection workflows.
    """
    engine = get_workflow_engine()
    
    # Create market data collection workflow
    market_data_workflow = engine.create_workflow(
        name="Market Data Collection",
        description="Collect market data from various sources",
    )
    
    # Add market data collection task
    engine.add_task(
        workflow_id=market_data_workflow.id,
        name="Collect Major Forex Pairs",
        function=collect_market_data,
        kwargs={
            "symbols": ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD"],
            "timeframes": ["M1", "M5", "H1", "D"],
        },
    )
    
    # Create news data collection workflow
    news_workflow = engine.create_workflow(
        name="News Data Collection",
        description="Collect news data from various sources",
    )
    
    # Add news data collection task
    engine.add_task(
        workflow_id=news_workflow.id,
        name="Collect Forex News",
        function=collect_news_data,
        kwargs={
            "keywords": ["forex", "currency", "central bank", "interest rate"],
            "days_back": 1,
        },
    )
    
    # Schedule workflows
    engine.schedule_workflow(market_data_workflow.id, 3600)  # Every hour
    engine.schedule_workflow(news_workflow.id, 3600 * 6)  # Every 6 hours
    
    logger.info("Data collection workflows set up successfully")

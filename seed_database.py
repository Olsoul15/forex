#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Seed database with initial data for Forex AI Trading System.

This script seeds the Supabase database with initial data for testing and development.
"""

import os
import sys
import json
import logging
import uuid
from datetime import datetime, timedelta
import argparse
from dotenv import load_dotenv
from supabase import create_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_supabase_client():
    """Get Supabase client."""
    load_dotenv()
    
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
    
    if not supabase_url or not supabase_key:
        logger.error("Supabase URL or key not found in environment variables")
        sys.exit(1)
        
    return create_client(supabase_url, supabase_key)


def create_test_users(supabase, count=3):
    """Create test users."""
    logger.info(f"Creating {count} test users")
    
    users = []
    for i in range(1, count + 1):
        user_id = str(uuid.uuid4())
        email = f"test{i}@example.com"
        
        # Create user in auth system
        try:
            # Check if user exists first
            response = supabase.table("users").select("*").eq("email", email).execute()
            
            if response.data:
                logger.info(f"User {email} already exists, skipping")
                users.append(response.data[0])
                continue
                
            # Create user record
            user_data = {
                "id": user_id,
                "email": email,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }
            
            response = supabase.table("users").insert(user_data).execute()
            
            if response.data:
                logger.info(f"Created user {email}")
                users.append(response.data[0])
            else:
                logger.error(f"Failed to create user {email}")
        except Exception as e:
            logger.error(f"Error creating user {email}: {str(e)}")
    
    return users


def create_test_accounts(supabase, users, accounts_per_user=2):
    """Create test accounts."""
    logger.info(f"Creating {accounts_per_user} accounts per user")
    
    accounts = []
    for user in users:
        for i in range(1, accounts_per_user + 1):
            account_id = str(uuid.uuid4())
            
            # Create account
            try:
                account_data = {
                    "id": account_id,
                    "user_id": user["id"],
                    "name": f"Demo Account {i}",
                    "balance": 10000.0 * i,
                    "currency": "USD" if i % 2 == 0 else "EUR",
                    "broker": "OANDA" if i % 2 == 0 else "FXCM",
                    "broker_account_id": f"broker-{account_id[:8]}",
                    "is_demo": True,
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                }
                
                response = supabase.table("accounts").insert(account_data).execute()
                
                if response.data:
                    logger.info(f"Created account {account_data['name']} for user {user['email']}")
                    accounts.append(response.data[0])
                else:
                    logger.error(f"Failed to create account {account_data['name']} for user {user['email']}")
            except Exception as e:
                logger.error(f"Error creating account for user {user['email']}: {str(e)}")
    
    return accounts


def create_test_trades(supabase, accounts, trades_per_account=5):
    """Create test trades."""
    logger.info(f"Creating {trades_per_account} trades per account")
    
    instruments = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD"]
    directions = ["buy", "sell"]
    statuses = ["open", "closed"]
    
    trades = []
    for account in accounts:
        for i in range(1, trades_per_account + 1):
            trade_id = str(uuid.uuid4())
            
            # Randomize trade data
            instrument = instruments[i % len(instruments)]
            direction = directions[i % len(directions)]
            status = statuses[i % len(statuses)]
            size = 0.1 * i
            entry_price = 1.1 + (i * 0.01)
            exit_price = entry_price + (0.005 if direction == "buy" else -0.005) if status == "closed" else None
            profit_loss = (exit_price - entry_price) * size * 10000 if exit_price else None
            entry_time = (datetime.now() - timedelta(days=i)).isoformat()
            exit_time = (datetime.now() - timedelta(days=i, hours=4)).isoformat() if status == "closed" else None
            
            # Create trade
            try:
                trade_data = {
                    "id": trade_id,
                    "user_id": account["user_id"],
                    "account_id": account["id"],
                    "instrument": instrument,
                    "direction": direction,
                    "size": size,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "profit_loss": profit_loss,
                    "status": status,
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "strategy_id": f"strategy-{i % 3}",
                    "strategy_name": ["Moving Average Crossover", "RSI Divergence", "Breakout Strategy"][i % 3],
                    "created_at": entry_time,
                    "updated_at": exit_time if exit_time else entry_time,
                }
                
                response = supabase.table("trades").insert(trade_data).execute()
                
                if response.data:
                    logger.info(f"Created trade {instrument} {direction} for account {account['name']}")
                    trades.append(response.data[0])
                else:
                    logger.error(f"Failed to create trade for account {account['name']}")
            except Exception as e:
                logger.error(f"Error creating trade for account {account['name']}: {str(e)}")
    
    return trades


def create_test_signals(supabase, count=10):
    """Create test signals."""
    logger.info(f"Creating {count} test signals")
    
    instruments = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD"]
    timeframes = ["M5", "M15", "M30", "H1", "H4"]
    directions = ["buy", "sell"]
    statuses = ["active", "executed", "expired", "cancelled"]
    
    signals = []
    for i in range(1, count + 1):
        signal_id = str(uuid.uuid4())
        
        # Randomize signal data
        instrument = instruments[i % len(instruments)]
        timeframe = timeframes[i % len(timeframes)]
        direction = directions[i % len(directions)]
        status = statuses[i % len(statuses)]
        entry_price = 1.1 + (i * 0.01)
        stop_loss = entry_price - (0.005 if direction == "buy" else -0.005)
        take_profit = entry_price + (0.01 if direction == "buy" else -0.01)
        confidence = 0.5 + (i * 0.05) % 0.5
        signal_time = (datetime.now() - timedelta(days=i)).isoformat()
        expiration_time = (datetime.now() + timedelta(days=1)).isoformat()
        
        # Create signal
        try:
            signal_data = {
                "id": signal_id,
                "strategy_id": f"strategy-{i % 3}",
                "strategy_name": ["Moving Average Crossover", "RSI Divergence", "Breakout Strategy"][i % 3],
                "instrument": instrument,
                "timeframe": timeframe,
                "direction": direction,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "confidence": confidence,
                "signal_time": signal_time,
                "expiration_time": expiration_time,
                "status": status,
                "notes": f"Test signal {i}",
                "created_at": signal_time,
                "updated_at": signal_time,
            }
            
            response = supabase.table("signals").insert(signal_data).execute()
            
            if response.data:
                logger.info(f"Created signal {instrument} {direction} {timeframe}")
                signals.append(response.data[0])
            else:
                logger.error(f"Failed to create signal {instrument} {direction}")
        except Exception as e:
            logger.error(f"Error creating signal: {str(e)}")
    
    return signals


def create_test_auto_trading_preferences(supabase, users):
    """Create test auto-trading preferences."""
    logger.info(f"Creating auto-trading preferences for {len(users)} users")
    
    preferences = []
    for user in users:
        pref_id = str(uuid.uuid4())
        
        # Create preferences
        try:
            pref_data = {
                "id": pref_id,
                "user_id": user["id"],
                "enabled": False,
                "risk_per_trade": 1.0,
                "max_daily_trades": 5,
                "max_open_trades": 3,
                "allowed_instruments": ["EUR_USD", "GBP_USD", "USD_JPY"],
                "trading_hours_start": "08:00",
                "trading_hours_end": "16:00",
                "trading_days": [0, 1, 2, 3, 4],
                "min_win_rate": 55.0,
                "min_profit_factor": 1.5,
                "stop_loss_required": True,
                "take_profit_required": True,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }
            
            response = supabase.table("auto_trading_preferences").insert(pref_data).execute()
            
            if response.data:
                logger.info(f"Created auto-trading preferences for user {user['email']}")
                preferences.append(response.data[0])
            else:
                logger.error(f"Failed to create auto-trading preferences for user {user['email']}")
        except Exception as e:
            logger.error(f"Error creating auto-trading preferences for user {user['email']}: {str(e)}")
    
    return preferences


def create_test_optimizer_jobs(supabase, users, jobs_per_user=3):
    """Create test optimizer jobs."""
    logger.info(f"Creating {jobs_per_user} optimizer jobs per user")
    
    statuses = ["pending", "running", "completed", "failed"]
    
    jobs = []
    for user in users:
        for i in range(1, jobs_per_user + 1):
            job_id = str(uuid.uuid4())
            
            # Randomize job data
            status = statuses[i % len(statuses)]
            progress = 100.0 if status == "completed" else (0.0 if status == "failed" else i * 10.0)
            created_at = (datetime.now() - timedelta(days=i)).isoformat()
            updated_at = (datetime.now() - timedelta(hours=i)).isoformat()
            
            # Create job
            try:
                job_data = {
                    "id": job_id,
                    "user_id": user["id"],
                    "status": status,
                    "progress": progress,
                    "parameters": {
                        "instrument": "EUR_USD",
                        "timeframe": "H1",
                        "start_date": "2025-01-01",
                        "end_date": "2025-06-01",
                        "population_size": 50,
                        "generations": 10,
                        "strategy_type": "moving_average_crossover",
                    },
                    "results": {
                        "best_fitness": 0.85,
                        "win_rate": 65.0,
                        "profit_factor": 2.1,
                        "parameters": {
                            "fast_ma": 12,
                            "slow_ma": 26,
                            "signal_ma": 9,
                        },
                    } if status == "completed" else None,
                    "message": "Optimization completed successfully" if status == "completed" else 
                              "Optimization failed" if status == "failed" else 
                              "Optimization in progress",
                    "created_at": created_at,
                    "updated_at": updated_at,
                }
                
                response = supabase.table("forex_optimizer_jobs").insert(job_data).execute()
                
                if response.data:
                    logger.info(f"Created optimizer job for user {user['email']}")
                    jobs.append(response.data[0])
                else:
                    logger.error(f"Failed to create optimizer job for user {user['email']}")
            except Exception as e:
                logger.error(f"Error creating optimizer job for user {user['email']}: {str(e)}")
    
    return jobs


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Seed database with initial data")
    parser.add_argument("--users", type=int, default=3, help="Number of test users to create")
    parser.add_argument("--accounts", type=int, default=2, help="Number of accounts per user")
    parser.add_argument("--trades", type=int, default=5, help="Number of trades per account")
    parser.add_argument("--signals", type=int, default=10, help="Number of signals to create")
    parser.add_argument("--jobs", type=int, default=3, help="Number of optimizer jobs per user")
    
    args = parser.parse_args()
    
    logger.info("Starting database seeding")
    
    try:
        # Get Supabase client
        supabase = get_supabase_client()
        
        # Create test data
        users = create_test_users(supabase, args.users)
        accounts = create_test_accounts(supabase, users, args.accounts)
        trades = create_test_trades(supabase, accounts, args.trades)
        signals = create_test_signals(supabase, args.signals)
        preferences = create_test_auto_trading_preferences(supabase, users)
        jobs = create_test_optimizer_jobs(supabase, users, args.jobs)
        
        logger.info("Database seeding completed successfully")
        logger.info(f"Created {len(users)} users")
        logger.info(f"Created {len(accounts)} accounts")
        logger.info(f"Created {len(trades)} trades")
        logger.info(f"Created {len(signals)} signals")
        logger.info(f"Created {len(preferences)} auto-trading preferences")
        logger.info(f"Created {len(jobs)} optimizer jobs")
        
    except Exception as e:
        logger.error(f"Error seeding database: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

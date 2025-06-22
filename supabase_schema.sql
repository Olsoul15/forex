-- Forex AI Trading System - Supabase Schema
-- This script creates all the necessary tables for the Forex AI Trading System

-- Enable the pgvector extension for vector embeddings (if needed)
CREATE EXTENSION IF NOT EXISTS pgvector;

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    auth_id UUID UNIQUE,
    email TEXT UNIQUE NOT NULL,
    first_name TEXT,
    last_name TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    last_login TIMESTAMP WITH TIME ZONE
);

-- Create accounts table
CREATE TABLE IF NOT EXISTS accounts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    broker TEXT NOT NULL,
    account_number TEXT,
    balance DECIMAL(20, 5) NOT NULL DEFAULT 0,
    currency TEXT NOT NULL,
    is_demo BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    
    -- Add index on user_id for faster lookups
    CONSTRAINT accounts_user_id_idx UNIQUE (user_id, name)
);

-- Create signals table
CREATE TABLE IF NOT EXISTS signals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    instrument TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('buy', 'sell')),
    strength DECIMAL(5, 4) NOT NULL CHECK (strength >= 0 AND strength <= 1),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT now(),
    strategy_id TEXT NOT NULL,
    strategy_name TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    entry_price DECIMAL(20, 5),
    confidence DECIMAL(5, 4) CHECK (confidence >= 0 AND confidence <= 1),
    signal_time TIMESTAMP WITH TIME ZONE NOT NULL,
    expiration_time TIMESTAMP WITH TIME ZONE NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('active', 'expired', 'executed', 'cancelled')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    
    -- Add indexes for common queries
    CONSTRAINT signals_instrument_idx UNIQUE (instrument, strategy_id, timestamp)
);

-- Create auto_trading_preferences table
CREATE TABLE IF NOT EXISTS auto_trading_preferences (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    enabled BOOLEAN NOT NULL DEFAULT FALSE,
    risk_level TEXT NOT NULL DEFAULT 'medium' CHECK (risk_level IN ('low', 'medium', 'high')),
    max_trades INTEGER NOT NULL DEFAULT 3,
    risk_per_trade DECIMAL(5, 2) NOT NULL DEFAULT 1.0,
    max_daily_trades INTEGER NOT NULL DEFAULT 5,
    max_open_trades INTEGER NOT NULL DEFAULT 3,
    allowed_instruments TEXT[] DEFAULT '{}',
    trading_hours_start TEXT NOT NULL DEFAULT '00:00',
    trading_hours_end TEXT NOT NULL DEFAULT '23:59',
    trading_days INTEGER[] DEFAULT '{0,1,2,3,4}', -- 0=Monday, 6=Sunday
    min_win_rate DECIMAL(5, 2) NOT NULL DEFAULT 50.0,
    min_profit_factor DECIMAL(5, 2) NOT NULL DEFAULT 1.2,
    stop_loss_required BOOLEAN NOT NULL DEFAULT TRUE,
    take_profit_required BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    
    -- Each user can only have one preferences record
    CONSTRAINT auto_trading_preferences_user_id_unique UNIQUE (user_id)
);

-- Create forex_optimizer_jobs table
CREATE TABLE IF NOT EXISTS forex_optimizer_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    job_type TEXT NOT NULL CHECK (job_type IN ('optimization', 'walkforward', 'montecarlo')),
    status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    progress DECIMAL(5, 2) NOT NULL DEFAULT 0.0,
    message TEXT,
    parameters JSONB NOT NULL DEFAULT '{}'::jsonb,
    results JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    
    -- Add index on user_id for faster lookups
    CONSTRAINT forex_optimizer_jobs_user_id_idx UNIQUE (user_id, created_at)
);

-- Create system_status table
CREATE TABLE IF NOT EXISTS system_status (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    component TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('healthy', 'degraded', 'down')),
    message TEXT,
    last_check TIMESTAMP WITH TIME ZONE DEFAULT now(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    
    -- Each component can only have one status record
    CONSTRAINT system_status_component_unique UNIQUE (component)
);

-- Create trades table with partitioning by entry_time
CREATE TABLE IF NOT EXISTS trades (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    account_id UUID NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    instrument TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('buy', 'sell')),
    size DECIMAL(20, 5) NOT NULL,
    entry_price DECIMAL(20, 5) NOT NULL,
    exit_price DECIMAL(20, 5),
    stop_loss DECIMAL(20, 5),
    take_profit DECIMAL(20, 5),
    entry_time TIMESTAMP WITH TIME ZONE NOT NULL,
    exit_time TIMESTAMP WITH TIME ZONE,
    profit_loss DECIMAL(20, 5),
    status TEXT NOT NULL CHECK (status IN ('open', 'closed', 'cancelled')),
    signal_id UUID REFERENCES signals(id),
    strategy TEXT,
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
) PARTITION BY RANGE (entry_time);

-- Create partitions for trades table (by month for the current year)
CREATE TABLE trades_2025_q1 PARTITION OF trades
    FOR VALUES FROM ('2025-01-01') TO ('2025-04-01');
    
CREATE TABLE trades_2025_q2 PARTITION OF trades
    FOR VALUES FROM ('2025-04-01') TO ('2025-07-01');
    
CREATE TABLE trades_2025_q3 PARTITION OF trades
    FOR VALUES FROM ('2025-07-01') TO ('2025-10-01');
    
CREATE TABLE trades_2025_q4 PARTITION OF trades
    FOR VALUES FROM ('2025-10-01') TO ('2026-01-01');

-- Add indexes for common queries on trades table
CREATE INDEX trades_user_id_idx ON trades(user_id);
CREATE INDEX trades_account_id_idx ON trades(account_id);
CREATE INDEX trades_instrument_idx ON trades(instrument);
CREATE INDEX trades_entry_time_idx ON trades(entry_time);
CREATE INDEX trades_status_idx ON trades(status);

-- Create Row Level Security (RLS) policies
-- Enable RLS on all tables
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE accounts ENABLE ROW LEVEL SECURITY;
ALTER TABLE signals ENABLE ROW LEVEL SECURITY;
ALTER TABLE auto_trading_preferences ENABLE ROW LEVEL SECURITY;
ALTER TABLE forex_optimizer_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE trades ENABLE ROW LEVEL SECURITY;

-- Create policies for users table
CREATE POLICY users_select_own ON users
    FOR SELECT
    USING (auth.uid() = auth_id);

-- Create policies for accounts table
CREATE POLICY accounts_select_own ON accounts
    FOR SELECT
    USING (auth.uid() IN (SELECT auth_id FROM users WHERE id = accounts.user_id));

-- Create policies for signals table
CREATE POLICY signals_select_all ON signals
    FOR SELECT
    TO authenticated
    USING (true);

-- Create policies for auto_trading_preferences table
CREATE POLICY auto_trading_preferences_select_own ON auto_trading_preferences
    FOR SELECT
    USING (auth.uid() IN (SELECT auth_id FROM users WHERE id = auto_trading_preferences.user_id));

-- Create policies for forex_optimizer_jobs table
CREATE POLICY forex_optimizer_jobs_select_own ON forex_optimizer_jobs
    FOR SELECT
    USING (auth.uid() IN (SELECT auth_id FROM users WHERE id = forex_optimizer_jobs.user_id));

-- Create policies for trades table
CREATE POLICY trades_select_own ON trades
    FOR SELECT
    USING (auth.uid() IN (SELECT auth_id FROM users WHERE id = trades.user_id));

-- Create initial system status records
INSERT INTO system_status (component, status, message)
VALUES 
    ('database', 'healthy', 'Database connection established'),
    ('api', 'healthy', 'API is running'),
    ('broker_integration', 'healthy', 'Broker integration is operational'),
    ('data_feeds', 'healthy', 'Market data feeds are active');

-- Create functions and triggers for updated_at timestamps
CREATE OR REPLACE FUNCTION update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for all tables
CREATE TRIGGER update_users_timestamp
BEFORE UPDATE ON users
FOR EACH ROW EXECUTE PROCEDURE update_timestamp();

CREATE TRIGGER update_accounts_timestamp
BEFORE UPDATE ON accounts
FOR EACH ROW EXECUTE PROCEDURE update_timestamp();

CREATE TRIGGER update_signals_timestamp
BEFORE UPDATE ON signals
FOR EACH ROW EXECUTE PROCEDURE update_timestamp();

CREATE TRIGGER update_auto_trading_preferences_timestamp
BEFORE UPDATE ON auto_trading_preferences
FOR EACH ROW EXECUTE PROCEDURE update_timestamp();

CREATE TRIGGER update_forex_optimizer_jobs_timestamp
BEFORE UPDATE ON forex_optimizer_jobs
FOR EACH ROW EXECUTE PROCEDURE update_timestamp();

CREATE TRIGGER update_system_status_timestamp
BEFORE UPDATE ON system_status
FOR EACH ROW EXECUTE PROCEDURE update_timestamp();

CREATE TRIGGER update_trades_timestamp
BEFORE UPDATE ON trades
FOR EACH ROW EXECUTE PROCEDURE update_timestamp(); 
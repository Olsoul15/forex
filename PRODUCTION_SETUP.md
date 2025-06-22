# Forex AI Trading System Production Setup Guide

This guide explains how to set up the Forex AI Trading System for production use, ensuring your clients **never** receive mock data.

## Production Environment Setup

### 1. Supabase Setup

1. Create a Supabase account at https://supabase.com/
2. Create a new project
3. Get your Supabase URL and API key from the project settings
4. Initialize the database schema:
   ```bash
   python setup_supabase_connection.py --init-schema
   ```

### 2. Environment Configuration

Create a `.env` file in the project root with the following content:

```
# Supabase Configuration
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-supabase-api-key

# Server Configuration
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
```

**IMPORTANT:** Never set `FOREX_AI_DEV_MODE=true` in production as this enables mock data.

### 3. Starting the Server in Production Mode

#### Windows (PowerShell)

Use the provided production startup script:

```powershell
.\run_production_server.ps1
```

#### Linux/Mac (Bash)

Use the provided production startup script:

```bash
chmod +x run_production_server.sh
./run_production_server.sh
```

Both scripts will:
1. Explicitly disable development mode
2. Verify that valid Supabase credentials are provided
3. Start the server in production mode

## How the Environment Works

The system determines whether to use real or mock data based on:

1. **Development Mode Setting**: If `FOREX_AI_DEV_MODE` is set to `true`, the system may use mock data
2. **Supabase Credentials**: Valid credentials must be provided in production

### Code Logic

In `forex_ai/auth/supabase.py`, the system:

1. Tries to connect to Supabase using the provided credentials
2. If the connection fails and `FOREX_AI_DEV_MODE` is `true`, it falls back to mock data
3. If the connection fails and `FOREX_AI_DEV_MODE` is not `true`, it raises an exception

This ensures that in production, the system will either:
- Connect to a real Supabase database, or
- Fail to start (rather than silently using mock data)

## Verifying Production Mode

To verify the system is using real data:

1. Check the server logs at startup - it should show:
   ```
   Initializing Supabase client with URL: https://your-project-id.supabase.co
   Dev mode enabled: False
   Successfully connected to Supabase
   ```

2. If you see any messages about "mock client" or "falling back to mock data", the system is NOT in production mode.

## Troubleshooting

If the server fails to start in production mode:

1. Verify your Supabase credentials are correct
2. Check that the Supabase project is active
3. Ensure the database schema has been initialized
4. Verify that `FOREX_AI_DEV_MODE` is not set to `true`

## Security Considerations

- Store your `.env` file securely and never commit it to version control
- Use a service role key with appropriate permissions for production
- Consider using environment-specific configuration for different deployment environments 
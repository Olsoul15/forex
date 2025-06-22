@echo off
echo Starting Forex AI Trading System API Server...
uvicorn forex_ai.api.main:app --host 0.0.0.0 --port 8000 --reload
pause 
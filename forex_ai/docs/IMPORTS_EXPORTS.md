# Import/Export Reference Guide

This document provides guidance on importing and exporting data in the Forex AI Trading System.

## Data Import

### Market Data Import

The system supports importing market data from various sources:

1. **Alpha Vantage API** - Automated through N8N workflows
   ```python
   from forex_ai.data.connectors.alpha_vantage import AlphaVantageConnector
   
   connector = AlphaVantageConnector()
   data = connector.fetch_historical_data(currency_pair="EUR/USD", timeframe="1h", lookback_days=30)
   ```

2. **TradingView** - Real-time and historical data
   ```python
   from forex_ai.data.connectors.trading_view import TradingViewConnector
   
   connector = TradingViewConnector()
   data = connector.fetch_real_time_data(currency_pair="EUR/USD")
   ```

3. **CSV Files** - Manual upload through dashboard or API
   ```python
   from forex_ai.data.pipelines.market_data import import_from_csv
   
   data = import_from_csv(file_path="path/to/eurusd_data.csv", timeframe="1h")
   ```

### Strategy Import

The system supports importing trading strategies in several formats:

1. **Pine Script Files** - TradingView Pine Script strategies
   ```python
   from forex_ai.analysis.technical.pine_script.manager import import_pine_script
   
   strategy = import_pine_script(file_path="path/to/strategy.pine")
   ```

2. **JSON Strategy Definitions** - Standard format for strategy parameters
   ```python
   from forex_ai.analysis.technical.pine_script.manager import import_strategy_json
   
   strategy = import_strategy_json(file_path="path/to/strategy.json")
   ```

3. **Python Strategy Modules** - Custom Python strategy implementations
   ```python
   from forex_ai.analysis.technical.pine_script.manager import import_python_strategy
   
   strategy = import_python_strategy(module_name="my_strategy_module")
   ```

## Data Export

### Market Data Export

The system supports exporting market data in several formats:

1. **CSV Export** - Historical data export
   ```python
   from forex_ai.data.pipelines.market_data import export_to_csv
   
   export_to_csv(data, file_path="path/to/exported_data.csv")
   ```

2. **JSON Export** - For interoperability
   ```python
   from forex_ai.data.pipelines.market_data import export_to_json
   
   export_to_json(data, file_path="path/to/exported_data.json")
   ```

3. **Excel Export** - For analysis in Excel
   ```python
   from forex_ai.data.pipelines.market_data import export_to_excel
   
   export_to_excel(data, file_path="path/to/exported_data.xlsx")
   ```

### Trading Results Export

The system supports exporting trading results for analysis:

1. **Performance Report** - Detailed analysis of strategy performance
   ```python
   from forex_ai.analysis.technical.pine_script.optimizer import export_performance_report
   
   export_performance_report(strategy_id="strategy_123", file_path="path/to/report.pdf")
   ```

2. **Trade History** - Historical trades for a strategy
   ```python
   from forex_ai.data.pipelines.market_data import export_trade_history
   
   export_trade_history(strategy_id="strategy_123", file_path="path/to/trades.csv")
   ```

3. **Optimization Results** - Results from strategy optimization
   ```python
   from forex_ai.analysis.technical.pine_script.optimizer import export_optimization_results
   
   export_optimization_results(optimization_id="opt_123", file_path="path/to/optimization.json")
   ```

## Data Transformation

Data transformation utilities are available for converting between formats:

```python
from forex_ai.data.pipelines.market_data import convert_timeframe
from forex_ai.data.pipelines.market_data import merge_data_sources
from forex_ai.data.pipelines.market_data import normalize_data

# Convert timeframe (e.g., from 1h to 4h)
converted_data = convert_timeframe(data, source_timeframe="1h", target_timeframe="4h")

# Merge data from multiple sources
merged_data = merge_data_sources([data1, data2, data3])

# Normalize data for machine learning
normalized_data = normalize_data(data)
```

## Import/Export APIs

The system provides RESTful APIs for importing and exporting data:

- `POST /api/v1/data/import` - Import data
- `GET /api/v1/data/export` - Export data
- `POST /api/v1/strategies/import` - Import strategies
- `GET /api/v1/strategies/export` - Export strategies

See the API documentation for detailed usage instructions. 
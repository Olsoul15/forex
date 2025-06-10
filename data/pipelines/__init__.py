"""
Data processing pipelines for the Forex AI Trading System.

This package provides data processing pipelines for:
- Market data processing
- News analysis and processing
- Data validation and cleaning
- Data transformation and normalization
"""

from forex_ai.data.pipelines.market_data import (
    fetch_market_data,
    import_from_csv,
    export_to_csv,
    export_to_json,
    export_to_excel,
    convert_timeframe,
    merge_data_sources,
    normalize_data,
    export_trade_history
)

__all__ = [
    "fetch_market_data",
    "import_from_csv",
    "export_to_csv",
    "export_to_json",
    "export_to_excel",
    "convert_timeframe",
    "merge_data_sources",
    "normalize_data",
    "export_trade_history"
] 
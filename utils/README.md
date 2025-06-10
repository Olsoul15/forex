# Utilities Module

This module provides specialized utility functions and helpers for the Forex AI Trading System.

## Overview

The utilities module contains a collection of specialized helper functions, tools, and utilities that support various components of the Forex AI Trading System. Unlike the common module, which focuses on shared core functionality, the utils module provides more specialized and targeted utilities for specific use cases.

## Key Components

### Forex Utilities

- **forex_utils.py**: Forex-specific utilities
  - Implements pip value calculation
  - Provides spread analysis functions
  - Includes market hours utilities
  - Offers currency pair information

### Technical Analysis

- **technical_analysis.py**: Technical analysis utilities
  - Implements standard technical indicators
  - Provides chart pattern recognition
  - Includes trend identification functions
  - Offers candlestick pattern detection

### Statistical Utilities

- **stats_utils.py**: Statistical analysis functions
  - Implements statistical measures and tests
  - Provides correlation and regression analysis
  - Includes distribution fitting functions
  - Offers statistical visualization helpers

### Data Processing

- **data_processing.py**: Data manipulation utilities
  - Implements data cleaning and normalization
  - Provides data transformation functions
  - Includes feature engineering helpers
  - Offers data validation utilities

### File Operations

- **file_utils.py**: File handling utilities
  - Implements data import/export functions
  - Provides file format conversion utilities
  - Includes file system operations
  - Offers file compression and archiving

### Visualization

- **visualization.py**: Visualization utilities
  - Implements chart generation functions
  - Provides data visualization helpers
  - Includes custom plot styles
  - Offers interactive visualization tools

## Usage Examples

### Forex Utilities

```python
from forex_ai.utils.forex_utils import (
    calculate_pip_value,
    is_market_open,
    get_major_pairs,
    calculate_swap_rates
)

# Calculate pip value
pip_value = calculate_pip_value(
    instrument="GBP_USD",
    units=100000,  # 1 standard lot
    account_currency="USD"
)
print(f"Pip value: ${pip_value}")

# Check if market is open
is_open = is_market_open("EUR_USD")
print(f"EUR/USD market is {'open' if is_open else 'closed'}")

# Get list of major currency pairs
majors = get_major_pairs()
print(f"Major pairs: {majors}")

# Calculate swap rates
swap_long, swap_short = calculate_swap_rates("AUD_USD")
print(f"Swap rates for AUD/USD: Long: {swap_long}, Short: {swap_short}")
```

### Technical Analysis

```python
from forex_ai.utils.technical_analysis import (
    calculate_rsi,
    calculate_moving_average,
    detect_support_resistance,
    identify_chart_patterns
)
import numpy as np

# Sample price data
close_prices = np.array([1.1050, 1.1052, 1.1057, 1.1065, 1.1070, 1.1068, 1.1063])

# Calculate RSI
rsi = calculate_rsi(close_prices, period=14)
print(f"RSI: {rsi}")

# Calculate moving average
ma = calculate_moving_average(close_prices, period=5, ma_type="EMA")
print(f"5-period EMA: {ma}")

# Detect support and resistance levels
support, resistance = detect_support_resistance(
    highs=np.array([1.1075, 1.1080, 1.1072, 1.1078, 1.1074]),
    lows=np.array([1.1045, 1.1048, 1.1040, 1.1042, 1.1039]),
    closes=close_prices
)
print(f"Support levels: {support}")
print(f"Resistance levels: {resistance}")

# Identify chart patterns
patterns = identify_chart_patterns(
    opens=np.array([1.1048, 1.1049, 1.1055, 1.1062, 1.1065, 1.1064, 1.1060]),
    highs=np.array([1.1055, 1.1057, 1.1063, 1.1068, 1.1075, 1.1069, 1.1065]),
    lows=np.array([1.1042, 1.1045, 1.1052, 1.1056, 1.1063, 1.1060, 1.1055]),
    closes=close_prices
)
print(f"Detected patterns: {patterns}")
```

### Statistical Utilities

```python
from forex_ai.utils.stats_utils import (
    calculate_correlation,
    calculate_sharpe_ratio,
    perform_linear_regression,
    test_normality
)
import numpy as np

# Sample data
returns_a = np.array([0.02, -0.01, 0.03, 0.01, -0.02, 0.02, 0.01])
returns_b = np.array([0.01, -0.01, 0.02, 0.01, -0.01, 0.01, 0.02])

# Calculate correlation
correlation = calculate_correlation(returns_a, returns_b)
print(f"Correlation: {correlation}")

# Calculate Sharpe ratio
sharpe_ratio = calculate_sharpe_ratio(returns_a, risk_free_rate=0.02/252)
print(f"Sharpe ratio: {sharpe_ratio}")

# Perform linear regression
slope, intercept, r_squared = perform_linear_regression(
    x=np.array([1, 2, 3, 4, 5, 6, 7]),
    y=returns_a
)
print(f"Regression results - Slope: {slope}, Intercept: {intercept}, R²: {r_squared}")

# Test for normality
is_normal, p_value = test_normality(returns_a)
print(f"Normality test - Normal: {is_normal}, p-value: {p_value}")
```

### Data Processing

```python
from forex_ai.utils.data_processing import (
    normalize_data,
    remove_outliers,
    create_features,
    resample_data
)
import numpy as np

# Sample data
prices = np.array([1.1050, 1.1052, 1.1057, 1.1065, 1.1070, 1.1068, 1.1063])

# Normalize data
normalized_prices = normalize_data(prices, method="z-score")
print(f"Normalized prices: {normalized_prices}")

# Remove outliers
clean_prices = remove_outliers(prices, method="iqr")
print(f"Prices without outliers: {clean_prices}")

# Create features
features = create_features(
    prices,
    feature_types=["returns", "lagged", "rolling_mean", "rolling_std"],
    window_sizes=[2, 3]
)
print(f"Generated features: {features}")

# Resample data
resampled_prices = resample_data(
    prices,
    timestamps=np.array(['2023-01-01 10:00', '2023-01-01 10:01', '2023-01-01 10:02',
                         '2023-01-01 10:03', '2023-01-01 10:04', '2023-01-01 10:05',
                         '2023-01-01 10:06']),
    target_frequency="3min"
)
print(f"Resampled prices: {resampled_prices}")
```

## Dependencies

- **NumPy**: For numerical operations
- **Pandas**: For data manipulation
- **SciPy**: For statistical functions
- **Matplotlib/Plotly**: For visualization
- **TA-Lib**: For technical indicators
- **scikit-learn**: For data processing and statistics 
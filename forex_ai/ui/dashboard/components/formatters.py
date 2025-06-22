"""
Formatters for dashboard components.

This module provides formatting functions for displaying data in the dashboard.
"""

from typing import Any, Dict, List, Union
from datetime import datetime


def format_currency(value: Union[float, int], currency: str = "USD", precision: int = 2) -> str:
    """
    Format a value as currency.

    Args:
        value: The value to format
        currency: The currency code
        precision: Number of decimal places

    Returns:
        Formatted currency string
    """
    if value is None:
        return "N/A"

    try:
        formatted = f"{float(value):.{precision}f}"
        if currency == "USD":
            return f"${formatted}"
        elif currency == "EUR":
            return f"€{formatted}"
        elif currency == "GBP":
            return f"£{formatted}"
        elif currency == "JPY":
            return f"¥{formatted}"
        else:
            return f"{formatted} {currency}"
    except (ValueError, TypeError):
        return "N/A"


def format_percentage(value: Union[float, int], precision: int = 2) -> str:
    """
    Format a value as a percentage.

    Args:
        value: The value to format
        precision: Number of decimal places

    Returns:
        Formatted percentage string
    """
    if value is None:
        return "N/A"

    try:
        return f"{float(value):.{precision}f}%"
    except (ValueError, TypeError):
        return "N/A"


def format_date(date: Union[datetime, str], format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format a date.

    Args:
        date: The date to format
        format_str: The format string

    Returns:
        Formatted date string
    """
    if date is None:
        return "N/A"

    try:
        if isinstance(date, str):
            date = datetime.fromisoformat(date.replace("Z", "+00:00"))
        return date.strftime(format_str)
    except (ValueError, TypeError):
        return "N/A"


def format_number(value: Union[float, int], precision: int = 2) -> str:
    """
    Format a number.

    Args:
        value: The value to format
        precision: Number of decimal places

    Returns:
        Formatted number string
    """
    if value is None:
        return "N/A"

    try:
        return f"{float(value):.{precision}f}"
    except (ValueError, TypeError):
        return "N/A"


def format_pips(value: Union[float, int], precision: int = 1) -> str:
    """
    Format a pip value.

    Args:
        value: The value to format
        precision: Number of decimal places

    Returns:
        Formatted pip string
    """
    if value is None:
        return "N/A"

    try:
        return f"{float(value):.{precision}f} pips"
    except (ValueError, TypeError):
        return "N/A" 
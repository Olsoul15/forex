"""
Template filters for the Forex AI Trading System dashboard.

This module provides custom filters for Jinja2 templates.
"""


def multiply(value, factor):
    """
    Multiply a value by a factor.

    Args:
        value: Value to multiply
        factor: Multiplication factor

    Returns:
        Multiplied value
    """
    if value is None:
        return None

    try:
        return float(value) * float(factor)
    except (ValueError, TypeError):
        return None


def round_value(value, digits=2):
    """
    Round a value to a specified number of digits.

    Args:
        value: Value to round
        digits: Number of decimal places

    Returns:
        Rounded value
    """
    if value is None:
        return None

    try:
        return round(float(value), digits)
    except (ValueError, TypeError):
        return None


def format_currency(value, currency="USD"):
    """
    Format a value as currency.

    Args:
        value: Value to format
        currency: Currency code

    Returns:
        Formatted currency string
    """
    if value is None:
        return None

    try:
        value = float(value)
        if currency == "USD":
            return f"${value:,.2f}"
        elif currency == "EUR":
            return f"€{value:,.2f}"
        elif currency == "GBP":
            return f"£{value:,.2f}"
        elif currency == "JPY":
            return f"¥{value:,.0f}"
        else:
            return f"{value:,.2f} {currency}"
    except (ValueError, TypeError):
        return None


def format_percentage(value, digits=2):
    """
    Format a value as a percentage.

    Args:
        value: Value to format (0-1)
        digits: Number of decimal places

    Returns:
        Formatted percentage string
    """
    if value is None:
        return None

    try:
        value = float(value) * 100.0
        return f"{value:.{digits}f}%"
    except (ValueError, TypeError):
        return None


def format_date(value, format="%Y-%m-%d %H:%M"):
    """
    Format a date value.

    Args:
        value: Date value to format
        format: Date format string

    Returns:
        Formatted date string
    """
    if value is None:
        return None

    try:
        if hasattr(value, "strftime"):
            return value.strftime(format)
        return str(value)
    except (ValueError, TypeError, AttributeError):
        return str(value)

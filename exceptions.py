"""
Custom exception hierarchy for the Forex AI Trading System.
"""


class ForexAiError(Exception):
    """Base exception for all Forex AI Trading System errors."""

    pass


class ValidationError(ForexAiError):
    """Base exception for validation errors."""

    pass


# Database Exceptions
class DatabaseError(ForexAiError):
    """Base exception for database-related errors."""

    pass


class DatabaseConnectionError(DatabaseError):
    """Raised when unable to connect to the database."""

    pass


class DatabaseQueryError(DatabaseError):
    """Raised when a database query fails."""

    pass


# Add the missing StrategyRepositoryError
class StrategyRepositoryError(DatabaseError):
    """Base exception for strategy repository-related errors."""

    pass


# Data Exceptions
class DataError(ForexAiError):
    """Base exception for data-related errors."""

    pass


class DataValidationError(DataError):
    """Raised when data validation fails."""

    pass


# Add missing DataFetchError
class DataFetchError(DataError):
    """Raised specifically when fetching data from a source fails."""

    pass


class DataSourceError(DataError):
    """Raised when there's an error with a data source."""

    pass


class DataProcessingError(DataError):
    """Raised when data processing fails."""

    pass


# Trading Exceptions
class TradingError(ForexAiError):
    """Base exception for trading-related errors."""

    pass


class OrderExecutionError(TradingError):
    """Raised when order execution fails."""

    pass


class StrategyError(TradingError):
    """Raised when a trading strategy encounters an error."""

    pass


class PositionManagementError(TradingError):
    """Raised when position management encounters an error."""

    pass


# API Exceptions
class ApiError(ForexAiError):
    """Base exception for API-related errors."""

    pass


class ApiConnectionError(ApiError):
    """Raised when unable to connect to an external API."""

    pass


class ApiResponseError(ApiError):
    """Raised when an API response is invalid or contains an error."""

    pass


class ApiRateLimitError(ApiError):
    """Raised when an API rate limit is exceeded."""

    pass


# Configuration Exceptions
class ConfigurationError(ForexAiError):
    """Raised when there's an error in configuration settings."""

    pass


# Authentication Exceptions
class AuthenticationError(ForexAiError):
    """Raised when authentication fails."""

    pass


# Agent Exceptions
class AgentError(ForexAiError):
    """Base exception for agent-related errors."""

    pass


class AgentCommunicationError(AgentError):
    """Raised when communication between agents fails."""

    pass


class AgentExecutionError(AgentError):
    """Raised when agent execution fails."""

    pass


class ToolExecutionError(AgentError):
    """Raised when a tool execution fails."""

    pass


class AgentToolError(AgentError):
    """Exception raised for errors in agent tools."""

    pass


# Workflow Exceptions
class WorkflowError(ForexAiError):
    """Base exception for workflow-related errors."""

    pass


# Analysis Exceptions
class AnalysisError(ForexAiError):
    """Base exception for analysis-related errors."""

    pass


class TechnicalAnalysisError(AnalysisError):
    """Raised when technical analysis fails."""

    pass


class FundamentalAnalysisError(AnalysisError):
    """Raised when fundamental analysis fails."""

    pass


class PineScriptError(AnalysisError):
    """Raised when Pine Script processing fails."""

    pass


class PatternError(TechnicalAnalysisError):
    """Raised when pattern recognition fails."""

    pass


class PatternAnalysisError(TechnicalAnalysisError):
    """Raised when pattern analysis fails."""

    pass


class IndicatorError(TechnicalAnalysisError):
    """Raised when indicator calculation fails."""

    pass


class OptimizationError(AnalysisError):
    """Raised when strategy optimization fails."""

    pass


class StrategyNotFoundError(StrategyError):
    """Raised when a strategy cannot be found."""

    pass


class StrategyValidationError(StrategyError):
    """Raised when strategy validation fails."""

    pass


# Model Exceptions
class ModelError(ForexAiError):
    """Base exception for model-related errors."""

    pass


class ModelInferenceError(ModelError):
    """Raised when model inference fails."""

    pass


class ModelLoadingError(ModelError):
    """Raised when model loading fails."""

    pass


# Cache Exceptions
class CacheError(ForexAiError):
    """Base exception for cache-related errors."""

    pass


class CacheMissError(CacheError):
    """Raised when a cache key is not found."""

    pass


class CacheWriteError(CacheError):
    """Raised when unable to write to cache."""

    pass


# System Exceptions
class SystemError(ForexAiError):
    """Base exception for system-related errors."""

    pass


class ResourceExhaustionError(SystemError):
    """Raised when system resources are exhausted."""

    pass


class ConcurrencyError(SystemError):
    """Raised when a concurrency issue occurs."""

    pass


class TimeoutError(SystemError):
    """Raised when an operation times out."""

    pass


# Validation Utilities
def validate_or_raise(condition: bool, exception_class: type, message: str) -> None:
    """
    Validate a condition and raise an exception if it fails.

    Args:
        condition: Condition to validate
        exception_class: Exception class to raise
        message: Exception message

    Raises:
        exception_class: If condition is False
    """
    if not condition:
        raise exception_class(message)


# Dashboard Exceptions
class DashboardError(ForexAiError):
    """Base exception for dashboard errors."""

    pass


class ChartError(DashboardError):
    """Raised when chart rendering fails."""

    pass


class PerformanceError(DashboardError):
    """Raised when performance data operations fail."""

    pass


class SignalError(DashboardError):
    """Raised when signal operations fail."""

    pass


class SentimentComponentError(DashboardError):
    """Raised when sentiment component operations fail."""

    pass


# API Exceptions
class APIError(ForexAiError):
    """Base exception for API errors."""

    pass


class RateLimitError(APIError):
    """Raised when API rate limit is exceeded."""

    pass


class EndpointError(APIError):
    """Raised when an API endpoint is invalid or not found."""

    pass


# Backtesting Exceptions
class BacktestingError(ForexAiError):
    """Raised when backtesting operations fail."""

    pass


# Add missing BacktestExecutionError
class BacktestExecutionError(BacktestingError):
    """Raised when the backtest execution itself fails."""

    pass


# Cache and Message Broker Exceptions
class MessageBrokerError(ForexAiError):
    """Base exception for message broker-related errors."""

    pass


class PublishError(MessageBrokerError):
    """Raised when unable to publish a message."""

    pass


class SubscriptionError(MessageBrokerError):
    """Raised when there's an error with message subscription."""

    pass


# Agent framework exceptions
class InvalidDataError(DataError):
    """Exception raised for invalid data inputs."""

    pass


class BacktestResultNotFoundError(DatabaseError):
    """Backtest result not found in the repository."""

    pass

"""
Component-specific health checks.

This module provides specialized health checks for different types of components
in the Forex AI system, including database connections, APIs, models, and agents.
"""

import logging
import time
import os
import traceback
from typing import Dict, Any, List, Optional, Callable
import importlib
import requests
from datetime import datetime, timedelta

from forex_ai.health.base import HealthCheck, HealthStatus, ComponentHealth

# Configure logging
logger = logging.getLogger(__name__)


# We'll use a lazy import approach to avoid circular imports
def _get_model_controller():
    """Lazy import of model controller to avoid circular import"""
    from forex_ai.models.controller import get_model_controller

    return get_model_controller()


class DatabaseHealthCheck(HealthCheck):
    """
    Health check for database connections.

    Verifies that the database is accessible and functioning properly.
    """

    def __init__(
        self,
        db_module_path: str,
        connection_func_name: str = "get_connection",
        test_query: str = "SELECT 1",
        check_interval: float = 60.0,
        timeout: float = 5.0,
        recovery_strategy: Optional[Callable] = None,
    ):
        """
        Initialize the database health check.

        Args:
            db_module_path: Import path for the database module
            connection_func_name: Name of the function to get a database connection
            test_query: SQL query to test the connection
            check_interval: How often to run the check (in seconds)
            timeout: Maximum time a check should take (in seconds)
            recovery_strategy: Optional function to call for recovery
        """
        super().__init__(
            name="database",
            check_interval=check_interval,
            timeout=timeout,
            dependencies=[],
            recovery_strategy=recovery_strategy,
        )

        self.db_module_path = db_module_path
        self.connection_func_name = connection_func_name
        self.test_query = test_query
        self.connection_latency = []  # Track connection times

    def _check_implementation(self) -> Dict[str, Any]:
        """
        Check database health by testing connection and query execution.

        Returns:
            Dict with database health check results
        """
        try:
            # Import the database module
            db_module = importlib.import_module(self.db_module_path)
            connection_func = getattr(db_module, self.connection_func_name)

            # Measure connection time
            start_time = time.time()
            connection = connection_func()
            connection_time = time.time() - start_time

            # Execute test query
            query_start_time = time.time()
            cursor = connection.cursor()
            cursor.execute(self.test_query)
            cursor.fetchone()
            query_time = time.time() - query_start_time

            # Clean up
            cursor.close()

            try:
                # Try to properly close connections, but don't fail health check if this fails
                connection.close()
            except Exception as e:
                logger.warning(f"Error closing database connection: {str(e)}")

            # Update connection latency history (keep last 10)
            self.connection_latency.append(connection_time)
            if len(self.connection_latency) > 10:
                self.connection_latency = self.connection_latency[-10:]

            # Calculate average latency
            avg_latency = sum(self.connection_latency) / len(self.connection_latency)

            # Check if latency is too high (more than 1 second)
            is_degraded = avg_latency > 1.0

            return {
                "status": "success",
                "message": "Database connection successful",
                "is_degraded": is_degraded,
                "details": {
                    "connection_time": connection_time,
                    "query_time": query_time,
                    "avg_latency": avg_latency,
                    "test_query": self.test_query,
                },
            }

        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
            return {
                "status": "failure",
                "message": f"Database connection failed: {str(e)}",
                "is_degraded": False,  # Database failure is not degraded, it's unhealthy
                "details": {
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "module_path": self.db_module_path,
                },
            }


class APIHealthCheck(HealthCheck):
    """
    Health check for external API endpoints.

    Verifies that an external API is accessible and responding properly.
    """

    def __init__(
        self,
        name: str,
        endpoint_url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, str]] = None,
        test_response_key: Optional[str] = None,
        expected_status_code: int = 200,
        check_interval: float = 300.0,  # External APIs checked less frequently
        timeout: float = 30.0,
        recovery_strategy: Optional[Callable] = None,
    ):
        """
        Initialize the API health check.

        Args:
            name: Name of the API (e.g., "alpha_vantage")
            endpoint_url: URL to check
            method: HTTP method to use
            headers: Optional headers to include
            params: Optional query parameters
            test_response_key: Optional key to verify in the response
            expected_status_code: Expected HTTP status code
            check_interval: How often to run the check (in seconds)
            timeout: Maximum time a check should take (in seconds)
            recovery_strategy: Optional function to call for recovery
        """
        super().__init__(
            name=f"api_{name}",
            check_interval=check_interval,
            timeout=timeout,
            dependencies=[],
            recovery_strategy=recovery_strategy,
        )

        self.endpoint_url = endpoint_url
        self.method = method.upper()
        self.headers = headers or {}
        self.params = params or {}
        self.test_response_key = test_response_key
        self.expected_status_code = expected_status_code
        self.response_times = []  # Track response times

        # Add user agent if not provided
        if "User-Agent" not in self.headers:
            self.headers["User-Agent"] = "Forex-AI-HealthCheck/1.0"

    def _check_implementation(self) -> Dict[str, Any]:
        """
        Check API health by making a request and verifying the response.

        Returns:
            Dict with API health check results
        """
        try:
            # Make the request
            start_time = time.time()
            response = requests.request(
                method=self.method,
                url=self.endpoint_url,
                headers=self.headers,
                params=self.params,
                timeout=self.timeout,
            )
            response_time = time.time() - start_time

            # Update response time history (keep last 10)
            self.response_times.append(response_time)
            if len(self.response_times) > 10:
                self.response_times = self.response_times[-10:]

            # Calculate average latency
            avg_latency = sum(self.response_times) / len(self.response_times)

            # Check status code
            status_code_ok = response.status_code == self.expected_status_code

            # Try to parse JSON response if expected
            response_data = None
            response_key_ok = True

            if self.test_response_key:
                try:
                    response_data = response.json()
                    response_key_ok = self.test_response_key in response_data
                except Exception as e:
                    logger.warning(f"Could not parse JSON response: {str(e)}")
                    response_key_ok = False

            # Determine overall status
            if status_code_ok and response_key_ok:
                # Check if latency is high (more than 2 seconds for API)
                is_degraded = avg_latency > 2.0

                return {
                    "status": "success",
                    "message": "API endpoint is responsive",
                    "is_degraded": is_degraded,
                    "details": {
                        "response_time": response_time,
                        "avg_latency": avg_latency,
                        "status_code": response.status_code,
                        "response_size": len(response.content),
                        "endpoint_url": self.endpoint_url,
                    },
                }
            else:
                # Determine the specific issue
                if not status_code_ok:
                    message = (
                        f"API returned unexpected status code: {response.status_code}"
                    )
                    details = {
                        "expected_status_code": self.expected_status_code,
                        "actual_status_code": response.status_code,
                        "response_content": response.text[
                            :1000
                        ],  # Truncate long responses
                    }
                else:
                    message = (
                        f"API response missing expected key: {self.test_response_key}"
                    )
                    details = {
                        "expected_key": self.test_response_key,
                        "response_keys": (
                            list(response_data.keys()) if response_data else []
                        ),
                    }

                return {
                    "status": "failure",
                    "message": message,
                    "is_degraded": True,  # API issues are often temporary, mark as degraded
                    "details": {
                        **details,
                        "response_time": response_time,
                        "endpoint_url": self.endpoint_url,
                    },
                }

        except requests.RequestException as e:
            logger.error(f"API health check failed for {self.endpoint_url}: {str(e)}")
            return {
                "status": "failure",
                "message": f"API request failed: {str(e)}",
                "is_degraded": True,  # Network issues are often temporary
                "details": {
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "endpoint_url": self.endpoint_url,
                },
            }
        except Exception as e:
            logger.error(f"Unexpected error in API health check: {str(e)}")
            return {
                "status": "failure",
                "message": f"Unexpected error in API health check: {str(e)}",
                "is_degraded": False,  # Unexpected errors are unhealthy, not degraded
                "details": {
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "endpoint_url": self.endpoint_url,
                },
            }


class ModelHealthCheck(HealthCheck):
    """
    Health check for AI models.

    Verifies that models are loaded correctly and can make predictions.
    Uses the centralized ModelController for model management.
    """

    def __init__(
        self,
        name: str,
        model_id: str,
        test_input: Any,
        check_interval: float = 300.0,  # Models checked less frequently
        timeout: float = 30.0,
        recovery_strategy: Optional[Callable] = None,
    ):
        """
        Initialize the model health check.

        Args:
            name: Name of the model (e.g., "sentiment_model")
            model_id: ID of the model in the ModelController
            test_input: Input to use for testing the model
            check_interval: How often to run the check (in seconds)
            timeout: Maximum time a check should take (in seconds)
            recovery_strategy: Optional function to call for recovery
        """
        super().__init__(
            name=f"model_{name}",
            check_interval=check_interval,
            timeout=timeout,
            dependencies=[],
            recovery_strategy=recovery_strategy,
        )

        self.model_id = model_id
        self.test_input = test_input
        self.model_controller = _get_model_controller()

    def _check_implementation(self) -> Dict[str, Any]:
        """
        Check model health by verifying its status and metrics.
        Uses the ModelController for model management.

        Returns:
            Dict with model health check results
        """
        try:
            # Get model status from controller
            status = self.model_controller.get_model_status(self.model_id)
            metrics = status["metrics"]

            # Check if model is degraded
            is_degraded = metrics["is_degraded"]

            # Verify model is loaded and can make predictions
            if not status["loaded"]:
                return {
                    "status": "failure",
                    "message": "Model is not loaded",
                    "is_degraded": False,
                    "details": {"model_id": self.model_id, "metrics": metrics},
                }

            # Get model instance and make a test prediction
            model = self.model_controller.get_model(self.model_id)
            start_time = time.time()
            model.predict(self.test_input)
            inference_time = time.time() - start_time

            return {
                "status": "success",
                "message": "Model loaded and inference successful",
                "is_degraded": is_degraded,
                "details": {
                    "model_load_time": metrics["model_load_time"],
                    "inference_time": inference_time,
                    "avg_inference_time": metrics["avg_response_time"],
                    "success_rate": metrics["success_rate"],
                    "calls": metrics["calls"],
                    "errors": metrics["errors"],
                    "last_call_time": metrics["last_call_time"],
                    "model_id": self.model_id,
                },
            }

        except Exception as e:
            logger.error(f"Model health check failed for {self.model_id}: {str(e)}")
            return {
                "status": "failure",
                "message": f"Model health check failed: {str(e)}",
                "is_degraded": False,  # Model failure is unhealthy, not degraded
                "details": {
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "model_id": self.model_id,
                },
            }


class AgentHealthCheck(HealthCheck):
    """
    Health check for agent components.

    Verifies that agents are operational and can execute their tasks.
    Uses the ModelController for managing the agent's underlying model.
    """

    def __init__(
        self,
        name: str,
        agent_id: str,
        model_id: str,
        test_input: Any,
        expected_output_key: Optional[str] = None,
        check_interval: float = 300.0,  # Agents checked less frequently
        timeout: float = 30.0,
        recovery_strategy: Optional[Callable] = None,
    ):
        """
        Initialize the agent health check.

        Args:
            name: Name of the agent (e.g., "sentiment_agent")
            agent_id: ID of the agent instance
            model_id: ID of the model used by the agent
            test_input: Input to use for testing the agent
            expected_output_key: Optional key to verify in the agent output
            check_interval: How often to run the check (in seconds)
            timeout: Maximum time a check should take (in seconds)
            recovery_strategy: Optional function to call for recovery
        """
        super().__init__(
            name=f"agent_{name}",
            check_interval=check_interval,
            timeout=timeout,
            dependencies=[],
            recovery_strategy=recovery_strategy,
        )

        self.agent_id = agent_id
        self.model_id = model_id
        self.test_input = test_input
        self.expected_output_key = expected_output_key
        self.execution_times = []  # Track execution times
        self.model_controller = _get_model_controller()

    def _check_implementation(self) -> Dict[str, Any]:
        """
        Check agent health by verifying its model status and executing a test task.
        Uses the ModelController for model management.

        Returns:
            Dict with agent health check results
        """
        try:
            # First check model health
            model_status = self.model_controller.get_model_status(self.model_id)
            if not model_status["loaded"]:
                return {
                    "status": "failure",
                    "message": f"Agent's model {self.model_id} is not loaded",
                    "is_degraded": False,
                    "details": {
                        "agent_id": self.agent_id,
                        "model_id": self.model_id,
                        "model_metrics": model_status["metrics"],
                    },
                }

            # Get model instance
            model = self.model_controller.get_model(self.model_id)

            # Execute test task
            execution_start = time.time()
            result = model.predict(self.test_input)
            execution_time = time.time() - execution_start

            # Update execution time history (keep last 10)
            self.execution_times.append(execution_time)
            if len(self.execution_times) > 10:
                self.execution_times = self.execution_times[-10:]

            # Calculate average execution time
            avg_execution_time = sum(self.execution_times) / len(self.execution_times)

            # Check for expected output key if specified
            output_key_ok = True
            if self.expected_output_key:
                if isinstance(result, dict):
                    output_key_ok = self.expected_output_key in result
                else:
                    output_key_ok = False
                    logger.warning(
                        f"Agent result is not a dict, cannot check for key {self.expected_output_key}"
                    )

            # Check if execution time is too high (more than 5 seconds)
            is_degraded = avg_execution_time > 5.0 or not output_key_ok

            # Get model metrics
            metrics = model_status["metrics"]

            return {
                "status": "success",
                "message": "Agent operational and test execution successful",
                "is_degraded": is_degraded,
                "details": {
                    "agent_id": self.agent_id,
                    "model_id": self.model_id,
                    "execution_time": execution_time,
                    "avg_execution_time": avg_execution_time,
                    "model_metrics": {
                        "success_rate": metrics["success_rate"],
                        "calls": metrics["calls"],
                        "errors": metrics["errors"],
                        "avg_response_time": metrics["avg_response_time"],
                        "last_call_time": metrics["last_call_time"],
                    },
                    "output_validation": {
                        "expected_key": self.expected_output_key,
                        "key_present": (
                            output_key_ok if self.expected_output_key else None
                        ),
                    },
                },
            }

        except Exception as e:
            logger.error(f"Agent health check failed for {self.agent_id}: {str(e)}")
            return {
                "status": "failure",
                "message": f"Agent health check failed: {str(e)}",
                "is_degraded": False,  # Agent failure is unhealthy, not degraded
                "details": {
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "agent_id": self.agent_id,
                    "model_id": self.model_id,
                },
            }

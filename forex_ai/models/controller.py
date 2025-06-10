"""
Model Controller for the Forex AI Trading System.

This module provides centralized management of all AI models used in the system,
including lifecycle management, versioning, monitoring, and error handling.
"""

import logging
from typing import Dict, Any, Optional, Type, List
from datetime import datetime
import importlib
import threading
import time
import traceback
from pathlib import Path

from forex_ai.exceptions import ModelError, ModelLoadingError, ModelInferenceError

# Importing ModelHealthCheck within a function instead to avoid circular imports
# from forex_ai.health.component import ModelHealthCheck
from forex_ai.health.locks import LockManager
from forex_ai.config.settings import get_settings

# Removed unused Agent import causing circular dependency
# from forex_ai.agents.framework.agent import Agent
# from forex_ai.execution.broker_api import TradingInterface
# from forex_ai.data.data_provider import DataProvider # Removed unused/missing import
from forex_ai.custom_types import AgentMetrics

logger = logging.getLogger(__name__)


class ModelMetrics(AgentMetrics):
    """
    Extended metrics for model performance monitoring.
    Inherits from AgentMetrics for compatibility.
    """

    def __init__(self):
        super().__init__()
        self.inference_times: List[float] = []  # Last 10 inference times
        self.model_load_time: Optional[float] = None
        self.last_reload_time: Optional[datetime] = None
        self.is_degraded: bool = False

    def update_inference_time(self, inference_time: float):
        """Update inference time history."""
        self.inference_times.append(inference_time)
        if len(self.inference_times) > 10:
            self.inference_times = self.inference_times[-10:]

        # Update average response time
        self.avg_response_time = sum(self.inference_times) / len(self.inference_times)

        # Check for degraded performance (over 2 seconds)
        self.is_degraded = self.avg_response_time > 2.0


class ModelController:
    """
    Centralized controller for AI model management.

    This class handles:
    - Model lifecycle (loading, unloading, reloading)
    - Model versioning
    - Performance monitoring
    - Error handling and recovery
    - Model caching
    """

    def __init__(self):
        """Initialize the model controller."""
        self.settings = get_settings()
        self.lock_manager = LockManager()
        self._models: Dict[str, Any] = {}  # Loaded models
        self._model_configs: Dict[str, Dict[str, Any]] = {}  # Model configurations
        self._model_metrics: Dict[str, ModelMetrics] = {}  # Model performance metrics
        self._health_checks: Dict[str, Any] = {}  # Health checks
        self._lock = threading.RLock()

        # Load model configurations
        self._load_model_configs()

    def _load_model_configs(self):
        """Load model configurations from settings."""
        self._model_configs = {
            "reasoning": {
                "name": self.settings.REASONING_MODEL,
                "type": "llm",
                "module_path": "forex_ai.models.llm",
                "class_name": "ReasoningModel",
            },
            "chat": {
                "name": self.settings.CHAT_MODEL,
                "type": "llm",
                "module_path": "forex_ai.models.llm",
                "class_name": "ChatModel",
            },
            "vision": {
                "name": self.settings.VISION_MODEL,
                "type": "vision",
                "module_path": "forex_ai.models.vision",
                "class_name": "VisionModel",
            },
            "embedding": {
                "name": self.settings.EMBEDDING_MODEL,
                "type": "embedding",
                "module_path": "forex_ai.models.embedding",
                "class_name": "EmbeddingModel",
            },
        }

    def get_model(self, model_id: str) -> Any:
        """
        Get a model instance, loading it if necessary.

        Args:
            model_id: Model identifier

        Returns:
            Model instance

        Raises:
            ModelError: If model cannot be loaded or is not configured
        """
        with self._lock:
            # Check if model is already loaded
            if model_id in self._models:
                return self._models[model_id]

            # Check if model is configured
            if model_id not in self._model_configs:
                raise ModelError(f"Model {model_id} is not configured")

            # Check if model is locked
            lock = self.lock_manager.get_lock(f"model_{model_id}")
            if lock and lock.is_locked():
                raise ModelError(f"Model {model_id} is currently locked")

            # Load the model
            try:
                model = self._load_model(model_id)
                self._models[model_id] = model
                return model
            except Exception as e:
                raise ModelLoadingError(f"Failed to load model {model_id}: {str(e)}")

    def _load_model(self, model_id: str) -> Any:
        """Load a model from its configuration."""
        config = self._model_configs[model_id]
        metrics = self._get_or_create_metrics(model_id)

        try:
            # Record load start time
            load_start = time.time()

            # Import the model module
            module = importlib.import_module(config["module_path"])
            model_class = getattr(module, config["class_name"])

            # Initialize the model
            model = model_class(name=config["name"])

            # Update metrics
            metrics.model_load_time = time.time() - load_start
            metrics.last_reload_time = datetime.now()

            # Create or update health check
            self._ensure_health_check(model_id, config)

            return model

        except Exception as e:
            logger.error(f"Error loading model {model_id}: {str(e)}")
            metrics.errors += 1
            raise

    def _get_or_create_metrics(self, model_id: str) -> ModelMetrics:
        """Get or create metrics for a model."""
        if model_id not in self._model_metrics:
            self._model_metrics[model_id] = ModelMetrics()
        return self._model_metrics[model_id]

    def _ensure_health_check(self, model_id: str, config: Dict[str, Any]):
        """Ensure health check exists for model."""
        if model_id not in self._health_checks:
            # Import ModelHealthCheck here to avoid circular imports
            from forex_ai.health.component import ModelHealthCheck

            self._health_checks[model_id] = ModelHealthCheck(
                name=model_id,
                model_module_path=config["module_path"],
                model_class_name=config["class_name"],
                test_input="test",  # Should be configured per model type
            )

    def unload_model(self, model_id: str) -> bool:
        """
        Unload a model from memory.

        Args:
            model_id: Model identifier

        Returns:
            Whether the model was unloaded
        """
        with self._lock:
            if model_id in self._models:
                del self._models[model_id]
                return True
            return False

    def reload_model(self, model_id: str) -> Any:
        """
        Reload a model.

        Args:
            model_id: Model identifier

        Returns:
            Reloaded model instance

        Raises:
            ModelError: If model cannot be reloaded
        """
        with self._lock:
            self.unload_model(model_id)
            return self.get_model(model_id)

    def get_model_status(self, model_id: str) -> Dict[str, Any]:
        """
        Get current status of a model.

        Args:
            model_id: Model identifier

        Returns:
            Dictionary with model status information
        """
        with self._lock:
            metrics = self._get_or_create_metrics(model_id)

            status = {
                "id": model_id,
                "loaded": model_id in self._models,
                "config": self._model_configs.get(model_id, {}),
                "metrics": {
                    "calls": metrics.calls,
                    "errors": metrics.errors,
                    "avg_response_time": metrics.avg_response_time,
                    "last_call_time": metrics.last_call_time,
                    "success_rate": metrics.success_rate,
                    "model_load_time": metrics.model_load_time,
                    "last_reload_time": metrics.last_reload_time,
                    "is_degraded": metrics.is_degraded,
                },
                "health": None,
                "lock_status": None,
            }

            # Add health check status if available
            if model_id in self._health_checks:
                status["health"] = self._health_checks[model_id].check_health().dict()

            # Add lock status if available
            lock = self.lock_manager.get_lock(f"model_{model_id}")
            if lock:
                status["lock_status"] = lock.get_status()

            return status

    def get_all_model_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all models.

        Returns:
            Dictionary mapping model IDs to their status information
        """
        return {
            model_id: self.get_model_status(model_id)
            for model_id in self._model_configs
        }

    def update_metrics(self, model_id: str, success: bool, inference_time: float):
        """
        Update performance metrics for a model.

        Args:
            model_id: Model identifier
            success: Whether the operation was successful
            inference_time: Time taken for inference
        """
        with self._lock:
            metrics = self._get_or_create_metrics(model_id)
            metrics.calls += 1
            if not success:
                metrics.errors += 1
            metrics.update_inference_time(inference_time)
            metrics.last_call_time = datetime.now()
            metrics.success_rate = (metrics.calls - metrics.errors) / metrics.calls

    def handle_model_error(self, model_id: str, error: Exception) -> bool:
        """
        Handle a model error, potentially triggering recovery actions.

        Args:
            model_id: Model identifier
            error: The error that occurred

        Returns:
            Whether recovery was successful
        """
        logger.error(f"Model error for {model_id}: {str(error)}")

        # Get or create error lock
        lock = self.lock_manager.get_lock(f"model_{model_id}")
        if not lock:
            lock = self.lock_manager.create_lock(
                f"model_{model_id}",
                lock_threshold=3,  # Three strikes
                lock_duration=300.0,  # 5 minutes
            )

        # Record the error
        lock.record_failure()

        # If not locked, try to recover
        if not lock.is_locked():
            try:
                self.reload_model(model_id)
                lock.record_success()
                return True
            except Exception as e:
                logger.error(f"Model recovery failed for {model_id}: {str(e)}")
                lock.record_failure()
                return False

        return False


# Singleton instance
_model_controller = None


def get_model_controller() -> ModelController:
    """Get the singleton model controller instance."""
    global _model_controller
    if _model_controller is None:
        _model_controller = ModelController()
    return _model_controller

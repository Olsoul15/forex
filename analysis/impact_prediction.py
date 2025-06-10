"""
Market impact prediction for news and events.
"""

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

from forex_ai.core.exceptions import AnalysisError
from forex_ai.models.controller import get_model_controller

logger = logging.getLogger(__name__)


class ImpactPredictor:
    """Predicts market impact of news and events."""

    def __init__(self, model_id: str = "impact_predictor"):
        """
        Initialize the impact predictor.

        Args:
            model_id: ID of the model to use (default: "impact_predictor")
        """
        self.model_id = model_id
        self.model_controller = get_model_controller()

        # Register model configuration if not already registered
        if self.model_id not in self.model_controller._model_configs:
            self.model_controller._model_configs[self.model_id] = {
                "name": "impact_prediction",
                "type": "regression",
                "module_path": "forex_ai.analysis.impact_prediction",
                "class_name": "ImpactModel",
            }

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict market impact from features.

        Args:
            features: Dict of feature values

        Returns:
            Dict with impact predictions

        Raises:
            AnalysisError: If prediction fails
        """
        try:
            # Get model through controller
            model = self.model_controller.get_model(self.model_id)

            # Make prediction
            result = model.predict(features)

            return {
                "features": features,
                "impact": result,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Impact prediction failed: {str(e)}")
            raise AnalysisError(f"Failed to predict impact: {str(e)}")

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get model performance metrics.

        Returns:
            Dict with model metrics
        """
        return self.model_controller.get_model_status(self.model_id)["metrics"]

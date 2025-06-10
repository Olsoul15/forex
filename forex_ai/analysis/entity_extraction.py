"""
Entity extraction for financial text analysis.
"""

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

from forex_ai.core.exceptions import AnalysisError
from forex_ai.models.controller import get_model_controller

logger = logging.getLogger(__name__)


class EntityExtractor:
    """Extracts financial entities from text."""

    def __init__(self, model_id: str = "entity_extractor"):
        """
        Initialize the entity extractor.

        Args:
            model_id: ID of the model to use (default: "entity_extractor")
        """
        self.model_id = model_id
        self.model_controller = get_model_controller()

        # Register model configuration if not already registered
        if self.model_id not in self.model_controller._model_configs:
            self.model_controller._model_configs[self.model_id] = {
                "name": "en_core_web_sm",  # Default spaCy model
                "type": "ner",
                "module_path": "forex_ai.analysis.entity_extraction",
                "class_name": "EntityModel",
            }

    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extract entities from text.

        Args:
            text: Text to analyze

        Returns:
            Dict with extracted entities

        Raises:
            AnalysisError: If extraction fails
        """
        try:
            # Get model through controller
            model = self.model_controller.get_model(self.model_id)

            # Process text
            result = model.predict(text)

            return {
                "text": text,
                "entities": result["entities"],
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Entity extraction failed: {str(e)}")
            raise AnalysisError(f"Failed to extract entities: {str(e)}")

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get model performance metrics.

        Returns:
            Dict with model metrics
        """
        return self.model_controller.get_model_status(self.model_id)["metrics"]

"""
LLM Controller
Manages the initialization and access to various LLM providers.
"""

import logging
from typing import Dict, Any, Optional

from google.cloud import aiplatform
import google.auth

from forex_ai.config.settings import LLMSettings

# Configure logging
logger = logging.getLogger(__name__)


class LLMController:
    """
    Manages the initialization and access to the Google Vertex AI client.
    """

    def __init__(self, settings: Optional[LLMSettings] = None):
        """
        Initializes the LLMController and the Vertex AI client.

        Args:
            settings: An optional LLMSettings object. If not provided,
                      default settings will be used.
        """
        self.settings = settings or LLMSettings()
        self._clients: Dict[str, Any] = {}
        self._initialize_clients()

    def _initialize_clients(self) -> None:
        """
        Initializes the Google Vertex AI client.
        """
        try:
            # The Google client library uses ADC (Application Default Credentials)
            # It will automatically find credentials if they are configured in the environment
            # (e.g., via `gcloud auth application-default login`)
            credentials, project = google.auth.default()
            aiplatform.init(project=project, credentials=credentials, location=self.settings.GCP_LOCATION)
            self._clients["vertex_ai"] = aiplatform
            logger.info("Google Vertex AI client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Google Vertex AI client: {str(e)}")

    def get_client(self, provider: str = "vertex_ai") -> Optional[Any]:
        """
        Returns the client for the specified provider.

        Args:
            provider: The name of the LLM provider. Defaults to "vertex_ai".

        Returns:
            The client object if available, otherwise None.
        """
        if provider != "vertex_ai":
            logger.warning(f"Provider '{provider}' is not supported. Only 'vertex_ai' is available.")
            return None
        return self._clients.get(provider)

    def get_available_models(self) -> Dict[str, Any]:
        """
        Returns a dictionary of available models, configured for Vertex AI.
        """
        if "vertex_ai" not in self._clients:
            return {}

        # Example Vertex AI models
        # Users should configure these in their settings
        vertex_models = {
            "gemini-1.0-pro": {
                "name": "gemini-1.0-pro",
                "provider": "vertex_ai",
                "class_name": "VertexAIModel",  # Placeholder for your model handling class
            },
            "gemini-1.5-flash": {
                "name": "gemini-1.5-flash-001",
                "provider": "vertex_ai",
                "class_name": "VertexAIModel", # Placeholder for your model handling class
            },
        }
        return vertex_models

    def get_status(self) -> Dict[str, bool]:
        """
        Returns the connection status of the Vertex AI client.

        Returns:
            A dictionary with the provider name and its connection status.
        """
        return {"vertex_ai": "vertex_ai" in self._clients}

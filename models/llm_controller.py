"""
LLM Controller for managing different LLM providers in the Forex AI Trading System.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import importlib

from openai import OpenAI, AzureOpenAI
from groq import Groq
import openrouter

from forex_ai.models.controller import ModelController
from forex_ai.config.settings import get_settings
from forex_ai.config.llm_config import get_llm_config_from_env
from forex_ai.exceptions import ModelError, ModelLoadingError

logger = logging.getLogger(__name__)


class LLMController(ModelController):
    """
    Controller for managing LLM models from different providers.
    Supports Azure OpenAI, OpenAI, Groq, and OpenRouter.
    """

    def __init__(self):
        """Initialize the LLM controller."""
        super().__init__()
        self.settings = get_settings()
        self.llm_config = get_llm_config_from_env()
        self._clients: Dict[str, Any] = {}  # Provider clients

        # Initialize provider clients
        self._initialize_clients()

    def _initialize_clients(self):
        """Initialize API clients for each configured provider."""
        # Azure OpenAI
        if self.settings.AZURE_OPENAI_KEY and self.settings.AZURE_OPENAI_ENDPOINT:
            try:
                self._clients["azure"] = AzureOpenAI(
                    api_key=self.settings.AZURE_OPENAI_KEY,
                    azure_endpoint=self.settings.AZURE_OPENAI_ENDPOINT,
                    api_version=self.settings.AZURE_OPENAI_API_VERSION,
                )
                logger.info("Azure OpenAI client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Azure OpenAI client: {str(e)}")

        # OpenAI
        if self.settings.OPENAI_API_KEY:
            try:
                self._clients["openai"] = OpenAI(api_key=self.settings.OPENAI_API_KEY)
                logger.info("OpenAI client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {str(e)}")

        # Groq
        if self.settings.GROQ_API_KEY:
            try:
                self._clients["groq"] = Groq(api_key=self.settings.GROQ_API_KEY)
                logger.info("Groq client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {str(e)}")

        # OpenRouter
        if self.settings.OPENROUTER_API_KEY:
            try:
                openrouter.api_key = self.settings.OPENROUTER_API_KEY
                self._clients["openrouter"] = openrouter
                logger.info("OpenRouter client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize OpenRouter client: {str(e)}")

    def _load_model_configs(self):
        """Load LLM model configurations."""
        super()._load_model_configs()

        # Add LLM-specific configurations
        self._model_configs.update(
            {
                # Azure OpenAI models
                "gpt4": {
                    "name": "gpt-4",
                    "type": "llm",
                    "provider": "azure",
                    "deployment": "gpt4",
                    "module_path": "forex_ai.models.llm",
                    "class_name": "AzureGPT4Model",
                },
                "gpt35turbo": {
                    "name": "gpt-35-turbo",
                    "type": "llm",
                    "provider": "azure",
                    "deployment": "gpt35turbo",
                    "module_path": "forex_ai.models.llm",
                    "class_name": "AzureGPT35Model",
                },
                # OpenAI models
                "gpt4openai": {
                    "name": "gpt-4-turbo-preview",
                    "type": "llm",
                    "provider": "openai",
                    "module_path": "forex_ai.models.llm",
                    "class_name": "OpenAIModel",
                },
                # Groq models
                "mixtral": {
                    "name": "mixtral-8x7b-32768",
                    "type": "llm",
                    "provider": "groq",
                    "module_path": "forex_ai.models.llm",
                    "class_name": "GroqModel",
                },
                "llama2": {
                    "name": "llama2-70b-4096",
                    "type": "llm",
                    "provider": "groq",
                    "module_path": "forex_ai.models.llm",
                    "class_name": "GroqModel",
                },
                # OpenRouter models
                "claude3": {
                    "name": "anthropic/claude-3-sonnet",
                    "type": "llm",
                    "provider": "openrouter",
                    "module_path": "forex_ai.models.llm",
                    "class_name": "OpenRouterModel",
                },
            }
        )

    def _load_model(self, model_id: str) -> Any:
        """
        Load an LLM model with the appropriate provider client.

        Args:
            model_id: Model identifier

        Returns:
            Initialized model instance

        Raises:
            ModelLoadingError: If model cannot be loaded
        """
        config = self._model_configs[model_id]
        provider = config.get("provider")

        if not provider:
            raise ModelLoadingError(f"No provider specified for model {model_id}")

        if provider not in self._clients:
            raise ModelLoadingError(f"Provider {provider} not initialized")

        try:
            # Import model class
            module = importlib.import_module(config["module_path"])
            model_class = getattr(module, config["class_name"])

            # Initialize model with provider client
            model = model_class(
                name=config["name"],
                client=self._clients[provider],
                deployment=config.get("deployment"),  # Only used by Azure
            )

            return model

        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {str(e)}")
            raise ModelLoadingError(f"Failed to load model {model_id}: {str(e)}")

    def get_available_providers(self) -> Dict[str, bool]:
        """
        Get status of available providers.

        Returns:
            Dictionary mapping provider names to availability status
        """
        return {
            "azure": "azure" in self._clients,
            "openai": "openai" in self._clients,
            "groq": "groq" in self._clients,
            "openrouter": "openrouter" in self._clients,
        }

    def get_provider_models(self, provider: str) -> List[str]:
        """
        Get list of available models for a provider.

        Args:
            provider: Provider name

        Returns:
            List of model IDs available for the provider
        """
        return [
            model_id
            for model_id, config in self._model_configs.items()
            if config.get("provider") == provider
        ]


# Singleton instance
_llm_controller = None


def get_llm_controller() -> LLMController:
    """Get the singleton LLM controller instance."""
    global _llm_controller
    if _llm_controller is None:
        _llm_controller = LLMController()
    return _llm_controller

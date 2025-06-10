"""
Module for sentiment analysis of financial news.
"""

from typing import List, Dict, Optional, Union, Any
import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np

from forex_ai.core.exceptions import ModelLoadError, AnalysisError
from forex_ai.models.controller import get_model_controller

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Advanced sentiment analysis for financial news using pre-trained models."""

    def __init__(self, model_name: str = "ProsusAI/finbert", use_gpu: bool = False):
        """
        Initialize the sentiment analyzer with a financial domain-specific model.

        Args:
            model_name: Name of the pre-trained model to use (default: 'ProsusAI/finbert')
            use_gpu: Whether to use GPU acceleration if available

        Raises:
            ModelLoadError: If the model cannot be loaded
        """
        self.model_name = model_name
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.model_id = f"sentiment_{model_name.replace('/', '_')}"

        # Get model controller
        self.model_controller = get_model_controller()

        try:
            # Register model configuration
            self.model_controller._model_configs[self.model_id] = {
                "name": model_name,
                "type": "sentiment",
                "module_path": "forex_ai.analysis.sentiment",
                "class_name": "SentimentModel",
                "device": self.device,
            }

            # Load model through controller
            model = self.model_controller.get_model(self.model_id)

            # Get label mapping from the model config
            if hasattr(model.config, "id2label"):
                self.labels = model.config.id2label
            else:
                # Default FinBERT labels
                self.labels = {0: "negative", 1: "neutral", 2: "positive"}

            logger.info(
                f"Sentiment model loaded successfully with labels: {self.labels}"
            )
        except Exception as e:
            logger.error(f"Failed to load sentiment model: {str(e)}")
            raise ModelLoadError(
                f"Failed to load sentiment model {model_name}: {str(e)}"
            )

    def analyze(self, texts: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Analyze sentiment of financial text(s).

        Args:
            texts: Single text or list of texts to analyze

        Returns:
            Dict with sentiment analysis results

        Raises:
            AnalysisError: If analysis fails
        """
        try:
            # Convert single text to list
            if isinstance(texts, str):
                texts = [texts]

            # Get model through controller
            model = self.model_controller.get_model(self.model_id)

            # Process texts in batches
            results = []
            for i in range(0, len(texts), 32):  # Process 32 texts at a time
                batch = texts[i : i + 32]
                batch_results = model.predict(batch)
                results.extend(batch_results)

            return {"texts": texts, "sentiments": results, "labels": self.labels}

        except Exception as e:
            logger.error(f"Sentiment analysis failed: {str(e)}")
            raise AnalysisError(f"Failed to analyze sentiment: {str(e)}")


class SentimentModel:
    """Model class for sentiment analysis."""

    def __init__(self, name: str, device: str = "cpu"):
        """
        Initialize the sentiment model.

        Args:
            name: Model name/path
            device: Device to use (cpu/cuda)
        """
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name)
        self.model.to(device)
        self.device = device
        self.config = self.model.config

    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Make sentiment predictions.

        Args:
            texts: List of texts to analyze

        Returns:
            List of sentiment predictions
        """
        # Tokenize texts
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.device)

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=1)

        # Convert predictions to list
        predictions = predictions.cpu().numpy().tolist()

        # Format results
        results = []
        for text, probs in zip(texts, predictions):
            sentiment_idx = np.argmax(probs)
            results.append(
                {
                    "text": text,
                    "sentiment": sentiment_idx,
                    "probabilities": {
                        "negative": probs[0],
                        "neutral": probs[1],
                        "positive": probs[2],
                    },
                }
            )

        return results

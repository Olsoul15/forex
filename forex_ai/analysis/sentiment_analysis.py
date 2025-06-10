"""
Advanced Sentiment Analysis for Forex AI Trading System.

This module provides sophisticated sentiment analysis tools specifically designed
for financial texts and forex-related news. It replaces the simple rule-based
sentiment analyzer with pre-trained financial NLP models.
"""

import logging
import os
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime

import numpy as np

# Remove the top-level imports of transformers and torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch

from forex_ai.exceptions import AnalysisError, ModelError, ModelLoadingError, DataError
from forex_ai.core.exceptions import AnalysisError
from forex_ai.models.controller import get_model_controller

logger = logging.getLogger(__name__)

# Flag to indicate whether ML libraries are available
ML_LIBRARIES_AVAILABLE = False
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    ML_LIBRARIES_AVAILABLE = True
    logger.info("ML libraries (torch, transformers) are available.")
except ImportError:
    logger.warning(
        "ML libraries (torch, transformers) are not available. Using fallback implementations."
    )


class FinancialSentimentAnalyzer:
    """Advanced sentiment analysis for financial text."""

    def __init__(self, model_id: str = "financial_sentiment"):
        """
        Initialize the sentiment analyzer.

        Args:
            model_id: ID of the model to use (default: "financial_sentiment")
        """
        self.model_id = model_id
        self.model_controller = get_model_controller()

        # Register model configuration if not already registered
        if self.model_id not in self.model_controller._model_configs:
            self.model_controller._model_configs[self.model_id] = {
                "name": "ProsusAI/finbert",  # Default model
                "type": "sentiment",
                "module_path": "forex_ai.analysis.sentiment",
                "class_name": "SentimentModel",
            }

    def analyze(self, text: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Analyze sentiment of financial text.

        Args:
            text: Text or list of texts to analyze

        Returns:
            Dict with sentiment analysis results

        Raises:
            AnalysisError: If analysis fails
        """
        try:
            # Get model through controller
            model = self.model_controller.get_model(self.model_id)

            # Convert single text to list
            texts = [text] if isinstance(text, str) else text

            # Process in batches
            results = []
            for i in range(0, len(texts), 32):  # Process 32 texts at a time
                batch = texts[i : i + 32]
                batch_results = model.predict(batch)
                results.extend(batch_results)

            return {
                "texts": texts,
                "results": results,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Sentiment analysis failed: {str(e)}")
            raise AnalysisError(f"Failed to analyze sentiment: {str(e)}")

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get model performance metrics.

        Returns:
            Dict with model metrics
        """
        return self.model_controller.get_model_status(self.model_id)["metrics"]


class SentimentAnalyzer:
    """
    Financial sentiment analyzer using pre-trained models.
    """

    def __init__(self, model_name: str = "ProsusAI/finbert", use_gpu: bool = False):
        """
        Initialize the sentiment analyzer.

        Args:
            model_name: Pre-trained model to use (default: finbert)
            use_gpu: Whether to use GPU for inference
        """
        self.model_name = model_name
        self.use_gpu = (
            use_gpu and torch.cuda.is_available() if ML_LIBRARIES_AVAILABLE else False
        )
        self.device = (
            torch.device("cuda" if self.use_gpu else "cpu")
            if ML_LIBRARIES_AVAILABLE
            else None
        )
        self.tokenizer = None
        self.model = None

        if ML_LIBRARIES_AVAILABLE:
            try:
                self._load_model()
                logger.info(f"Loaded sentiment analysis model: {model_name}")
            except Exception as e:
                logger.error(f"Error loading sentiment model: {str(e)}")
                logger.warning("Using fallback sentiment analysis implementation")
        else:
            logger.info("Using fallback sentiment analysis (rule-based)")

    def _load_model(self):
        """Load the pre-trained model and tokenizer."""
        if not ML_LIBRARIES_AVAILABLE:
            return

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )

            if self.use_gpu:
                self.model = self.model.to(self.device)

            # Set model to evaluation mode
            self.model.eval()
        except Exception as e:
            raise ModelLoadingError(
                f"Failed to load sentiment model {self.model_name}: {str(e)}"
            )

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze the sentiment of a text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with sentiment analysis results
        """
        if not text or not isinstance(text, str):
            raise DataError("Invalid input text for sentiment analysis")

        # Fallback implementation if ML libraries are not available
        if not ML_LIBRARIES_AVAILABLE or self.model is None:
            return self._rule_based_sentiment(text)

        try:
            # Tokenize text
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            )

            # Move inputs to device if using GPU
            if self.use_gpu:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Perform inference
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Get scores and labels
            scores = (
                torch.nn.functional.softmax(outputs.logits, dim=1)
                .detach()
                .cpu()
                .numpy()[0]
            )

            # Get sentiment label and score
            labels = ["negative", "neutral", "positive"]  # FinBERT labels
            sentiment_label = labels[np.argmax(scores)]
            sentiment_score = float(np.max(scores))

            # Calculate sentiment value (from -1 to 1)
            sentiment_value = float(
                (scores[2] - scores[0]) / (scores[0] + scores[1] + scores[2])
            )

            return {
                "text": text,
                "sentiment": sentiment_label,
                "sentiment_score": sentiment_score,
                "sentiment_value": sentiment_value,
                "confidence": sentiment_score,
                "scores": {
                    "negative": float(scores[0]),
                    "neutral": float(scores[1]),
                    "positive": float(scores[2]),
                },
            }
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            # Fall back to rule-based if ML analysis fails
            return self._rule_based_sentiment(text)

    def _rule_based_sentiment(self, text: str) -> Dict[str, Any]:
        """Simple rule-based sentiment analysis for fallback."""
        # Calculate based on positive and negative word counts
        positive_words = [
            "bullish",
            "growth",
            "profit",
            "gain",
            "positive",
            "increase",
            "rise",
            "up",
            "higher",
            "strong",
        ]
        negative_words = [
            "bearish",
            "decline",
            "loss",
            "drop",
            "negative",
            "decrease",
            "fall",
            "down",
            "lower",
            "weak",
        ]

        text_lower = text.lower()

        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        total_count = positive_count + negative_count

        if total_count == 0:
            sentiment_label = "neutral"
            sentiment_value = 0.0
            sentiment_score = 0.5
            scores = {"negative": 0.2, "neutral": 0.6, "positive": 0.2}
        elif positive_count > negative_count:
            sentiment_label = "positive"
            sentiment_value = min(
                1.0, (positive_count - negative_count) / (total_count * 2)
            )
            sentiment_score = 0.5 + sentiment_value / 2
            scores = {"negative": 0.1, "neutral": 0.4, "positive": 0.5}
        else:
            sentiment_label = "negative"
            sentiment_value = max(
                -1.0, (negative_count - positive_count) / (total_count * 2)
            )
            sentiment_score = 0.5 - abs(sentiment_value) / 2
            scores = {"negative": 0.5, "neutral": 0.4, "positive": 0.1}

        return {
            "text": text,
            "sentiment": sentiment_label,
            "sentiment_score": sentiment_score,
            "sentiment_value": sentiment_value,
            "confidence": sentiment_score,
            "scores": scores,
            "method": "rule_based",  # Indicate this is from the fallback method
        }

    def batch_analyze(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze the sentiment of multiple texts.

        Args:
            texts: List of texts to analyze

        Returns:
            List of dictionaries with sentiment analysis results
        """
        if not texts:
            return []

        # Use individual analysis for fallback or small batches
        if not ML_LIBRARIES_AVAILABLE or self.model is None or len(texts) < 5:
            return [self.analyze(text) for text in texts]

        try:
            # Tokenize texts
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )

            # Move inputs to device if using GPU
            if self.use_gpu:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Perform inference in batches
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Get scores
            scores = (
                torch.nn.functional.softmax(outputs.logits, dim=1)
                .detach()
                .cpu()
                .numpy()
            )

            # Process results
            labels = ["negative", "neutral", "positive"]  # FinBERT labels
            results = []

            for i, text in enumerate(texts):
                sentiment_label = labels[np.argmax(scores[i])]
                sentiment_score = float(np.max(scores[i]))
                sentiment_value = float(
                    (scores[i][2] - scores[i][0])
                    / (scores[i][0] + scores[i][1] + scores[i][2])
                )

                results.append(
                    {
                        "text": text,
                        "sentiment": sentiment_label,
                        "sentiment_score": sentiment_score,
                        "sentiment_value": sentiment_value,
                        "confidence": sentiment_score,
                        "scores": {
                            "negative": float(scores[i][0]),
                            "neutral": float(scores[i][1]),
                            "positive": float(scores[i][2]),
                        },
                    }
                )

            return results
        except Exception as e:
            logger.error(f"Error in batch sentiment analysis: {str(e)}")
            # Fall back to individual processing
            return [self.analyze(text) for text in texts]


class ForexEntityExtractor:
    """
    Entity extraction for forex-related text.
    Extracts currencies, economic indicators, and relationships.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the entity extractor.

        Args:
            model_path: Path to custom model (optional)
        """
        self.model_path = model_path
        self.model = None
        self.currencies = [
            "USD",
            "EUR",
            "GBP",
            "JPY",
            "AUD",
            "CAD",
            "CHF",
            "NZD",
            "CNY",
            "HKD",
            "SGD",
            "SEK",
            "NOK",
            "MXN",
            "ZAR",
            "TRY",
            "BRL",
            "INR",
            "RUB",
            "KRW",
        ]
        self.indicators = [
            "GDP",
            "CPI",
            "PMI",
            "NFP",
            "Unemployment",
            "Interest Rates",
            "Inflation",
            "Retail Sales",
            "Trade Balance",
            "Industrial Production",
        ]

        if ML_LIBRARIES_AVAILABLE and model_path:
            try:
                self._load_model()
                logger.info(f"Loaded entity extraction model from {model_path}")
            except Exception as e:
                logger.error(f"Error loading entity extraction model: {str(e)}")
                logger.warning("Using rule-based entity extraction as fallback")
        else:
            logger.info("Using rule-based entity extraction")

    def _load_model(self):
        """Load the entity extraction model if available."""
        if not ML_LIBRARIES_AVAILABLE:
            return

        try:
            # Import here to avoid top-level dependency
            from transformers import AutoTokenizer, AutoModelForTokenClassification

            # Load the model for named entity recognition
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.model_path
            )
            self.model.eval()
        except Exception as e:
            raise ModelLoadingError(f"Failed to load entity extraction model: {str(e)}")

    def extract_currencies(self, text: str) -> List[str]:
        """
        Extract currency mentions from text.

        Args:
            text: Text to analyze

        Returns:
            List of currency codes found in the text
        """
        if not text or not isinstance(text, str):
            return []

        # If model is not available or text is too short, use rule-based approach
        if not ML_LIBRARIES_AVAILABLE or self.model is None or len(text) < 10:
            return self._rule_based_currency_extraction(text)

        try:
            # Implementation using ML model
            # ...

            # For now, fall back to rule-based since we don't have the actual ML implementation
            return self._rule_based_currency_extraction(text)
        except Exception as e:
            logger.error(f"Error in currency extraction: {str(e)}")
            return self._rule_based_currency_extraction(text)

    def _rule_based_currency_extraction(self, text: str) -> List[str]:
        """Extract currencies using simple pattern matching."""
        found_currencies = []
        text_upper = text.upper()

        # Look for currency codes
        for currency in self.currencies:
            # Check for currency code with word boundaries
            if (
                f" {currency} " in f" {text_upper} "
                or f"/{currency}" in text_upper
                or f"{currency}/" in text_upper
            ):
                found_currencies.append(currency)

        # Look for currency names
        currency_names = {
            "DOLLAR": "USD",
            "EURO": "EUR",
            "POUND": "GBP",
            "STERLING": "GBP",
            "YEN": "JPY",
            "FRANC": "CHF",
            "AUSSIE": "AUD",
            "LOONIE": "CAD",
            "KIWI": "NZD",
            "YUAN": "CNY",
            "RENMINBI": "CNY",
        }

        for name, code in currency_names.items():
            if name in text_upper and code not in found_currencies:
                found_currencies.append(code)

        return found_currencies

    def extract_economic_indicators(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract economic indicators from text.

        Args:
            text: Text to analyze

        Returns:
            List of dictionaries with economic indicator information
        """
        if not text or not isinstance(text, str):
            return []

        # Use rule-based approach regardless of ML availability for now
        return self._rule_based_indicator_extraction(text)

    def _rule_based_indicator_extraction(self, text: str) -> List[Dict[str, Any]]:
        """Extract economic indicators using pattern matching."""
        found_indicators = []
        text_upper = text.upper()

        # Define patterns for indicators with potential values
        indicator_patterns = {
            "GDP": [
                r"GDP\s+(?:growth|increase|decrease|drop|fell|rose|expanded)\s+(?:by|to|at)?\s+([\d\.]+)%?",
                r"GDP\s+(?:of|at)\s+([\d\.]+)%?",
            ],
            "CPI": [
                r"CPI\s+(?:rose|fell|increased|decreased|dropped|jumped)\s+(?:by|to|at)?\s+([\d\.]+)%?",
                r"inflation\s+(?:rate|of|at)\s+([\d\.]+)%?",
            ],
            "Unemployment": [
                r"unemployment\s+(?:rate|of|at)\s+([\d\.]+)%?",
                r"jobless\s+(?:rate|claims)\s+(?:of|at)\s+([\d\.]+)",
            ],
            "Interest Rates": [
                r"interest\s+rates?\s+(?:of|at|to)\s+([\d\.]+)%?",
                r"rates?\s+(?:hike|cut|increase|decrease)\s+(?:of|by|to)\s+([\d\.]+)",
            ],
        }

        # Look for indicator mentions
        for indicator in self.indicators:
            if indicator.upper() in text_upper:
                # Extract potential values if they exist
                indicator_data = {
                    "indicator": indicator,
                    "value": None,
                    "country": None,
                }

                # Try to identify country context
                for country in [
                    "US",
                    "USA",
                    "EU",
                    "UK",
                    "Japan",
                    "China",
                    "Germany",
                    "France",
                ]:
                    if country.upper() in text_upper:
                        indicator_data["country"] = country
                        break

                found_indicators.append(indicator_data)

        return found_indicators

    def extract_relationships(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract relationships between entities.

        Args:
            text: Text to analyze

        Returns:
            List of dictionaries with relationship information
        """
        if not text or not isinstance(text, str):
            return []

        # Currently using rule-based approach
        currencies = self.extract_currencies(text)
        indicators = self.extract_economic_indicators(text)

        relationships = []

        # Simple rules to detect cause-effect relationships
        if len(currencies) >= 1 and len(indicators) >= 1:
            text_lower = text.lower()

            # Effect phrases
            effect_phrases = [
                "impact",
                "affect",
                "influence",
                "cause",
                "lead to",
                "result in",
            ]

            for indicator in indicators:
                for currency in currencies:
                    # Check if there's a relationship mentioned between the indicator and currency
                    if any(phrase in text_lower for phrase in effect_phrases):
                        relationships.append(
                            {
                                "source": indicator["indicator"],
                                "source_type": "indicator",
                                "target": currency,
                                "target_type": "currency",
                                "relationship_type": "impacts",
                                "confidence": 0.6,  # Medium confidence for rule-based
                            }
                        )

        return relationships


class NewsImpactPredictor:
    """
    Predict the impact of news on currency pairs.
    """

    def __init__(
        self, model_path: Optional[str] = None, vectorizer_path: Optional[str] = None
    ):
        """
        Initialize the news impact predictor.

        Args:
            model_path: Path to trained model (optional)
            vectorizer_path: Path to text vectorizer (optional)
        """
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.model = None
        self.vectorizer = None
        self.sentiment_analyzer = SentimentAnalyzer()
        self.entity_extractor = ForexEntityExtractor()

        if ML_LIBRARIES_AVAILABLE and model_path and vectorizer_path:
            try:
                self._load_model()
                logger.info(f"Loaded news impact model from {model_path}")
            except Exception as e:
                logger.error(f"Error loading news impact model: {str(e)}")
                logger.warning("Using rule-based news impact prediction as fallback")
        else:
            logger.info("Using rule-based news impact prediction")

    def _load_model(self):
        """Load the impact prediction model and vectorizer if available."""
        if not ML_LIBRARIES_AVAILABLE:
            return

        try:
            # Import locally to avoid top-level dependency
            import pickle
            import joblib

            # Load the trained model and vectorizer
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)

            self.vectorizer = joblib.load(self.vectorizer_path)
            logger.info("News impact model and vectorizer loaded successfully")
        except Exception as e:
            raise ModelLoadingError(f"Failed to load news impact model: {str(e)}")

    def build_training_dataset(self, news_items, market_data):
        """
        Build a training dataset from news and market data.
        This would be used to train the model in a production system.

        Args:
            news_items: List of news items with text and metadata
            market_data: Market data for measuring impact

        Returns:
            Dataset suitable for training the model
        """
        # This is a placeholder implementation
        logger.info("Building training dataset for news impact model")

        # Basic structure for training data
        training_data = []

        for news_item in news_items:
            # Extract features from news
            features = self._extract_features(news_item)

            # Calculate impact based on market data
            impact = self._calculate_impact(news_item, market_data)

            # Add to training data
            training_data.append({"features": features, "impact": impact})

        return training_data

    def _extract_features(self, news_item):
        """
        Extract features from a news item for impact prediction.

        Args:
            news_item: News item with text and metadata

        Returns:
            Feature dictionary
        """
        text = news_item.get("text", "")

        # Get sentiment
        sentiment_result = self.sentiment_analyzer.analyze(text)

        # Get entities
        currencies = self.entity_extractor.extract_currencies(text)
        indicators = self.entity_extractor.extract_economic_indicators(text)

        # Build feature dictionary
        features = {
            "sentiment_value": sentiment_result.get("sentiment_value", 0),
            "sentiment_confidence": sentiment_result.get("confidence", 0),
            "currency_count": len(currencies),
            "indicator_count": len(indicators),
            "text_length": len(text),
            "has_major_currency": any(
                c in currencies for c in ["USD", "EUR", "GBP", "JPY"]
            ),
            "has_economic_data": len(indicators) > 0,
        }

        return features

    def _calculate_impact(self, news_item, market_data):
        """
        Calculate the actual impact of a news item on market data.

        Args:
            news_item: News item with timestamp
            market_data: Market data before and after the news

        Returns:
            Impact metrics
        """
        # Simplified implementation
        # In a real system, this would calculate price movements after news

        # Default impact
        impact = {
            "price_change_pct": 0.0,
            "volatility_change": 0.0,
            "volume_change_pct": 0.0,
            "impact_duration_minutes": 0,
        }

        # If market data is available, calculate real impact
        if market_data and "before" in market_data and "after" in market_data:
            pass  # Complex calculation would go here

        return impact

    def _determine_impact(self, impact_metrics):
        """Convert raw impact metrics to impact rating."""
        # Simple rating system based on price movement
        price_change = abs(impact_metrics.get("price_change_pct", 0))

        if price_change > 0.5:  # More than 0.5% move
            return "high"
        elif price_change > 0.2:  # 0.2% to 0.5% move
            return "medium"
        else:
            return "low"

    def predict(self, news_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict the impact of a news item on currency markets.

        Args:
            news_item: Dictionary with news text and metadata

        Returns:
            Impact prediction with confidence scores
        """
        if not news_item or "text" not in news_item:
            return {"impact": "unknown", "confidence": 0.0, "affected_pairs": []}

        text = news_item["text"]

        # Rule-based fallback method
        if not ML_LIBRARIES_AVAILABLE or self.model is None:
            return self._rule_based_prediction(text)

        try:
            # Extract features
            features = self._extract_features(news_item)

            # Vectorize features for the model
            feature_vector = self.vectorizer.transform([features])

            # Get prediction from model
            impact_prediction = self.model.predict(feature_vector)[0]
            confidence = self.model.predict_proba(feature_vector).max()

            # Extract affected currency pairs
            currencies = self.entity_extractor.extract_currencies(text)
            affected_pairs = self._determine_affected_pairs(currencies)

            return {
                "impact": impact_prediction,
                "confidence": float(confidence),
                "affected_pairs": affected_pairs,
                "reasoning": f"Based on sentiment, entities, and historical patterns",
            }
        except Exception as e:
            logger.error(f"Error in news impact prediction: {str(e)}")
            return self._rule_based_prediction(text)

    def _rule_based_prediction(self, text: str) -> Dict[str, Any]:
        """Simple rule-based impact prediction when ML model is unavailable."""
        # Get sentiment as a first signal
        sentiment_result = self.sentiment_analyzer.analyze(text)
        sentiment_value = sentiment_result.get("sentiment_value", 0)

        # Extract entities
        currencies = self.entity_extractor.extract_currencies(text)
        indicators = self.entity_extractor.extract_economic_indicators(text)

        # Determine impact based on sentiment strength and entity presence
        if abs(sentiment_value) > 0.7 and len(currencies) > 0 and len(indicators) > 0:
            impact = "high"
            confidence = 0.7
        elif abs(sentiment_value) > 0.3 and (
            len(currencies) > 0 or len(indicators) > 0
        ):
            impact = "medium"
            confidence = 0.6
        else:
            impact = "low"
            confidence = 0.5

        # Determine affected pairs
        affected_pairs = self._determine_affected_pairs(currencies)

        return {
            "impact": impact,
            "confidence": confidence,
            "affected_pairs": affected_pairs,
            "reasoning": "Based on sentiment analysis and currency/indicator mentions",
            "method": "rule_based",
        }

    def _determine_affected_pairs(self, currencies):
        """Determine which currency pairs are likely affected based on mentioned currencies."""
        if not currencies:
            return []

        # Common pairs involving the mentioned currencies
        pairs = []

        # If USD is mentioned, add major USD pairs
        if "USD" in currencies:
            for other in ["EUR", "GBP", "JPY", "CAD", "AUD", "NZD", "CHF"]:
                if other in currencies:
                    pairs.append(f"{other}/USD")
                else:
                    pairs.append(f"{other}/USD")

        # If EUR is mentioned, add EUR pairs
        if "EUR" in currencies:
            for other in ["GBP", "JPY", "CHF"]:
                if (
                    other in currencies
                    and f"{other}/EUR" not in pairs
                    and f"EUR/{other}" not in pairs
                ):
                    pairs.append(f"EUR/{other}")

        # If no specific pairs can be determined but we have currencies
        if not pairs and currencies:
            # Default to major pairs
            return ["EUR/USD", "GBP/USD", "USD/JPY"]

        return pairs


class RAGNewsProcessor:
    """
    Process news using Retrieval-Augmented Generation.
    """

    def __init__(
        self,
        vector_store_path: Optional[str] = None,
        impact_model_path: Optional[str] = None,
    ):
        """
        Initialize the RAG news processor.

        Args:
            vector_store_path: Path to vector store for retrievals (optional)
            impact_model_path: Path to impact prediction model (optional)
        """
        self.vector_store_path = vector_store_path
        self.impact_model_path = impact_model_path
        self.vector_store = None
        self.impact_model = None
        self.sentiment_analyzer = SentimentAnalyzer()
        self.entity_extractor = ForexEntityExtractor()
        self.news_impact_predictor = NewsImpactPredictor(model_path=impact_model_path)

        if ML_LIBRARIES_AVAILABLE and vector_store_path:
            try:
                self._load_vector_store()
                logger.info(f"Loaded vector store from {vector_store_path}")
            except Exception as e:
                logger.error(f"Error loading vector store: {str(e)}")
                logger.warning("RAG functionality will be limited without vector store")

    def _load_vector_store(self):
        """Load the vector store if available."""
        if not ML_LIBRARIES_AVAILABLE:
            return

        try:
            # Conditional imports
            import faiss
            import pickle

            # Load the vector store
            with open(self.vector_store_path, "rb") as f:
                self.vector_store = pickle.load(f)

            logger.info("Vector store loaded successfully")
        except Exception as e:
            raise ModelLoadingError(f"Failed to load vector store: {str(e)}")

    def setup_vector_store(self, historical_news):
        """
        Set up a vector store from historical news data.

        Args:
            historical_news: List of historical news items

        Returns:
            Path to saved vector store
        """
        if not ML_LIBRARIES_AVAILABLE:
            logger.warning("Cannot set up vector store without ML libraries")
            return None

        try:
            # Conditional imports
            import faiss
            import pickle
            import os
            from transformers import AutoTokenizer, AutoModel
            import torch
            import numpy as np

            # Create vector store (simplified)
            logger.info("Creating vector store from historical news")

            # This is a placeholder - in a real implementation, we would:
            # 1. Extract text from each news item
            # 2. Create embeddings using a sentence transformer
            # 3. Create a FAISS index from the embeddings
            # 4. Save the index and metadata

            # Placeholder for vector store path
            output_path = "data/vector_store.pkl"

            # Return the path
            return output_path

        except Exception as e:
            logger.error(f"Error setting up vector store: {str(e)}")
            return None

    def process_news(self, news_item):
        """
        Process a news item with RAG to provide enhanced insights.

        Args:
            news_item: Dictionary with news text and metadata

        Returns:
            Enhanced news item with insights
        """
        if not news_item or "text" not in news_item:
            return {"error": "Invalid news item format"}

        text = news_item["text"]
        result = {}

        # Base analysis regardless of ML availability
        try:
            # Get sentiment
            sentiment_result = self.sentiment_analyzer.analyze(text)
            result["sentiment"] = sentiment_result

            # Extract entities
            currencies = self.entity_extractor.extract_currencies(text)
            indicators = self.entity_extractor.extract_economic_indicators(text)
            relationships = self.entity_extractor.extract_relationships(text)

            result["entities"] = {
                "currencies": currencies,
                "indicators": indicators,
                "relationships": relationships,
            }

            # Get impact prediction
            impact = self.news_impact_predictor.predict(news_item)
            result["impact"] = impact

            # Add original news
            result["original_news"] = news_item

        except Exception as e:
            logger.error(f"Error in base news processing: {str(e)}")
            result["error"] = str(e)

        # Enhanced processing with RAG if available
        if ML_LIBRARIES_AVAILABLE and self.vector_store:
            try:
                # This would use the vector store to find similar historical news
                # and provide additional context

                # For now, just add a placeholder
                result["similar_historical_news"] = []
                result["enhanced_insights"] = {
                    "historical_context": "No historical context available",
                    "potential_consequences": "Unable to determine without RAG",
                    "reliability": 0.5,
                }
            except Exception as e:
                logger.error(f"Error in enhanced RAG processing: {str(e)}")

        return result

    def batch_process(self, news_items):
        """
        Process multiple news items in batch.

        Args:
            news_items: List of news item dictionaries

        Returns:
            List of processed news items with insights
        """
        if not news_items:
            return []

        # Process each news item individually
        results = [self.process_news(item) for item in news_items]

        # If we have ML capabilities, we could add cross-news analysis
        if ML_LIBRARIES_AVAILABLE and len(results) > 1:
            try:
                # Identify trends across multiple news items
                # This would be more sophisticated in a real implementation

                # Find common currencies
                all_currencies = set()
                for result in results:
                    if "entities" in result and "currencies" in result["entities"]:
                        all_currencies.update(result["entities"]["currencies"])

                # Find common indicators
                all_indicators = set()
                for result in results:
                    if "entities" in result and "indicators" in result["entities"]:
                        for indicator in result["entities"]["indicators"]:
                            all_indicators.add(indicator.get("indicator"))

                # Add cross-news insights to each result
                for result in results:
                    result["cross_news_insights"] = {
                        "common_currencies": list(all_currencies),
                        "common_indicators": list(all_indicators),
                        "trend_strength": 0.0,  # Placeholder
                    }
            except Exception as e:
                logger.error(f"Error in batch cross-news analysis: {str(e)}")

        return results


class SocialMediaSentimentAnalyzer:
    """
    Specialized sentiment analyzer for social media content related to forex trading.
    Handles Twitter, Reddit, StockTwits and other social platforms with specialized
    processing for the unique language patterns found in these sources.
    """

    def __init__(
        self,
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment",
        use_gpu: bool = False,
        sources: List[str] = None,
    ):
        """
        Initialize the social media sentiment analyzer.

        Args:
            model_name: Pre-trained model to use (default: twitter-roberta-base-sentiment)
            use_gpu: Whether to use GPU for inference
            sources: List of social media sources to analyze (e.g., ['twitter', 'reddit', 'stocktwits'])
        """
        self.model_name = model_name
        self.use_gpu = (
            use_gpu and torch.cuda.is_available() if ML_LIBRARIES_AVAILABLE else False
        )
        self.device = (
            torch.device("cuda" if self.use_gpu else "cpu")
            if ML_LIBRARIES_AVAILABLE
            else None
        )
        self.tokenizer = None
        self.model = None
        self.sources = sources or ["twitter", "reddit", "stocktwits"]

        # Special tokens and patterns for social media
        self.special_patterns = {
            "twitter": {
                "cashtags": re.compile(r"\$[A-Za-z]+"),
                "hashtags": re.compile(r"#[A-Za-z0-9_]+"),
                "mentions": re.compile(r"@[A-Za-z0-9_]+"),
            },
            "reddit": {
                "subreddits": re.compile(r"r/[A-Za-z0-9_]+"),
                "users": re.compile(r"u/[A-Za-z0-9_]+"),
            },
            "stocktwits": {
                "cashtags": re.compile(r"\$[A-Za-z.]+"),
                "bullish": re.compile(
                    r"\b(bullish|long|calls|to the moon|🚀|💎)\b", re.IGNORECASE
                ),
                "bearish": re.compile(
                    r"\b(bearish|short|puts|drilling|💩|🐻)\b", re.IGNORECASE
                ),
            },
        }

        # Emoji sentiment mapping (simplified)
        self.emoji_sentiment = {
            "🚀": 1.0,  # very positive
            "💎": 0.8,  # positive
            "📈": 0.7,  # positive
            "🐂": 0.6,  # positive (bull)
            "👍": 0.5,  # positive
            "😊": 0.4,  # positive
            "🤔": 0.0,  # neutral
            "😐": 0.0,  # neutral
            "👎": -0.5,  # negative
            "😡": -0.6,  # negative
            "📉": -0.7,  # negative
            "🐻": -0.8,  # negative (bear)
            "💩": -1.0,  # very negative
        }

        # Load the model if ML libraries are available
        if ML_LIBRARIES_AVAILABLE:
            self._load_model()
        else:
            logger.warning(
                "ML libraries not available. Social media sentiment analyzer will use rule-based methods."
            )

    def _load_model(self):
        """Load the pre-trained model and tokenizer."""
        try:
            logger.info(f"Loading social media sentiment model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )
            self.model.to(self.device)
            logger.info("Social media sentiment model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load social media sentiment model: {e}")
            raise ModelLoadingError(f"Failed to load social media sentiment model: {e}")

    def preprocess_social_text(self, text: str, source: str = None) -> str:
        """
        Preprocess social media text to improve sentiment analysis accuracy.

        Args:
            text: The raw social media text
            source: The source platform (twitter, reddit, stocktwits)

        Returns:
            Preprocessed text
        """
        if not text:
            return ""

        # Apply source-specific preprocessing if specified
        if source and source.lower() in self.special_patterns:
            patterns = self.special_patterns[source.lower()]

            # Extract special tokens for later analysis
            for pattern_name, pattern in patterns.items():
                # Save matches for later but remove/normalize for the NLP model
                if pattern_name in ["cashtags", "hashtags"]:
                    text = pattern.sub(" ", text)
                elif pattern_name in ["mentions", "users", "subreddits"]:
                    text = pattern.sub(" ", text)

        # General social media preprocessing
        # Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        # Convert HTML entities
        text = re.sub(r"&amp;", "&", text)
        # Handle repeated characters (e.g., "sooooo good" -> "so good")
        text = re.sub(r"(.)\1{2,}", r"\1\1", text)
        # Handle all caps (often indicates shouting/emphasis)
        if text.isupper():
            text = text.lower() + " [EMPHASIS]"

        return text

    def extract_symbols(self, text: str, source: str = None) -> List[str]:
        """
        Extract financial symbols (cashtags) from social media text.

        Args:
            text: The social media text
            source: The source platform

        Returns:
            List of extracted symbols
        """
        symbols = []

        # Use source-specific pattern if available
        if source and source.lower() in self.special_patterns:
            if "cashtags" in self.special_patterns[source.lower()]:
                pattern = self.special_patterns[source.lower()]["cashtags"]
                symbols = [m.group(0)[1:] for m in pattern.finditer(text)]
        else:
            # Generic cashtag pattern as fallback
            symbols = [m.group(0)[1:] for m in re.finditer(r"\$([A-Za-z.]+)", text)]

        # Handle forex pairs specifically - look for major currencies
        currency_patterns = [
            (r"\bEUR/USD\b", "EURUSD"),
            (r"\bGBP/USD\b", "GBPUSD"),
            (r"\bUSD/JPY\b", "USDJPY"),
            (r"\bAUD/USD\b", "AUDUSD"),
            (r"\bUSD/CAD\b", "USDCAD"),
            (r"\bUSD/CHF\b", "USDCHF"),
            (r"\bNZD/USD\b", "NZDUSD"),
        ]

        for pattern, symbol in currency_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                symbols.append(symbol)

        return list(set(symbols))  # Remove duplicates

    def analyze(
        self, text: str, source: str = None, include_symbols: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze sentiment of social media text.

        Args:
            text: The text to analyze
            source: The source platform (twitter, reddit, stocktwits)
            include_symbols: Whether to extract and include financial symbols

        Returns:
            Dict containing sentiment analysis results
        """
        if not text or not text.strip():
            return {
                "sentiment": "neutral",
                "score": 0.0,
                "confidence": 1.0,
                "source": source,
                "symbols": [] if include_symbols else None,
            }

        # Extract symbols if requested
        symbols = self.extract_symbols(text, source) if include_symbols else None

        # Preprocess text for sentiment analysis
        processed_text = self.preprocess_social_text(text, source)

        # Use ML model if available, otherwise fall back to rule-based
        if ML_LIBRARIES_AVAILABLE and self.model and self.tokenizer:
            try:
                # Tokenize and prepare for model
                inputs = self.tokenizer(
                    processed_text, return_tensors="pt", truncation=True, max_length=128
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Get model predictions
                with torch.no_grad():
                    outputs = self.model(**inputs)

                # Process outputs
                scores = torch.nn.functional.softmax(outputs.logits, dim=1)
                scores = scores.detach().cpu().numpy()[0]

                # Map scores to sentiment categories
                if len(scores) == 3:  # Negative, Neutral, Positive
                    sentiment = ["negative", "neutral", "positive"][np.argmax(scores)]
                    sentiment_score = float(
                        scores[2] - scores[0]
                    )  # Positive - Negative
                    confidence = float(scores[np.argmax(scores)])
                else:
                    # Other models might have different output formats
                    sentiment = "positive" if np.argmax(scores) == 1 else "negative"
                    sentiment_score = (
                        float(scores[1] * 2 - 1) if len(scores) == 2 else 0.0
                    )
                    confidence = float(scores[np.argmax(scores)])
            except Exception as e:
                logger.warning(
                    f"ML model sentiment analysis failed, falling back to rule-based: {e}"
                )
                return self._rule_based_social_sentiment(text, source, symbols)
        else:
            # Use rule-based method
            return self._rule_based_social_sentiment(text, source, symbols)

        return {
            "sentiment": sentiment,
            "score": sentiment_score,  # Range from -1 (negative) to 1 (positive)
            "confidence": confidence,
            "source": source,
            "symbols": symbols if include_symbols else None,
            "raw_text": text,
        }

    def _rule_based_social_sentiment(
        self, text: str, source: str = None, symbols: List[str] = None
    ) -> Dict[str, Any]:
        """
        Rule-based sentiment analysis for social media text when ML model is unavailable.

        Args:
            text: The text to analyze
            source: The source platform
            symbols: Pre-extracted symbols (if any)

        Returns:
            Dict containing sentiment analysis results
        """
        if not text:
            return {
                "sentiment": "neutral",
                "score": 0.0,
                "confidence": 0.5,
                "source": source,
                "symbols": symbols or [],
            }

        text = text.lower()

        # Extract symbols if not provided
        if symbols is None:
            symbols = self.extract_symbols(text, source)

        # Basic sentiment dictionaries
        positive_words = [
            "bullish",
            "long",
            "buy",
            "calls",
            "up",
            "upside",
            "higher",
            "support",
            "strong",
            "strength",
            "breakout",
            "rally",
            "gain",
            "profit",
            "win",
            "good",
            "great",
            "excellent",
            "moon",
            "rocket",
        ]

        negative_words = [
            "bearish",
            "short",
            "sell",
            "puts",
            "down",
            "downside",
            "lower",
            "resistance",
            "weak",
            "weakness",
            "breakdown",
            "dump",
            "crash",
            "loss",
            "lose",
            "bad",
            "terrible",
            "poor",
            "fail",
            "drilling",
        ]

        # Count occurrences
        pos_count = sum(
            1 for word in positive_words if re.search(r"\b" + word + r"\b", text)
        )
        neg_count = sum(
            1 for word in negative_words if re.search(r"\b" + word + r"\b", text)
        )

        # Check for source-specific sentiment patterns
        if source and source.lower() in self.special_patterns:
            patterns = self.special_patterns[source.lower()]
            if "bullish" in patterns:
                pos_count += len(re.findall(patterns["bullish"], text))
            if "bearish" in patterns:
                neg_count += len(re.findall(patterns["bearish"], text))

        # Check for emojis
        for emoji, value in self.emoji_sentiment.items():
            if emoji in text:
                if value > 0:
                    pos_count += (
                        text.count(emoji) * value * 2
                    )  # Give emojis more weight
                elif value < 0:
                    neg_count += text.count(emoji) * abs(value) * 2

        # Determine sentiment
        if pos_count > neg_count:
            sentiment = "positive"
            score = min(pos_count / (pos_count + neg_count + 1) * 2 - 1, 1.0)
            confidence = min(
                0.5 + (pos_count - neg_count) / 10, 0.9
            )  # Cap at 0.9 for rule-based
        elif neg_count > pos_count:
            sentiment = "negative"
            score = max(neg_count / (pos_count + neg_count + 1) * -2 + 1, -1.0)
            confidence = min(0.5 + (neg_count - pos_count) / 10, 0.9)
        else:
            sentiment = "neutral"
            score = 0.0
            confidence = 0.5

        return {
            "sentiment": sentiment,
            "score": score,  # Range from -1 (negative) to 1 (positive)
            "confidence": confidence,
            "source": source,
            "symbols": symbols,
            "raw_text": text,
            "method": "rule-based",
        }

    def batch_analyze(
        self, texts: List[str], sources: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple social media texts efficiently in a batch.

        Args:
            texts: List of texts to analyze
            sources: List of sources corresponding to each text

        Returns:
            List of sentiment analysis results for each text
        """
        if not texts:
            return []

        # Handle sources
        if sources is None:
            sources = [None] * len(texts)
        elif len(sources) != len(texts):
            sources = sources[: len(texts)] + [None] * (len(texts) - len(sources))

        results = []

        # Use ML model if available
        if ML_LIBRARIES_AVAILABLE and self.model and self.tokenizer:
            try:
                # Process in batches of 16 for memory efficiency
                batch_size = 16
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i : i + batch_size]
                    batch_sources = sources[i : i + batch_size]

                    # Preprocess texts
                    processed_texts = [
                        self.preprocess_social_text(text, source)
                        for text, source in zip(batch_texts, batch_sources)
                    ]

                    # Extract symbols
                    batch_symbols = [
                        self.extract_symbols(text, source)
                        for text, source in zip(batch_texts, batch_sources)
                    ]

                    # Tokenize
                    inputs = self.tokenizer(
                        processed_texts,
                        return_tensors="pt",
                        truncation=True,
                        max_length=128,
                        padding=True,
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    # Get predictions
                    with torch.no_grad():
                        outputs = self.model(**inputs)

                    # Process results
                    scores = torch.nn.functional.softmax(outputs.logits, dim=1)
                    scores = scores.detach().cpu().numpy()

                    for j, (score_array, text, source, symbols) in enumerate(
                        zip(scores, batch_texts, batch_sources, batch_symbols)
                    ):
                        if len(score_array) == 3:  # Negative, Neutral, Positive
                            sentiment = ["negative", "neutral", "positive"][
                                np.argmax(score_array)
                            ]
                            sentiment_score = float(score_array[2] - score_array[0])
                            confidence = float(score_array[np.argmax(score_array)])
                        else:
                            sentiment = (
                                "positive"
                                if np.argmax(score_array) == 1
                                else "negative"
                            )
                            sentiment_score = (
                                float(score_array[1] * 2 - 1)
                                if len(score_array) == 2
                                else 0.0
                            )
                            confidence = float(score_array[np.argmax(score_array)])

                        results.append(
                            {
                                "sentiment": sentiment,
                                "score": sentiment_score,
                                "confidence": confidence,
                                "source": source,
                                "symbols": symbols,
                                "raw_text": text,
                            }
                        )
            except Exception as e:
                logger.warning(
                    f"Batch ML sentiment analysis failed, falling back to rule-based: {e}"
                )
                return [
                    self._rule_based_social_sentiment(text, source)
                    for text, source in zip(texts, sources)
                ]
        else:
            # Use rule-based method for all texts
            return [
                self._rule_based_social_sentiment(text, source)
                for text, source in zip(texts, sources)
            ]

        return results

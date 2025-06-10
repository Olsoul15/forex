"""
Deep Learning Models for Technical Analysis in the AI Forex Trading System.

This module provides neural network models for price prediction and pattern
recognition in forex market data.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Deep learning imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model, Model
    from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional, GRU
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from forex_ai.utils.logging import get_logger

logger = get_logger(__name__)


class TimeSeriesGenerator:
    """Generate batches of time series data for model training."""

    def __init__(
        self,
        lookback: int = 20,
        forecast: int = 1,
        batch_size: int = 32,
        scale_features: bool = True,
    ):
        """
        Initialize the time series generator.

        Args:
            lookback: Number of previous time steps to use as input features
            forecast: Number of future time steps to predict
            batch_size: Number of samples per batch
            scale_features: Whether to scale features
        """
        self.lookback = lookback
        self.forecast = forecast
        self.batch_size = batch_size
        self.scale_features = scale_features
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))

    def prepare_data(
        self,
        data: pd.DataFrame,
        target_col: str = "close",
        feature_cols: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for time series forecasting.

        Args:
            data: DataFrame with time series data
            target_col: Column to predict
            feature_cols: List of columns to use as features

        Returns:
            Tuple of (X, y) arrays for model training
        """
        if len(data) <= self.lookback:
            raise ValueError(
                f"Data length ({len(data)}) must be greater than lookback ({self.lookback})"
            )

        # Default to all numeric columns if feature_cols not provided
        if feature_cols is None:
            feature_cols = [
                col
                for col in data.columns
                if pd.api.types.is_numeric_dtype(data[col]) and col != target_col
            ]

        # Prepare feature matrix
        features = data[feature_cols].values
        target = data[target_col].values.reshape(-1, 1)

        # Scale data if requested
        if self.scale_features:
            features = self.feature_scaler.fit_transform(features)
            target = self.target_scaler.fit_transform(target)

        X, y = [], []

        # Create sequences
        for i in range(len(data) - self.lookback - self.forecast + 1):
            # Input sequence (lookback period)
            X.append(features[i : i + self.lookback])

            # Target value (forecast period)
            if self.forecast == 1:
                y.append(target[i + self.lookback])
            else:
                y.append(target[i + self.lookback : i + self.lookback + self.forecast])

        return np.array(X), np.array(y)

    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled target values.

        Args:
            y: Scaled target values

        Returns:
            Original scale target values
        """
        if self.scale_features:
            # Reshape if needed (1D array)
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)
            return self.target_scaler.inverse_transform(y)
        return y


class DeepLearningModel:
    """Deep learning model for time series forecasting."""

    def __init__(
        self,
        model_type: str = "lstm",
        lookback: int = 20,
        forecast: int = 1,
        n_features: int = 10,
    ):
        """
        Initialize the deep learning model.

        Args:
            model_type: Type of model ('lstm', 'gru', 'bidirectional_lstm')
            lookback: Number of previous time steps to use as input
            forecast: Number of future time steps to predict
            n_features: Number of input features
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError(
                "TensorFlow is not available. Please install tensorflow to use deep learning models."
            )

        self.model_type = model_type
        self.lookback = lookback
        self.forecast = forecast
        self.n_features = n_features
        self.model = None
        self.data_generator = TimeSeriesGenerator(lookback=lookback, forecast=forecast)

    def build_model(self) -> None:
        """Build the neural network model architecture."""
        if self.model_type == "lstm":
            self.model = Sequential(
                [
                    LSTM(
                        units=100,
                        return_sequences=True,
                        input_shape=(self.lookback, self.n_features),
                    ),
                    Dropout(0.2),
                    LSTM(units=50, return_sequences=False),
                    Dropout(0.2),
                    Dense(units=self.forecast),
                ]
            )
        elif self.model_type == "gru":
            self.model = Sequential(
                [
                    GRU(
                        units=100,
                        return_sequences=True,
                        input_shape=(self.lookback, self.n_features),
                    ),
                    Dropout(0.2),
                    GRU(units=50, return_sequences=False),
                    Dropout(0.2),
                    Dense(units=self.forecast),
                ]
            )
        elif self.model_type == "bidirectional_lstm":
            self.model = Sequential(
                [
                    Bidirectional(
                        LSTM(units=100, return_sequences=True),
                        input_shape=(self.lookback, self.n_features),
                    ),
                    Dropout(0.2),
                    Bidirectional(LSTM(units=50, return_sequences=False)),
                    Dropout(0.2),
                    Dense(units=self.forecast),
                ]
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
        logger.info(
            f"Built {self.model_type} model with {self.model.count_params()} parameters"
        )

    def train(
        self,
        data: pd.DataFrame,
        target_col: str = "close",
        feature_cols: Optional[List[str]] = None,
        epochs: int = 100,
        validation_split: float = 0.2,
        patience: int = 10,
        model_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train the deep learning model.

        Args:
            data: DataFrame with time series data
            target_col: Column to predict
            feature_cols: List of columns to use as features
            epochs: Number of training epochs
            validation_split: Fraction of data to use for validation
            patience: Patience for early stopping
            model_path: Path to save the model

        Returns:
            Dictionary with training results and metrics
        """
        if feature_cols is None:
            feature_cols = [
                col
                for col in data.columns
                if pd.api.types.is_numeric_dtype(data[col]) and col != target_col
            ]

        self.n_features = len(feature_cols)

        # Prepare data
        X, y = self.data_generator.prepare_data(data, target_col, feature_cols)

        # Build model if not already built
        if self.model is None:
            self.build_model()

        # Training-validation split
        train_size = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor="val_loss", patience=patience, restore_best_weights=True
            ),
        ]

        if model_path:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            callbacks.append(
                ModelCheckpoint(
                    filepath=model_path,
                    monitor="val_loss",
                    save_best_only=True,
                    verbose=1,
                )
            )

        # Train model
        history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1,
        )

        # Evaluate model
        train_loss = history.history["loss"][-1]
        val_loss = history.history["val_loss"][-1]

        # Make predictions
        y_pred_val = self.model.predict(X_val)

        # Convert back to original scale
        y_val_orig = self.data_generator.inverse_transform_target(y_val)
        y_pred_val_orig = self.data_generator.inverse_transform_target(y_pred_val)

        # Calculate metrics
        mse = mean_squared_error(y_val_orig, y_pred_val_orig)
        mae = mean_absolute_error(y_val_orig, y_pred_val_orig)
        r2 = r2_score(y_val_orig, y_pred_val_orig)

        logger.info(
            f"Model training completed - MSE: {mse:.5f}, MAE: {mae:.5f}, R²: {r2:.5f}"
        )

        return {
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "mse": float(mse),
            "mae": float(mae),
            "r2": float(r2),
            "epochs_trained": len(history.history["loss"]),
            "model_path": model_path if model_path else None,
            "features": feature_cols,
            "target": target_col,
            "lookback": self.lookback,
            "forecast": self.forecast,
        }

    def predict(
        self,
        data: pd.DataFrame,
        target_col: str = "close",
        feature_cols: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Make predictions with the trained model.

        Args:
            data: DataFrame with input features
            target_col: Target column for prediction
            feature_cols: Feature columns to use

        Returns:
            Numpy array with predictions in original scale
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded")

        # Default to all numeric columns if feature_cols not provided
        if feature_cols is None:
            feature_cols = [
                col
                for col in data.columns
                if pd.api.types.is_numeric_dtype(data[col]) and col != target_col
            ]

        # Prepare input data (last lookback period)
        X, _ = self.data_generator.prepare_data(data, target_col, feature_cols)

        # Make prediction
        predictions = self.model.predict(X)

        # Convert back to original scale
        return self.data_generator.inverse_transform_target(predictions)

    def save(self, model_path: str, metadata_path: Optional[str] = None) -> None:
        """
        Save the trained model and metadata.

        Args:
            model_path: Path to save the Keras model
            metadata_path: Path to save additional metadata
        """
        if self.model is None:
            raise ValueError("No model to save")

        # Save Keras model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)

        # Save metadata if path provided
        if metadata_path:
            metadata = {
                "model_type": self.model_type,
                "lookback": self.lookback,
                "forecast": self.forecast,
                "n_features": self.n_features,
                "timestamp": datetime.now().isoformat(),
            }

            os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
            with open(metadata_path, "w") as f:
                json.dump(metadata, f)

        logger.info(f"Model saved to {model_path}")

    def load(self, model_path: str, metadata_path: Optional[str] = None) -> None:
        """
        Load a trained model and metadata.

        Args:
            model_path: Path to the saved Keras model
            metadata_path: Path to the metadata file
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load metadata if available
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            self.model_type = metadata.get("model_type", self.model_type)
            self.lookback = metadata.get("lookback", self.lookback)
            self.forecast = metadata.get("forecast", self.forecast)
            self.n_features = metadata.get("n_features", self.n_features)

        # Load Keras model
        self.model = load_model(model_path)

        logger.info(f"Loaded model from {model_path}")

    def plot_prediction(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        title: str = "Model Prediction",
        figsize: Tuple[int, int] = (12, 6),
    ) -> None:
        """
        Plot actual vs predicted values.

        Args:
            actual: Array of actual values
            predicted: Array of predicted values
            title: Plot title
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        plt.plot(actual, label="Actual", color="blue")
        plt.plot(predicted, label="Predicted", color="red", linestyle="--")
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def create_price_prediction_model(
    data: pd.DataFrame,
    lookback: int = 20,
    forecast: int = 1,
    model_type: str = "lstm",
    target_col: str = "close",
    feature_cols: Optional[List[str]] = None,
    save_dir: str = "models/deep_learning",
) -> Dict[str, Any]:
    """
    Create and train a deep learning model for price prediction.

    Args:
        data: DataFrame with OHLCV and indicator data
        lookback: Number of past periods to use for prediction
        forecast: Number of periods to forecast
        model_type: Type of model architecture
        target_col: Column to predict
        feature_cols: Features to use for prediction
        save_dir: Directory to save model

    Returns:
        Dictionary with model information and training results
    """
    if not TENSORFLOW_AVAILABLE:
        logger.warning("TensorFlow not available. Cannot create deep learning model.")
        return {"error": "TensorFlow not available"}

    # Create model subdirectory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_subdir = f"{model_type}_{timestamp}"
    model_dir = os.path.join(save_dir, model_subdir)
    os.makedirs(model_dir, exist_ok=True)

    # Default feature columns if none provided
    if feature_cols is None:
        feature_cols = [
            col
            for col in data.columns
            if pd.api.types.is_numeric_dtype(data[col])
            and col != target_col
            and not pd.isna(data[col]).any()
        ]  # Exclude columns with NaN values

    # Create model
    model = DeepLearningModel(
        model_type=model_type,
        lookback=lookback,
        forecast=forecast,
        n_features=len(feature_cols),
    )

    # Clean data
    clean_data = data.dropna()
    if len(clean_data) < lookback + forecast + 50:  # Need sufficient data
        return {
            "error": f"Insufficient data after removing NaN values: {len(clean_data)} rows"
        }

    # Model paths
    model_path = os.path.join(model_dir, "model.keras")
    metadata_path = os.path.join(model_dir, "metadata.json")

    # Train model
    train_results = model.train(
        data=clean_data,
        target_col=target_col,
        feature_cols=feature_cols,
        epochs=200,
        validation_split=0.2,
        patience=20,
        model_path=model_path,
    )

    # Save model and metadata
    model.save(model_path, metadata_path)

    # Add model info to results
    results = {
        "model_dir": model_dir,
        "model_path": model_path,
        "metadata_path": metadata_path,
        "model_type": model_type,
        "lookback": lookback,
        "forecast": forecast,
        "features": feature_cols,
        "target": target_col,
        "data_points": len(clean_data),
        "training_results": train_results,
    }

    return results


def load_price_prediction_model(model_dir: str) -> DeepLearningModel:
    """
    Load a saved price prediction model.

    Args:
        model_dir: Directory containing the model

    Returns:
        Loaded DeepLearningModel object
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow not available. Cannot load deep learning model.")

    model_path = os.path.join(model_dir, "model.keras")
    metadata_path = os.path.join(model_dir, "metadata.json")

    # Check if files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load metadata
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        model_type = metadata.get("model_type", "lstm")
        lookback = metadata.get("lookback", 20)
        forecast = metadata.get("forecast", 1)
        n_features = metadata.get("n_features", 10)
    else:
        # Default values if metadata not available
        model_type = "lstm"
        lookback = 20
        forecast = 1
        n_features = 10

    # Create model
    model = DeepLearningModel(
        model_type=model_type,
        lookback=lookback,
        forecast=forecast,
        n_features=n_features,
    )

    # Load model weights
    model.load(model_path, metadata_path if os.path.exists(metadata_path) else None)

    return model


def predict_future_prices(
    model: DeepLearningModel,
    data: pd.DataFrame,
    periods: int = 5,
    target_col: str = "close",
    feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Make future price predictions using the trained model.

    Args:
        model: Trained DeepLearningModel
        data: DataFrame with historical data
        periods: Number of periods to predict
        target_col: Column to predict
        feature_cols: Features to use for prediction

    Returns:
        DataFrame with future predictions
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow not available. Cannot make predictions.")

    # Default feature columns if none provided
    if feature_cols is None:
        feature_cols = [
            col
            for col in data.columns
            if pd.api.types.is_numeric_dtype(data[col]) and col != target_col
        ]

    # Clean data
    clean_data = data.dropna().copy()
    if len(clean_data) < model.lookback:
        raise ValueError(
            f"Insufficient data for prediction: {len(clean_data)} rows, need at least {model.lookback}"
        )

    # Start with most recent data
    working_data = clean_data.iloc[-model.lookback :].copy()

    predictions = []
    prediction_dates = []
    last_date = working_data.index[-1]
    freq = pd.infer_freq(working_data.index)

    # Generate dates for predictions
    for i in range(1, periods + 1):
        if freq:
            next_date = pd.date_range(start=last_date, periods=i + 1, freq=freq)[-1]
        else:
            # If frequency can't be inferred, estimate from last two points
            if hasattr(working_data.index, "to_series"):
                date_diff = working_data.index.to_series().diff().median()
                next_date = last_date + date_diff * i
            else:
                next_date = i  # Just use integer index

        prediction_dates.append(next_date)

    # Make predictions one step at a time
    for i in range(periods):
        # Predict next value
        pred_value = model.predict(working_data, target_col, feature_cols)[-1][0]
        predictions.append(pred_value)

        # Create new row with prediction
        new_row = working_data.iloc[-1:].copy()
        new_row.index = [prediction_dates[i]]
        new_row[target_col] = pred_value

        # Append to working data and drop oldest row
        working_data = pd.concat([working_data, new_row])
        working_data = working_data.iloc[1:]

    # Create prediction DataFrame
    results = pd.DataFrame(
        {target_col: predictions, "prediction_date": prediction_dates}
    )

    results.set_index("prediction_date", inplace=True)

    return results

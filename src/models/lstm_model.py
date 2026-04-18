"""
LSTM Deep Learning Model
========================
TensorFlow/Keras LSTM for sequential demand forecasting.
Handles sequence preparation, training pipeline, and early stopping.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.models.base_model import BaseForecaster
from src.utils.config import cfg
from src.utils.logger import get_logger, log_execution_time

log = get_logger(__name__)


def _make_sequences(
    data: np.ndarray,
    seq_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a 1-D time series into (X, y) sequence pairs.

    Args:
        data: Scaled demand array, shape (T,).
        seq_length: Number of past timesteps in each input window.

    Returns:
        X shape (N, seq_length, 1), y shape (N,)
    """
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length:i])
        y.append(data[i])
    X = np.array(X, dtype=np.float32)[..., np.newaxis]  # (N, T, 1)
    y = np.array(y, dtype=np.float32)
    return X, y


class LSTMForecaster(BaseForecaster):
    """
    Multi-layer LSTM for sequence-to-one demand forecasting.

    Architecture:
        LSTM(128) → Dropout → LSTM(64) → Dropout → Dense(32) → Dense(1)
    """

    name = "lstm"

    def __init__(self) -> None:
        params = cfg.models.deep_learning.lstm.to_dict()
        super().__init__(params)
        self._tf_model = None
        self.history_ = None
        self.seq_length: int = params.get("sequence_length", 168)
        self._scaler_mean: float = 0.0
        self._scaler_std: float = 1.0
        self._available = self._check_tf()

    # ── Public ────────────────────────────────────────────────────────

    @log_execution_time
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        val_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
    ) -> "LSTMForecaster":
        """
        Train LSTM on demand time series.

        Note: LSTM uses the raw demand series (from y), not tabular X.
        X is accepted for API compatibility but the sequence is derived from y.
        """
        if not self._available:
            raise ImportError("TensorFlow not installed.")

        import tensorflow as tf

        p = self.params
        demand = y.values.astype(np.float32)

        # Standardise
        self._scaler_mean = demand.mean()
        self._scaler_std = demand.std() + 1e-8
        scaled = (demand - self._scaler_mean) / self._scaler_std

        X_seq, y_seq = _make_sequences(scaled, self.seq_length)
        log.info(
            "LSTM sequences — X: {}, y: {}",
            X_seq.shape, y_seq.shape,
        )

        # Build model
        self._tf_model = self._build_model(X_seq.shape[1:])
        self._tf_model.summary(print_fn=lambda s: log.debug(s))

        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=p.get("patience", 10),
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1,
            ),
        ]

        log.info(
            "Training LSTM — epochs={}, batch_size={}, seq_length={}",
            p.get("epochs"), p.get("batch_size"), self.seq_length,
        )

        self.history_ = self._tf_model.fit(
            X_seq,
            y_seq,
            epochs=p.get("epochs", 50),
            batch_size=p.get("batch_size", 64),
            validation_split=p.get("validation_split", 0.1),
            callbacks=callbacks,
            verbose=1,
            shuffle=False,  # preserve temporal order
        )

        self.is_fitted = True
        best_val_loss = min(self.history_.history.get("val_loss", [np.inf]))
        log.info("LSTM training complete — best val_loss={:.4f}", best_val_loss)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions.
        X must contain a 'demand' column with historical values.
        """
        self._check_fitted()
        if not self._available:
            raise ImportError("TensorFlow not installed.")

        demand = X["demand"].values.astype(np.float32)
        scaled = (demand - self._scaler_mean) / self._scaler_std

        preds = []
        for i in range(self.seq_length, len(scaled) + 1):
            seq = scaled[max(0, i - self.seq_length):i]
            if len(seq) < self.seq_length:
                # Pad with zeros at the beginning
                seq = np.pad(seq, (self.seq_length - len(seq), 0))
            seq_input = seq.reshape(1, self.seq_length, 1)
            p = self._tf_model.predict(seq_input, verbose=0)[0, 0]
            preds.append(p)

        preds_scaled_back = np.array(preds) * self._scaler_std + self._scaler_mean
        return preds_scaled_back.clip(0)

    def predict_sequence(self, seed: np.ndarray, n_steps: int) -> np.ndarray:
        """
        Multi-step ahead autoregressive forecast.

        Args:
            seed: Array of length >= seq_length (last known values, original scale).
            n_steps: Number of future timesteps to predict.

        Returns:
            Array of length n_steps with predicted demand.
        """
        self._check_fitted()
        scaled = (seed - self._scaler_mean) / self._scaler_std
        buffer = list(scaled[-self.seq_length:])
        preds = []

        for _ in range(n_steps):
            seq = np.array(buffer[-self.seq_length:], dtype=np.float32).reshape(1, -1, 1)
            next_val = self._tf_model.predict(seq, verbose=0)[0, 0]
            preds.append(next_val)
            buffer.append(next_val)

        result = np.array(preds) * self._scaler_std + self._scaler_mean
        return result.clip(0)

    def save(self, path: str | Path) -> None:
        """Save Keras model weights + scaler parameters."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._tf_model.save(str(path / "keras_model"))
        import json
        meta = {
            "name": self.name,
            "params": self.params,
            "seq_length": self.seq_length,
            "scaler_mean": float(self._scaler_mean),
            "scaler_std": float(self._scaler_std),
        }
        with open(path / "meta.json", "w") as f:
            json.dump(meta, f)
        log.info("LSTM saved → {}", path)

    @classmethod
    def load(cls, path: str | Path) -> "LSTMForecaster":
        """Restore LSTM from saved directory."""
        import json
        import tensorflow as tf

        path = Path(path)
        with open(path / "meta.json") as f:
            meta = json.load(f)

        instance = cls.__new__(cls)
        instance.params = meta["params"]
        instance.seq_length = meta["seq_length"]
        instance._scaler_mean = meta["scaler_mean"]
        instance._scaler_std = meta["scaler_std"]
        instance.name = meta["name"]
        instance.is_fitted = True
        instance._available = True
        instance._tf_model = tf.keras.models.load_model(str(path / "keras_model"))
        instance.feature_names = None
        instance.history_ = None
        log.info("LSTM loaded ← {}", path)
        return instance

    # ── Private ───────────────────────────────────────────────────────

    def _build_model(self, input_shape: tuple):
        """Build the LSTM architecture."""
        import tensorflow as tf

        p = self.params
        lstm_units = p.get("lstm_units", [128, 64])
        dense_units = p.get("dense_units", [32])
        dropout = p.get("dropout_rate", 0.2)
        rec_dropout = p.get("recurrent_dropout", 0.1)
        lr = p.get("learning_rate", 0.001)

        inp = tf.keras.layers.Input(shape=input_shape)
        x = inp

        for i, units in enumerate(lstm_units):
            return_sequences = (i < len(lstm_units) - 1)
            x = tf.keras.layers.LSTM(
                units,
                return_sequences=return_sequences,
                dropout=dropout,
                recurrent_dropout=rec_dropout,
            )(x)
            x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Dropout(dropout)(x)

        for units in dense_units:
            x = tf.keras.layers.Dense(units, activation="relu")(x)
            x = tf.keras.layers.Dropout(dropout / 2)(x)

        out = tf.keras.layers.Dense(1, activation="linear")(x)

        model = tf.keras.Model(inputs=inp, outputs=out)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="huber",               # robust to outliers
            metrics=["mae"],
        )
        return model

    @staticmethod
    def _check_tf() -> bool:
        try:
            import tensorflow  # noqa: F401
            return True
        except ImportError:
            log.warning("TensorFlow not available — LSTM will not run")
            return False

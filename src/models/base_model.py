"""
Abstract Base Model
===================
Defines the interface all forecasting models must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.utils.helpers import save_artifact, load_artifact
from src.utils.logger import get_logger

log = get_logger(__name__)


class BaseForecaster(ABC):
    """
    Abstract interface for all demand forecasting models.
    Ensures consistent train/predict/save/load API across model types.
    """

    name: str = "base"

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self.params = params or {}
        self.model: Any = None
        self.is_fitted: bool = False
        self.feature_names: Optional[list[str]] = None

    # ── Abstract Interface ────────────────────────────────────────────

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseForecaster":
        """Train the model on X, y."""
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions for feature matrix X."""
        ...

    # ── Shared Methods ────────────────────────────────────────────────

    def fit_predict(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Train and return in-sample predictions."""
        return self.fit(X, y).predict(X)

    def save(self, path: str | Path) -> None:
        """Persist the fitted model to disk."""
        if not self.is_fitted:
            raise RuntimeError(f"Model '{self.name}' has not been fitted yet.")
        save_artifact(self, path)
        log.info("Model '{}' saved → {}", self.name, path)

    @classmethod
    def load(cls, path: str | Path) -> "BaseForecaster":
        """Restore a fitted model from disk."""
        model = load_artifact(path)
        log.info("Model '{}' loaded ← {}", model.name, path)
        return model

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                f"Model '{self.name}' is not fitted. Call fit() first."
            )

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "unfitted"
        return f"{self.__class__.__name__}(name={self.name!r}, status={status})"

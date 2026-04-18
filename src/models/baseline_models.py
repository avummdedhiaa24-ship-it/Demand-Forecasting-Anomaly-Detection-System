"""
Baseline Models
===============
Linear Regression, Ridge, and Lasso as interpretable baselines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge

from src.models.base_model import BaseForecaster
from src.utils.config import cfg
from src.utils.logger import get_logger, log_execution_time

log = get_logger(__name__)


class LinearRegressionForecaster(BaseForecaster):
    """Ordinary least squares linear regression baseline."""

    name = "linear_regression"

    def __init__(self) -> None:
        params = cfg.models.baseline.linear_regression.to_dict()
        super().__init__(params)
        self.model = LinearRegression(**params)

    @log_execution_time
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LinearRegressionForecaster":
        log.info("Training {} on {} samples × {} features", self.name, len(X), X.shape[1])
        self.feature_names = list(X.columns)
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self.model.predict(X).clip(0)


class RidgeForecaster(BaseForecaster):
    """Ridge regression (L2 regularisation)."""

    name = "ridge"

    def __init__(self) -> None:
        params = cfg.models.baseline.ridge.to_dict()
        super().__init__(params)
        self.model = Ridge(**params)

    @log_execution_time
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RidgeForecaster":
        log.info("Training {} — alpha={}", self.name, self.params.get("alpha"))
        self.feature_names = list(X.columns)
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self.model.predict(X).clip(0)


class LassoForecaster(BaseForecaster):
    """Lasso regression (L1 regularisation, induces sparsity)."""

    name = "lasso"

    def __init__(self) -> None:
        params = cfg.models.baseline.lasso.to_dict()
        super().__init__(params)
        self.model = Lasso(**params)

    @log_execution_time
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LassoForecaster":
        log.info("Training {} — alpha={}", self.name, self.params.get("alpha"))
        self.feature_names = list(X.columns)
        self.model.fit(X, y)
        self.is_fitted = True

        n_zero = (self.model.coef_ == 0).sum()
        log.info(
            "Lasso: {} / {} features zeroed out (sparsity={:.1%})",
            n_zero, len(self.model.coef_), n_zero / len(self.model.coef_),
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self.model.predict(X).clip(0)

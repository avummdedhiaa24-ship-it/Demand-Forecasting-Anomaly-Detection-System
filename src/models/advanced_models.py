"""
Advanced ML Models
==================
Random Forest, XGBoost, and LightGBM for demand forecasting.
All support feature importance extraction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.models.base_model import BaseForecaster
from src.utils.config import cfg
from src.utils.logger import get_logger, log_execution_time

log = get_logger(__name__)


class RandomForestForecaster(BaseForecaster):
    """Random Forest ensemble regressor."""

    name = "random_forest"

    def __init__(self) -> None:
        params = cfg.models.advanced.random_forest.to_dict()
        super().__init__(params)
        self.model = RandomForestRegressor(**params)

    @log_execution_time
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestForecaster":
        log.info(
            "Training {} — {} estimators on {} samples × {} features",
            self.name, self.params.get("n_estimators"), len(X), X.shape[1],
        )
        self.feature_names = list(X.columns)
        self.model.fit(X, y)
        self.is_fitted = True

        top_feats = pd.Series(
            self.model.feature_importances_, index=X.columns
        ).nlargest(5)
        log.info("Top-5 features: {}", top_feats.to_dict())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self.model.predict(X).clip(0)

    @property
    def feature_importances(self) -> pd.Series:
        self._check_fitted()
        return pd.Series(
            self.model.feature_importances_, index=self.feature_names
        ).sort_values(ascending=False)


class XGBoostForecaster(BaseForecaster):
    """XGBoost gradient boosting regressor."""

    name = "xgboost"

    def __init__(self) -> None:
        params = cfg.models.advanced.xgboost.to_dict()
        super().__init__(params)
        try:
            import xgboost as xgb
            self.model = xgb.XGBRegressor(**params)
            self._available = True
        except ImportError:
            log.warning("xgboost not installed — XGBoostForecaster unavailable")
            self._available = False

    @log_execution_time
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostForecaster":
        if not self._available:
            raise ImportError("xgboost package not installed.")
        log.info("Training {} on {} samples", self.name, len(X))
        self.feature_names = list(X.columns)
        self.model.fit(
            X, y,
            eval_set=[(X, y)],
            verbose=False,
        )
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self.model.predict(X).clip(0)

    @property
    def feature_importances(self) -> pd.Series:
        self._check_fitted()
        return pd.Series(
            self.model.feature_importances_, index=self.feature_names
        ).sort_values(ascending=False)


class LightGBMForecaster(BaseForecaster):
    """LightGBM gradient boosting regressor."""

    name = "lightgbm"

    def __init__(self) -> None:
        params = cfg.models.advanced.lightgbm.to_dict()
        super().__init__(params)
        try:
            import lightgbm as lgb
            self.model = lgb.LGBMRegressor(**params)
            self._available = True
        except ImportError:
            log.warning("lightgbm not installed — LightGBMForecaster unavailable")
            self._available = False

    @log_execution_time
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LightGBMForecaster":
        if not self._available:
            raise ImportError("lightgbm package not installed.")
        log.info("Training {} on {} samples", self.name, len(X))
        self.feature_names = list(X.columns)
        self.model.fit(
            X, y,
            eval_set=[(X, y)],
            callbacks=[],
        )
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self.model.predict(X).clip(0)

    @property
    def feature_importances(self) -> pd.Series:
        self._check_fitted()
        return pd.Series(
            self.model.feature_importances_, index=self.feature_names
        ).sort_values(ascending=False)

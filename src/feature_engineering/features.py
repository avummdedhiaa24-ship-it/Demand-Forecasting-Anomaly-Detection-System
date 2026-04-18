"""
Feature Engineering Pipeline
=============================
Generates lag features, rolling statistics, calendar features,
and cyclical encodings for the demand forecasting task.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor

from src.utils.config import cfg
from src.utils.logger import get_logger, log_execution_time

log = get_logger(__name__)


class FeatureEngineer:
    """
    Transforms a cleaned time-series DataFrame into a rich ML feature matrix.

    Features generated:
      - Lag features: demand(t-k) for configured lags
      - Rolling stats: mean, std, min, max over configured windows
      - Calendar: hour, day-of-week, month, week-of-year, is_weekend
      - Cyclical: sin/cos encoding for periodic calendar features
      - Holiday flag (basic UK/US holidays)

    Usage:
        fe = FeatureEngineer()
        X_train = fe.fit_transform(train_df)
        X_test  = fe.transform(test_df)
    """

    def __init__(self) -> None:
        self.fe_cfg = cfg.feature_engineering
        self.selected_features: Optional[List[str]] = None
        self._feature_names: List[str] = []

    # ── Public API ────────────────────────────────────────────────────

    @log_execution_time
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build features on training data and record selected feature names."""
        df = self._build_all_features(df)
        self._feature_names = [
            c for c in df.columns
            if c not in ("timestamp", "demand", "is_anomaly")
        ]
        log.info("Generated {} features", len(self._feature_names))
        return df

    @log_execution_time
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply identical feature engineering to new data."""
        return self._build_all_features(df)

    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        top_k: int = 40,
    ) -> List[str]:
        """
        Select top-k features using Random Forest importance.

        Args:
            X: Feature matrix (no target column).
            y: Target series.
            top_k: Number of features to keep.

        Returns:
            List of selected feature names.
        """
        log.info("Running feature selection (top_k={})", top_k)
        rf = RandomForestRegressor(
            n_estimators=50,
            max_depth=8,
            n_jobs=-1,
            random_state=42,
        )
        rf.fit(X, y)

        importances = pd.Series(rf.feature_importances_, index=X.columns)
        top_features = importances.nlargest(top_k).index.tolist()
        self.selected_features = top_features

        log.info(
            "Selected {} features — top 5: {}",
            len(top_features),
            importances.nlargest(5).to_dict(),
        )
        return top_features

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names

    # ── Feature Builders ─────────────────────────────────────────────

    def _build_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Orchestrate all feature generation steps."""
        df = df.copy()
        df = df.set_index("timestamp") if "timestamp" in df.columns else df

        df = self._lag_features(df)
        df = self._rolling_features(df)
        df = self._calendar_features(df)
        if self.fe_cfg.use_cyclical_encoding:
            df = self._cyclical_features(df)
        if self.fe_cfg.use_holiday_features:
            df = self._holiday_features(df)

        # Drop NaN rows created by lag/rolling (only in training)
        n_before = len(df)
        df = df.dropna()
        n_after = len(df)
        if n_before != n_after:
            log.debug("Dropped {} NaN rows after feature engineering", n_before - n_after)

        df = df.reset_index()
        return df

    def _lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lag features: demand at t-k hours."""
        for lag in self.fe_cfg.lag_features:
            df[f"demand_lag_{lag}h"] = df["demand"].shift(lag)
        log.debug("Lag features: {}", self.fe_cfg.lag_features)
        return df

    def _rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling window statistics over demand."""
        stats = self.fe_cfg.rolling_stats
        for window in self.fe_cfg.rolling_windows:
            rolling = df["demand"].shift(1).rolling(window=window, min_periods=1)
            if "mean" in stats:
                df[f"demand_roll_mean_{window}h"] = rolling.mean()
            if "std" in stats:
                df[f"demand_roll_std_{window}h"] = rolling.std().fillna(0)
            if "min" in stats:
                df[f"demand_roll_min_{window}h"] = rolling.min()
            if "max" in stats:
                df[f"demand_roll_max_{window}h"] = rolling.max()
        log.debug("Rolling features windows: {}", self.fe_cfg.rolling_windows)
        return df

    def _calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract calendar-based features from the timestamp index."""
        idx = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.DatetimeIndex(df.index)

        df["hour"] = idx.hour.astype(np.int8)
        df["day_of_week"] = idx.dayofweek.astype(np.int8)      # 0=Mon, 6=Sun
        df["day_of_month"] = idx.day.astype(np.int8)
        df["month"] = idx.month.astype(np.int8)
        df["quarter"] = idx.quarter.astype(np.int8)
        df["week_of_year"] = idx.isocalendar().week.astype(np.int8).values
        df["day_of_year"] = idx.dayofyear.astype(np.int16)
        df["is_weekend"] = (idx.dayofweek >= 5).astype(np.int8)
        df["is_month_start"] = idx.is_month_start.astype(np.int8)
        df["is_month_end"] = idx.is_month_end.astype(np.int8)

        # Hour buckets (peak / off-peak)
        df["is_peak_hour"] = (
            ((df["hour"] >= 7) & (df["hour"] <= 9)) |
            ((df["hour"] >= 17) & (df["hour"] <= 21))
        ).astype(np.int8)

        df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(np.int8)

        log.debug("Calendar features added")
        return df

    def _cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode periodic features using sin/cos transforms so that
        the model understands hour 23 and hour 0 are adjacent.
        """
        cycles = {
            "hour": 24,
            "day_of_week": 7,
            "month": 12,
            "day_of_year": 365,
        }
        for feat, period in cycles.items():
            if feat in df.columns:
                angle = 2 * np.pi * df[feat] / period
                df[f"{feat}_sin"] = np.sin(angle).astype(np.float32)
                df[f"{feat}_cos"] = np.cos(angle).astype(np.float32)
        log.debug("Cyclical encodings applied")
        return df

    def _holiday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mark common UK public holidays as a binary feature."""
        idx = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.DatetimeIndex(df.index)

        # Fixed-date holidays (month-day)
        fixed_holidays = {
            (1, 1),   # New Year's Day
            (12, 25), # Christmas Day
            (12, 26), # Boxing Day
            (7, 4),   # US Independence Day (for generalisation)
        }

        is_holiday = pd.Series(
            [(m, d) in fixed_holidays for m, d in zip(idx.month, idx.day)],
            index=df.index,
            dtype=np.int8,
        )
        df["is_holiday"] = is_holiday
        log.debug("Holiday features added")
        return df


# ── Convenience ───────────────────────────────────────────────────────

def get_feature_target_split(
    df: pd.DataFrame,
    target_col: str = "demand",
    drop_cols: Optional[List[str]] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split engineered DataFrame into feature matrix X and target y.

    Args:
        df: Feature-engineered DataFrame.
        target_col: Name of the target column.
        drop_cols: Additional columns to exclude from X.

    Returns:
        (X, y) tuple.
    """
    exclude = {"timestamp", target_col, "is_anomaly"}
    if drop_cols:
        exclude.update(drop_cols)

    X = df.drop(columns=[c for c in exclude if c in df.columns])
    y = df[target_col]

    # Remove any remaining object columns
    obj_cols = X.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        log.warning("Dropping object columns from X: {}", obj_cols)
        X = X.drop(columns=obj_cols)

    return X, y

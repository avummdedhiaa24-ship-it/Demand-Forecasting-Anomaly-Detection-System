"""
Data Preprocessing & Cleaning Pipeline
=======================================
Handles null imputation, outlier treatment, timestamp reindexing,
and scaling for the demand forecasting system.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from src.utils.config import cfg
from src.utils.helpers import save_artifact, load_artifact
from src.utils.logger import get_logger, log_execution_time

log = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────

SCALER_MAP = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


# ── Cleaner ───────────────────────────────────────────────────────────

class DataCleaner:
    """
    Production-grade data cleaning pipeline for time-series demand data.

    Steps:
      1. Reindex to full regular time-series grid
      2. Impute missing values
      3. Detect and treat outliers
      4. Clip domain-specific value ranges
      5. Fit and apply feature scaler
    """

    def __init__(self) -> None:
        self.pp_cfg = cfg.preprocessing
        self.scaler: Optional[StandardScaler] = None
        self._stats: dict = {}

    # ── Public ────────────────────────────────────────────────────────

    @log_execution_time
    def fit_transform(
        self,
        df: pd.DataFrame,
        freq: str | None = None,
    ) -> pd.DataFrame:
        """
        Full cleaning pipeline (training mode).
        Fits scalers/stats in addition to transforming.
        """
        freq = freq or cfg.data.frequency
        df = df.copy()

        log.info("Starting cleaning pipeline — {} rows", len(df))

        df = self._reindex(df, freq)
        df = self._impute_missing(df)
        df, outlier_mask = self._treat_outliers(df)
        df = self._clip_range(df)

        # Record statistics for monitoring
        self._stats["n_rows"] = len(df)
        self._stats["n_outliers"] = int(outlier_mask.sum())
        self._stats["demand_mean"] = float(df["demand"].mean())
        self._stats["demand_std"] = float(df["demand"].std())
        self._stats["demand_min"] = float(df["demand"].min())
        self._stats["demand_max"] = float(df["demand"].max())

        log.info(
            "Cleaning complete — {} rows | {} outliers treated | "
            "demand μ={:.1f} σ={:.1f}",
            self._stats["n_rows"],
            self._stats["n_outliers"],
            self._stats["demand_mean"],
            self._stats["demand_std"],
        )
        return df

    @log_execution_time
    def transform(self, df: pd.DataFrame, freq: str | None = None) -> pd.DataFrame:
        """Transform new data using already-fitted pipeline (inference mode)."""
        freq = freq or cfg.data.frequency
        df = df.copy()
        df = self._reindex(df, freq)
        df = self._impute_missing(df)
        _, _ = self._treat_outliers(df)  # detect only, don't refit
        df = self._clip_range(df)
        return df

    def fit_scaler(self, series: pd.Series) -> None:
        """Fit the demand scaler on training data."""
        ScalerClass = SCALER_MAP.get(self.pp_cfg.scaling_method, StandardScaler)
        self.scaler = ScalerClass()
        self.scaler.fit(series.values.reshape(-1, 1))
        log.info("Fitted {} scaler", self.pp_cfg.scaling_method)

    def scale(self, series: pd.Series) -> np.ndarray:
        """Apply fitted scaler."""
        if self.scaler is None:
            raise RuntimeError("Scaler not fitted. Call fit_scaler() first.")
        return self.scaler.transform(series.values.reshape(-1, 1)).flatten()

    def inverse_scale(self, arr: np.ndarray) -> np.ndarray:
        """Inverse scale predictions back to original units."""
        if self.scaler is None:
            raise RuntimeError("Scaler not fitted.")
        return self.scaler.inverse_transform(arr.reshape(-1, 1)).flatten()

    def save(self, path: str) -> None:
        """Persist the fitted cleaner."""
        save_artifact(self, path)

    @classmethod
    def load(cls, path: str) -> "DataCleaner":
        """Restore a fitted cleaner."""
        return load_artifact(path)

    @property
    def stats(self) -> dict:
        """Return computed statistics from the last fit_transform call."""
        return self._stats.copy()

    # ── Private ───────────────────────────────────────────────────────

    def _reindex(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """Reindex to a complete regular time grid, introducing NaNs for gaps."""
        df = df.set_index("timestamp").sort_index()
        full_idx = pd.date_range(df.index.min(), df.index.max(), freq=freq)
        n_before = len(df)
        df = df.reindex(full_idx)
        df.index.name = "timestamp"
        n_after = len(df)
        if n_after > n_before:
            log.warning(
                "Reindexed {} → {} rows ({} timestamps filled with NaN)",
                n_before, n_after, n_after - n_before,
            )
        return df.reset_index()

    def _impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute NaN values in demand column."""
        n_missing = df["demand"].isna().sum()
        if n_missing == 0:
            return df

        method = self.pp_cfg.missing_value_strategy
        log.info("Imputing {} missing values using '{}'", n_missing, method)

        if method == "interpolate":
            df["demand"] = df["demand"].interpolate(method="linear")
        elif method == "ffill":
            df["demand"] = df["demand"].ffill().bfill()
        elif method == "bfill":
            df["demand"] = df["demand"].bfill().ffill()
        elif method == "drop":
            df = df.dropna(subset=["demand"])
        else:
            raise ValueError(f"Unknown imputation strategy: {method}")

        # Any remaining NaNs (e.g., at boundaries) → forward fill
        df["demand"] = df["demand"].ffill().bfill()
        return df

    def _treat_outliers(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Detect and cap outliers using configured method.

        Returns:
            Cleaned DataFrame and boolean mask of detected outliers.
        """
        method = self.pp_cfg.outlier_method
        demand = df["demand"].copy()
        outlier_mask = pd.Series(False, index=df.index)

        if method in ("iqr", "both"):
            q1 = demand.quantile(0.25)
            q3 = demand.quantile(0.75)
            iqr = q3 - q1
            k = self.pp_cfg.iqr_multiplier
            lower, upper = q1 - k * iqr, q3 + k * iqr
            mask = (demand < lower) | (demand > upper)
            outlier_mask |= mask
            df.loc[mask, "demand"] = demand.clip(lower, upper)
            log.debug("IQR outliers: {} (bounds [{:.1f}, {:.1f}])", mask.sum(), lower, upper)

        if method in ("zscore", "both"):
            z = np.abs(stats.zscore(df["demand"].fillna(0)))
            threshold = self.pp_cfg.zscore_threshold
            mask = z > threshold
            outlier_mask |= mask
            median = df["demand"].median()
            df.loc[mask, "demand"] = median
            log.debug("Z-score outliers: {}", mask.sum())

        return df, outlier_mask

    def _clip_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clip demand to physically plausible range."""
        df["demand"] = df["demand"].clip(
            lower=self.pp_cfg.min_demand,
            upper=self.pp_cfg.max_demand,
        )
        return df


# ── Train/Val/Test Splitter ───────────────────────────────────────────

def temporal_split(
    df: pd.DataFrame,
    train_ratio: float | None = None,
    val_ratio: float | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time series data respecting temporal order.
    No shuffling — future data must not leak into training.

    Args:
        df: DataFrame sorted by timestamp.
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.

    Returns:
        (train_df, val_df, test_df) tuple.
    """
    train_ratio = train_ratio or cfg.data.train_ratio
    val_ratio = val_ratio or cfg.data.val_ratio

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    log.info(
        "Temporal split → train: {} ({:.0%}) | val: {} ({:.0%}) | test: {} ({:.0%})",
        len(train), train_ratio,
        len(val), val_ratio,
        len(test), 1 - train_ratio - val_ratio,
    )
    return train, val, test

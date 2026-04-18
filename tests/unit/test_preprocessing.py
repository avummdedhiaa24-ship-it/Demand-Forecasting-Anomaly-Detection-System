"""
Unit tests for the preprocessing module.
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.cleaner import DataCleaner, temporal_split


@pytest.fixture
def sample_df():
    """Generate a minimal synthetic DataFrame for testing."""
    n = 500
    timestamps = pd.date_range("2022-01-01", periods=n, freq="h")
    rng = np.random.default_rng(42)
    demand = 800 + 100 * np.sin(2 * np.pi * timestamps.hour / 24) + rng.normal(0, 30, n)
    demand = np.clip(demand, 0, None)
    return pd.DataFrame({"timestamp": timestamps, "demand": demand})


@pytest.fixture
def df_with_nulls(sample_df):
    """DataFrame with intentional NaN gaps."""
    df = sample_df.copy()
    df.loc[10:15, "demand"] = np.nan
    df.loc[100, "demand"] = np.nan
    return df


@pytest.fixture
def df_with_outliers(sample_df):
    """DataFrame with injected outliers."""
    df = sample_df.copy()
    df.loc[50, "demand"] = 9999.0
    df.loc[200, "demand"] = -500.0
    return df


# ── Tests: DataCleaner ────────────────────────────────────────────────

class TestDataCleaner:

    def test_fit_transform_returns_dataframe(self, sample_df):
        cleaner = DataCleaner()
        result = cleaner.fit_transform(sample_df)
        assert isinstance(result, pd.DataFrame)
        assert "timestamp" in result.columns
        assert "demand" in result.columns

    def test_no_nulls_after_cleaning(self, df_with_nulls):
        cleaner = DataCleaner()
        result = cleaner.fit_transform(df_with_nulls)
        assert result["demand"].isna().sum() == 0

    def test_demand_non_negative(self, df_with_outliers):
        cleaner = DataCleaner()
        result = cleaner.fit_transform(df_with_outliers)
        assert (result["demand"] >= 0).all()

    def test_outliers_treated_iqr(self, df_with_outliers):
        cleaner = DataCleaner()
        result = cleaner.fit_transform(df_with_outliers)
        # 9999 should be capped
        assert result["demand"].max() < 9999.0

    def test_stats_populated(self, sample_df):
        cleaner = DataCleaner()
        cleaner.fit_transform(sample_df)
        stats = cleaner.stats
        assert "n_rows" in stats
        assert "demand_mean" in stats
        assert stats["demand_mean"] > 0

    def test_scaler_fit_and_transform(self, sample_df):
        cleaner = DataCleaner()
        cleaner.fit_transform(sample_df)
        cleaner.fit_scaler(sample_df["demand"].dropna())
        scaled = cleaner.scale(sample_df["demand"].dropna())
        assert abs(scaled.mean()) < 1.0   # near zero for StandardScaler

    def test_inverse_scale_roundtrip(self, sample_df):
        cleaner = DataCleaner()
        cleaner.fit_transform(sample_df)
        cleaner.fit_scaler(sample_df["demand"].dropna())
        original = sample_df["demand"].dropna().values
        scaled = cleaner.scale(sample_df["demand"].dropna())
        reconstructed = cleaner.inverse_scale(scaled)
        np.testing.assert_allclose(original, reconstructed, rtol=1e-5)

    def test_missing_timestamp_reindex(self, sample_df):
        # Drop a few rows to simulate gaps
        df_gaps = sample_df.drop([5, 6, 7]).reset_index(drop=True)
        cleaner = DataCleaner()
        result = cleaner.fit_transform(df_gaps)
        # Should have same length as original after reindexing
        assert len(result) == len(sample_df)


# ── Tests: temporal_split ─────────────────────────────────────────────

class TestTemporalSplit:

    def test_split_sizes(self, sample_df):
        cleaner = DataCleaner()
        df = cleaner.fit_transform(sample_df)
        train, val, test = temporal_split(df, train_ratio=0.7, val_ratio=0.15)
        total = len(train) + len(val) + len(test)
        assert total == len(df)

    def test_temporal_order_preserved(self, sample_df):
        cleaner = DataCleaner()
        df = cleaner.fit_transform(sample_df)
        train, val, test = temporal_split(df)
        assert train["timestamp"].max() <= val["timestamp"].min()
        assert val["timestamp"].max() <= test["timestamp"].min()

    def test_no_data_leakage(self, sample_df):
        cleaner = DataCleaner()
        df = cleaner.fit_transform(sample_df)
        train, val, test = temporal_split(df)
        # Check no overlap
        train_ts = set(train["timestamp"])
        val_ts = set(val["timestamp"])
        test_ts = set(test["timestamp"])
        assert len(train_ts & val_ts) == 0
        assert len(train_ts & test_ts) == 0
        assert len(val_ts & test_ts) == 0

    def test_split_ratios(self, sample_df):
        cleaner = DataCleaner()
        df = cleaner.fit_transform(sample_df)
        n = len(df)
        train, val, test = temporal_split(df, train_ratio=0.6, val_ratio=0.2)
        assert abs(len(train) / n - 0.6) < 0.05
        assert abs(len(val) / n - 0.2) < 0.05

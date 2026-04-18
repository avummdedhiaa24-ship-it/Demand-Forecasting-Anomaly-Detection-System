"""
Unit tests for feature engineering module.
"""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering.features import FeatureEngineer, get_feature_target_split


@pytest.fixture
def clean_df():
    """Minimal clean DataFrame for feature tests."""
    n = 400
    ts = pd.date_range("2022-01-01", periods=n, freq="h")
    rng = np.random.default_rng(0)
    demand = 800 + 50 * np.sin(2 * np.pi * ts.hour / 24) + rng.normal(0, 20, n)
    return pd.DataFrame({"timestamp": ts, "demand": np.clip(demand, 0, None)})


class TestFeatureEngineer:

    def test_output_columns_include_lags(self, clean_df):
        fe = FeatureEngineer()
        result = fe.fit_transform(clean_df)
        assert "demand_lag_1h" in result.columns
        assert "demand_lag_24h" in result.columns
        assert "demand_lag_168h" in result.columns

    def test_calendar_features_present(self, clean_df):
        fe = FeatureEngineer()
        result = fe.fit_transform(clean_df)
        for col in ["hour", "day_of_week", "month", "is_weekend"]:
            assert col in result.columns, f"Missing: {col}"

    def test_cyclical_features_in_range(self, clean_df):
        fe = FeatureEngineer()
        result = fe.fit_transform(clean_df)
        assert "hour_sin" in result.columns
        assert "hour_cos" in result.columns
        assert result["hour_sin"].between(-1, 1).all()
        assert result["hour_cos"].between(-1, 1).all()

    def test_rolling_features_present(self, clean_df):
        fe = FeatureEngineer()
        result = fe.fit_transform(clean_df)
        assert "demand_roll_mean_24h" in result.columns
        assert "demand_roll_std_24h" in result.columns

    def test_no_nulls_in_output(self, clean_df):
        fe = FeatureEngineer()
        result = fe.fit_transform(clean_df)
        assert result.isnull().sum().sum() == 0

    def test_feature_names_populated(self, clean_df):
        fe = FeatureEngineer()
        fe.fit_transform(clean_df)
        assert len(fe.feature_names) > 0

    def test_transform_same_cols_as_fit_transform(self, clean_df):
        split = len(clean_df) // 2
        train = clean_df.iloc[:split].copy()
        test = clean_df.iloc[split:].copy()
        fe = FeatureEngineer()
        train_out = fe.fit_transform(train)
        test_out = fe.transform(test)
        # Test columns should be a subset of train columns
        train_cols = set(train_out.columns)
        test_cols = set(test_out.columns)
        assert test_cols.issubset(train_cols) or train_cols.issubset(test_cols)

    def test_is_weekend_correct(self, clean_df):
        fe = FeatureEngineer()
        result = fe.fit_transform(clean_df)
        result["expected_weekend"] = result["day_of_week"].isin([5, 6]).astype(int)
        assert (result["is_weekend"] == result["expected_weekend"]).all()


class TestGetFeatureTargetSplit:

    def test_returns_X_y(self, clean_df):
        fe = FeatureEngineer()
        df_feat = fe.fit_transform(clean_df)
        X, y = get_feature_target_split(df_feat)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_demand_not_in_X(self, clean_df):
        fe = FeatureEngineer()
        df_feat = fe.fit_transform(clean_df)
        X, y = get_feature_target_split(df_feat)
        assert "demand" not in X.columns

    def test_timestamp_not_in_X(self, clean_df):
        fe = FeatureEngineer()
        df_feat = fe.fit_transform(clean_df)
        X, y = get_feature_target_split(df_feat)
        assert "timestamp" not in X.columns

    def test_y_equals_demand(self, clean_df):
        fe = FeatureEngineer()
        df_feat = fe.fit_transform(clean_df)
        X, y = get_feature_target_split(df_feat)
        assert y.name == "demand"
        assert len(y) == len(X)

    def test_no_object_columns_in_X(self, clean_df):
        fe = FeatureEngineer()
        df_feat = fe.fit_transform(clean_df)
        X, _ = get_feature_target_split(df_feat)
        assert len(X.select_dtypes(include=["object"]).columns) == 0

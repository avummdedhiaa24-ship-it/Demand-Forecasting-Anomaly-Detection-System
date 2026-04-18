"""
Data Ingestion Pipeline
=======================
Handles downloading, loading, and schema-validating raw time-series data.
Uses the UCI Electricity Load Diagrams dataset (370 consumers, 15-min resolution).
We aggregate to hourly and use one consumer as the "demand" target.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from src.utils.config import cfg
from src.utils.logger import get_logger, log_execution_time
from src.utils.helpers import validate_dataframe, memory_usage, downcast_dtypes

log = get_logger(__name__)


# ── Schema Definition ─────────────────────────────────────────────────

REQUIRED_SCHEMA = {
    "timestamp": "datetime64[ns]",
    "demand": "float64",
}

OPTIONAL_COLUMNS = ["temperature", "holiday", "source"]


# ── Downloader ────────────────────────────────────────────────────────

class DataDownloader:
    """Downloads and caches the raw dataset."""

    def __init__(self, raw_dir: str | Path) -> None:
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    @log_execution_time
    def download_electricity_dataset(self) -> Path:
        """
        Download UCI Electricity Load Diagrams 2011-2014.
        Returns path to the extracted CSV.
        """
        dest = self.raw_dir / "LD2011_2014.txt"
        if dest.exists():
            log.info("Dataset already cached at {}", dest)
            return dest

        url = cfg.data.dataset_url
        log.info("Downloading electricity dataset from {}", url)

        response = requests.get(url, timeout=120, stream=True)
        response.raise_for_status()

        # The file is a zip
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            names = zf.namelist()
            log.debug("Zip contains: {}", names)
            zf.extractall(self.raw_dir)

        log.info("Dataset downloaded and extracted to {}", self.raw_dir)
        return dest

    @log_execution_time
    def generate_synthetic_dataset(
        self,
        n_days: int = 365 * 2,
        freq: str = "h",
        seed: int = 42,
    ) -> Path:
        """
        Generate a realistic synthetic electricity demand dataset.
        Used as fallback when download is unavailable.

        Pattern includes:
          - Daily seasonality (peak morning/evening)
          - Weekly seasonality (lower on weekends)
          - Annual seasonality (higher in summer/winter)
          - Random noise
          - Injected anomalies
        """
        rng = np.random.default_rng(seed)
        dest = self.raw_dir / "synthetic_demand.csv"

        if dest.exists():
            log.info("Synthetic dataset already exists at {}", dest)
            return dest

        log.info("Generating synthetic demand dataset ({} days, freq={})", n_days, freq)

        timestamps = pd.date_range("2021-01-01", periods=n_days * 24, freq=freq)
        n = len(timestamps)

        hours = timestamps.hour.values
        days_of_week = timestamps.dayofweek.values
        day_of_year = timestamps.dayofyear.values

        # Daily pattern: two peaks (morning 8am, evening 7pm)
        daily = (
            300 * np.sin(2 * np.pi * (hours - 6) / 24) +
            200 * np.sin(2 * np.pi * (hours - 17) / 12) +
            150
        )

        # Weekly pattern: -20% on weekends
        weekly = np.where(days_of_week >= 5, -80, 0)

        # Annual seasonal pattern
        annual = 100 * np.sin(2 * np.pi * day_of_year / 365)

        # Trend (slight upward)
        trend = np.linspace(800, 900, n)

        # Base demand
        demand = trend + daily + weekly + annual
        demand = np.clip(demand, 50, None)

        # Gaussian noise
        noise = rng.normal(0, 30, n)
        demand = demand + noise

        # Inject ~2% anomalies
        n_anomalies = int(0.02 * n)
        anomaly_idx = rng.choice(n, n_anomalies, replace=False)
        demand[anomaly_idx] += rng.choice([-400, 500, 600], n_anomalies)
        demand = np.clip(demand, 0, None)

        df = pd.DataFrame({
            "timestamp": timestamps,
            "demand": demand.round(2),
            "is_anomaly": False,  # ground truth
        })
        df.loc[anomaly_idx, "is_anomaly"] = True

        df.to_csv(dest, index=False)
        log.info("Synthetic dataset saved → {} ({} rows)", dest, len(df))
        return dest


# ── Loader ────────────────────────────────────────────────────────────

class DataLoader:
    """Loads raw data files into validated DataFrames."""

    def __init__(self, raw_dir: str | Path) -> None:
        self.raw_dir = Path(raw_dir)
        self.downloader = DataDownloader(raw_dir)

    @log_execution_time
    def load(self, path: Optional[str | Path] = None) -> pd.DataFrame:
        """
        Load dataset. Falls back to synthetic if no path given and
        the electricity dataset is unavailable.

        Args:
            path: Optional explicit file path.

        Returns:
            Validated DataFrame with at minimum [timestamp, demand] columns.
        """
        if path is None:
            # Try synthetic (guaranteed to work offline)
            synth_path = self.raw_dir / "synthetic_demand.csv"
            if not synth_path.exists():
                self.downloader.generate_synthetic_dataset()
            path = synth_path

        path = Path(path)
        log.info("Loading data from {}", path)

        suffix = path.suffix.lower()
        if suffix == ".csv":
            df = self._load_csv(path)
        elif suffix in (".txt", ".tsv"):
            df = self._load_txt(path)
        elif suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        df = self._enforce_schema(df)
        log.info(
            "Loaded {} rows × {} cols | Memory: {}",
            len(df), len(df.columns), memory_usage(df),
        )
        return df

    # ── Private ───────────────────────────────────────────────────────

    def _load_csv(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, parse_dates=["timestamp"])
        return df

    def _load_txt(self, path: Path) -> pd.DataFrame:
        """Load UCI electricity dataset (semicolon-delimited, European decimals)."""
        df = pd.read_csv(
            path,
            sep=";",
            decimal=",",
            parse_dates=True,
            index_col=0,
        )
        # Take first consumer column as proxy demand
        df.index = pd.to_datetime(df.index)
        df = df.resample("h").mean()  # aggregate 15-min → hourly
        first_col = df.columns[0]
        df = df[[first_col]].reset_index()
        df.columns = ["timestamp", "demand"]
        df["demand"] = df["demand"] * 4  # kWh per 15min → kWh per hour
        return df

    def _enforce_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and coerce DataFrame to required schema."""
        validate_dataframe(df, required_cols=["timestamp", "demand"], name="RawData")

        # Coerce types
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["demand"] = pd.to_numeric(df["demand"], errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Schema type check
        for col, dtype in REQUIRED_SCHEMA.items():
            actual = str(df[col].dtype)
            if not actual.startswith(dtype.replace("64", "")):
                log.warning("Column '{}' dtype '{}' — expected '{}'", col, actual, dtype)

        log.debug("Schema enforced — dtypes: {}", df.dtypes.to_dict())
        return df


# ── Missing Timestamp Detection ───────────────────────────────────────

def detect_missing_timestamps(
    df: pd.DataFrame,
    freq: str = "h",
    ts_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Find gaps in the time series index.

    Returns:
        DataFrame of missing timestamps with their expected positions.
    """
    full_range = pd.date_range(
        start=df[ts_col].min(),
        end=df[ts_col].max(),
        freq=freq,
    )
    missing = full_range.difference(df[ts_col])

    if len(missing) == 0:
        log.info("No missing timestamps detected.")
        return pd.DataFrame(columns=["timestamp", "gap_duration"])

    # Compute consecutive gap sizes
    gaps = []
    if len(missing) > 0:
        prev = missing[0]
        start = missing[0]
        for ts in missing[1:]:
            delta = ts - prev
            if delta > pd.Timedelta(freq):
                gaps.append({"start": start, "end": prev, "n_missing": int((prev - start) / pd.Timedelta(freq)) + 1})
                start = ts
            prev = ts
        gaps.append({"start": start, "end": prev, "n_missing": int((prev - start) / pd.Timedelta(freq)) + 1})

    result = pd.DataFrame({"timestamp": missing})
    log.warning(
        "Found {} missing timestamps across {} gaps",
        len(missing), len(gaps),
    )
    return result

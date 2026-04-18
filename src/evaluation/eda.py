"""
Exploratory Data Analysis (EDA)
================================
Trend/seasonality decomposition, ACF/PACF, distribution plots,
and time heatmaps. All outputs saved as PNG files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

from src.utils.config import cfg
from src.utils.logger import get_logger, log_execution_time

log = get_logger(__name__)

PLOT_DIR = Path(cfg.evaluation.plots_path) / "eda"
PLOT_STYLE = "seaborn-v0_8-whitegrid"
DPI = 120


class EDAAnalyser:
    """
    Runs a complete EDA suite on the cleaned demand time-series DataFrame.

    All plots are saved to artifacts/plots/eda/.
    """

    def __init__(self, output_dir: Optional[str] = None) -> None:
        self.output_dir = Path(output_dir) if output_dir else PLOT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────

    @log_execution_time
    def run_full_eda(self, df: pd.DataFrame) -> dict:
        """
        Execute all EDA steps and return summary statistics.

        Args:
            df: Cleaned DataFrame with [timestamp, demand] columns.

        Returns:
            Dict of summary statistics.
        """
        log.info("Starting EDA on {} rows", len(df))
        df = df.copy()

        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        df.index = pd.to_datetime(df.index)

        stats = self._summary_stats(df)
        self.plot_demand_overview(df)
        self.plot_seasonality_decomposition(df)
        self.plot_acf_pacf(df)
        self.plot_distribution(df)
        self.plot_hour_day_heatmap(df)
        self.plot_weekly_pattern(df)
        self.plot_monthly_boxplots(df)
        self.plot_rolling_statistics(df)

        log.info("EDA complete — plots saved to {}", self.output_dir)
        return stats

    # ── Individual Plots ──────────────────────────────────────────────

    def plot_demand_overview(self, df: pd.DataFrame) -> None:
        """Full time-series overview with rolling mean."""
        plt.style.use(PLOT_STYLE)
        fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True)

        # Top: raw demand
        axes[0].plot(df.index, df["demand"], lw=0.7, color="#1565C0", alpha=0.8)
        axes[0].plot(
            df.index,
            df["demand"].rolling(24 * 7).mean(),
            lw=2.5, color="#E53935", label="7-day rolling mean",
        )
        axes[0].set_title("Electricity Demand — Full History", fontsize=13)
        axes[0].set_ylabel("Demand (kWh)")
        axes[0].legend()

        # Bottom: daily range
        daily = df["demand"].resample("D").agg(["min", "max", "mean"])
        axes[1].fill_between(
            daily.index, daily["min"], daily["max"],
            alpha=0.3, color="#7B1FA2", label="Daily range",
        )
        axes[1].plot(daily.index, daily["mean"], lw=1.5, color="#7B1FA2", label="Daily mean")
        axes[1].set_title("Daily Demand Range", fontsize=13)
        axes[1].set_ylabel("Demand (kWh)")
        axes[1].set_xlabel("Date")
        axes[1].legend()

        fig.autofmt_xdate()
        fig.tight_layout()
        self._save(fig, "demand_overview.png")

    def plot_seasonality_decomposition(
        self, df: pd.DataFrame, period: int = 24 * 7
    ) -> None:
        """Seasonal decomposition (trend, seasonal, residual)."""
        try:
            result = seasonal_decompose(
                df["demand"].interpolate(),
                model="additive",
                period=period,
                extrapolate_trend="freq",
            )
        except Exception as exc:
            log.warning("Decomposition failed: {}", exc)
            return

        fig, axes = plt.subplots(4, 1, figsize=(18, 12), sharex=True)
        components = [
            (df["demand"], "Observed", "#1565C0"),
            (result.trend, "Trend", "#E53935"),
            (result.seasonal, "Seasonality", "#2E7D32"),
            (result.resid, "Residual", "#F57F17"),
        ]
        for ax, (series, title, color) in zip(axes, components):
            ax.plot(series.index, series.values, lw=0.9, color=color)
            ax.set_title(title, fontsize=11)
            ax.set_ylabel("kWh")

        fig.suptitle("Seasonal Decomposition (Weekly Period)", fontsize=14)
        fig.autofmt_xdate()
        fig.tight_layout()
        self._save(fig, "seasonality_decomposition.png")

    def plot_acf_pacf(
        self, df: pd.DataFrame, lags: int = 72, sample_size: int = 5000
    ) -> None:
        """ACF and PACF plots to identify AR/MA orders."""
        series = df["demand"].dropna().tail(sample_size)

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        plot_acf(series, lags=lags, ax=axes[0], alpha=0.05)
        axes[0].set_title(f"ACF (lags=0..{lags})", fontsize=12)
        axes[0].set_xlabel("Lag (hours)")

        plot_pacf(series, lags=lags, ax=axes[1], alpha=0.05, method="ywm")
        axes[1].set_title(f"PACF (lags=0..{lags})", fontsize=12)
        axes[1].set_xlabel("Lag (hours)")

        fig.suptitle("Autocorrelation Analysis", fontsize=14)
        fig.tight_layout()
        self._save(fig, "acf_pacf.png")

    def plot_distribution(self, df: pd.DataFrame) -> None:
        """Demand distribution: histogram + KDE + Q-Q plot."""
        plt.style.use(PLOT_STYLE)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        demand = df["demand"].dropna()

        # Histogram + KDE
        sns.histplot(demand, bins=80, kde=True, ax=axes[0], color="#1565C0")
        axes[0].axvline(demand.mean(), color="red", lw=2, label=f"Mean={demand.mean():.0f}")
        axes[0].axvline(demand.median(), color="orange", lw=2, label=f"Median={demand.median():.0f}")
        axes[0].set_title("Demand Distribution")
        axes[0].set_xlabel("Demand (kWh)")
        axes[0].legend()

        # Box plot
        axes[1].boxplot(demand, vert=True, patch_artist=True,
                        boxprops={"facecolor": "#BBDEFB"})
        axes[1].set_title("Box Plot")
        axes[1].set_ylabel("Demand (kWh)")
        axes[1].set_xticks([])

        # Q-Q plot
        from scipy import stats
        (osm, osr), (slope, intercept, r) = stats.probplot(demand, dist="norm")
        axes[2].scatter(osm, osr, s=4, alpha=0.4, color="#7B1FA2")
        lims = [min(osm), max(osm)]
        axes[2].plot(lims, [slope * x + intercept for x in lims], "r-", lw=2)
        axes[2].set_title(f"Q-Q Plot (R²={r**2:.4f})")
        axes[2].set_xlabel("Theoretical Quantiles")
        axes[2].set_ylabel("Sample Quantiles")

        fig.suptitle("Demand Distribution Analysis", fontsize=14)
        fig.tight_layout()
        self._save(fig, "distribution.png")

    def plot_hour_day_heatmap(self, df: pd.DataFrame) -> None:
        """Heatmap: average demand by hour (y) × day of week (x)."""
        df2 = df.copy()
        df2["hour"] = df2.index.hour
        df2["day_name"] = df2.index.day_name()

        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        pivot = (
            df2.groupby(["hour", "day_name"])["demand"]
            .mean()
            .unstack("day_name")
            .reindex(columns=day_order)
        )

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(
            pivot,
            ax=ax,
            cmap="YlOrRd",
            fmt=".0f",
            annot=True,
            linewidths=0.5,
            cbar_kws={"label": "Avg Demand (kWh)"},
        )
        ax.set_title("Average Demand by Hour and Day of Week", fontsize=14)
        ax.set_xlabel("Day of Week")
        ax.set_ylabel("Hour of Day")
        fig.tight_layout()
        self._save(fig, "hour_day_heatmap.png")

    def plot_weekly_pattern(self, df: pd.DataFrame) -> None:
        """Average demand profile for each day of week."""
        df2 = df.copy()
        df2["hour"] = df2.index.hour
        df2["dow"] = df2.index.dayofweek

        fig, ax = plt.subplots(figsize=(14, 6))
        colors = plt.cm.tab10.colors
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        for dow in range(7):
            profile = (
                df2[df2["dow"] == dow]
                .groupby("hour")["demand"]
                .mean()
            )
            style = "--" if dow >= 5 else "-"
            ax.plot(
                profile.index, profile.values,
                label=day_names[dow],
                lw=2, color=colors[dow], linestyle=style,
            )

        ax.set_title("Average Hourly Demand by Day of Week", fontsize=13)
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Average Demand (kWh)")
        ax.set_xticks(range(0, 24, 2))
        ax.legend(ncol=7, loc="upper left")
        ax.grid(True, alpha=0.4)
        fig.tight_layout()
        self._save(fig, "weekly_pattern.png")

    def plot_monthly_boxplots(self, df: pd.DataFrame) -> None:
        """Box plots of demand grouped by month."""
        df2 = df.copy()
        df2["month"] = df2.index.month

        months = {
            1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
            7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
        }
        df2["month_name"] = df2["month"].map(months)

        fig, ax = plt.subplots(figsize=(14, 6))
        month_order = list(months.values())
        groups = [
            df2[df2["month_name"] == m]["demand"].dropna().values
            for m in month_order
            if m in df2["month_name"].values
        ]
        bp = ax.boxplot(groups, patch_artist=True, notch=False)
        cmap = plt.cm.coolwarm
        for patch, color in zip(bp["boxes"], cmap(np.linspace(0, 1, len(groups)))):
            patch.set_facecolor(color)

        present_months = [m for m in month_order if m in df2["month_name"].values]
        ax.set_xticklabels(present_months, rotation=45)
        ax.set_title("Monthly Demand Distribution", fontsize=13)
        ax.set_xlabel("Month")
        ax.set_ylabel("Demand (kWh)")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        self._save(fig, "monthly_boxplots.png")

    def plot_rolling_statistics(self, df: pd.DataFrame) -> None:
        """Rolling mean and std to visualise trend and volatility."""
        sample = df["demand"].resample("6h").mean().dropna()

        fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

        for window, color in [(24 * 7, "#1565C0"), (24 * 30, "#E53935")]:
            axes[0].plot(
                sample.rolling(window).mean(),
                lw=1.5, color=color,
                label=f"{window//24}-day rolling mean",
            )
        axes[0].set_title("Rolling Mean (Trend)", fontsize=12)
        axes[0].set_ylabel("kWh")
        axes[0].legend()

        axes[1].plot(
            sample.rolling(24 * 7).std(),
            lw=1.5, color="#7B1FA2",
            label="7-day rolling std",
        )
        axes[1].set_title("Rolling Std (Volatility)", fontsize=12)
        axes[1].set_ylabel("kWh")
        axes[1].set_xlabel("Date")
        axes[1].legend()

        fig.autofmt_xdate()
        fig.tight_layout()
        self._save(fig, "rolling_statistics.png")

    # ── Private ───────────────────────────────────────────────────────

    def _summary_stats(self, df: pd.DataFrame) -> dict:
        demand = df["demand"].dropna()
        stats = {
            "n_observations": len(demand),
            "start_date": str(df.index.min()),
            "end_date": str(df.index.max()),
            "mean": round(float(demand.mean()), 2),
            "std": round(float(demand.std()), 2),
            "min": round(float(demand.min()), 2),
            "max": round(float(demand.max()), 2),
            "median": round(float(demand.median()), 2),
            "skewness": round(float(demand.skew()), 4),
            "kurtosis": round(float(demand.kurt()), 4),
            "pct_missing": round(float(demand.isna().mean() * 100), 3),
        }
        log.info("Summary stats: {}", stats)
        return stats

    def _save(self, fig: plt.Figure, filename: str) -> None:
        path = self.output_dir / filename
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        log.debug("EDA plot saved → {}", path)

"""
Model Evaluation
================
Implements RMSE, MAE, MAPE, R², time-series cross-validation,
and residual analysis with plot generation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server environments
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.models.base_model import BaseForecaster
from src.utils.config import cfg
from src.utils.helpers import smape, save_artifact
from src.utils.logger import get_logger, log_execution_time

log = get_logger(__name__)

PLOT_STYLE = "seaborn-v0_8-darkgrid"
FIGURE_DPI = 120


# ── Core Metrics ──────────────────────────────────────────────────────

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    mask = np.abs(y_true) > eps
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-8))


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "",
) -> Dict[str, float]:
    """Return all metrics as a dict."""
    p = f"{prefix}_" if prefix else ""
    return {
        f"{p}rmse": rmse(y_true, y_pred),
        f"{p}mae": mae(y_true, y_pred),
        f"{p}mape": mape(y_true, y_pred),
        f"{p}r2": r2(y_true, y_pred),
        f"{p}smape": smape(y_true, y_pred),
    }


# ── Time-Series Cross-Validation ──────────────────────────────────────

@log_execution_time
def time_series_cv(
    model: BaseForecaster,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int | None = None,
    gap: int | None = None,
) -> Dict[str, float]:
    """
    Expanding-window time-series cross-validation.

    Args:
        model: Unfitted (or re-fittable) model instance.
        X: Feature matrix.
        y: Target series.
        n_splits: Number of CV folds.
        gap: Timesteps gap between train and validation.

    Returns:
        Mean and std of each metric across folds.
    """
    n_splits = n_splits or cfg.evaluation.cv_splits
    gap = gap or cfg.evaluation.cv_gap

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    fold_metrics: List[Dict[str, float]] = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Re-instantiate to avoid state leakage
        fresh_model = model.__class__()
        fresh_model.fit(X_tr, y_tr)
        preds = fresh_model.predict(X_val)

        metrics = compute_all_metrics(y_val.values, preds, prefix=f"fold{fold}")
        fold_metrics.append(metrics)

        log.debug(
            "CV fold {} — RMSE={:.2f} MAE={:.2f} MAPE={:.2f}%",
            fold,
            metrics[f"fold{fold}_rmse"],
            metrics[f"fold{fold}_mae"],
            metrics[f"fold{fold}_mape"],
        )

    # Aggregate
    result: Dict[str, float] = {}
    for metric_base in ["rmse", "mae", "mape", "r2"]:
        values = [
            v
            for fold_dict in fold_metrics
            for k, v in fold_dict.items()
            if k.endswith(f"_{metric_base}")
        ]
        result[f"cv_{metric_base}_mean"] = float(np.mean(values))
        result[f"cv_{metric_base}_std"] = float(np.std(values))

    log.info(
        "CV ({} folds) — RMSE {:.2f} ± {:.2f} | MAPE {:.2f}% ± {:.2f}%",
        n_splits,
        result["cv_rmse_mean"], result["cv_rmse_std"],
        result["cv_mape_mean"], result["cv_mape_std"],
    )
    return result


# ── Model Comparison ──────────────────────────────────────────────────

class ModelEvaluator:
    """
    Evaluates and compares multiple forecasting models.
    Generates residual plots and model comparison charts.
    """

    def __init__(self, plots_dir: str | None = None) -> None:
        self.plots_dir = Path(plots_dir or cfg.evaluation.plots_path)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.results: Dict[str, Dict] = {}

    def evaluate(
        self,
        model: BaseForecaster,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        timestamps: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """
        Evaluate a fitted model on the test set.
        Generates and saves residual plots.
        """
        preds = model.predict(X_test)
        metrics = compute_all_metrics(y_test.values, preds, prefix="test")

        log.info(
            "[{}] Test — RMSE={:.2f} MAE={:.2f} MAPE={:.2f}% R²={:.4f}",
            model.name,
            metrics["test_rmse"],
            metrics["test_mae"],
            metrics["test_mape"],
            metrics["test_r2"],
        )

        self.results[model.name] = {
            "metrics": metrics,
            "predictions": preds,
            "actuals": y_test.values,
        }

        if cfg.evaluation.save_plots:
            self._plot_forecast(
                y_true=y_test.values,
                y_pred=preds,
                model_name=model.name,
                timestamps=timestamps,
            )
            self._plot_residuals(y_test.values, preds, model.name)

        return metrics

    def compare_models(self) -> pd.DataFrame:
        """Return a sorted comparison DataFrame of all evaluated models."""
        rows = []
        for name, data in self.results.items():
            row = {"model": name}
            row.update(data["metrics"])
            rows.append(row)

        df = pd.DataFrame(rows).set_index("model")
        df = df.sort_values("test_rmse")
        log.info("Model comparison:\n{}", df.to_string())
        return df

    def best_model_name(self) -> str:
        """Return the name of the model with lowest test RMSE."""
        return min(
            self.results.keys(),
            key=lambda k: self.results[k]["metrics"]["test_rmse"],
        )

    def save_results(self, path: str) -> None:
        """Persist evaluation results as JSON."""
        serialisable = {
            k: {
                "metrics": v["metrics"],
                "n_predictions": len(v["predictions"]),
            }
            for k, v in self.results.items()
        }
        with open(path, "w") as f:
            json.dump(serialisable, f, indent=2)
        log.info("Evaluation results saved → {}", path)

    # ── Plots ─────────────────────────────────────────────────────────

    def _plot_forecast(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        timestamps: Optional[pd.Series] = None,
        n_points: int = 500,
    ) -> None:
        plt.style.use(PLOT_STYLE)
        fig, ax = plt.subplots(figsize=(16, 5))

        idx = range(min(n_points, len(y_true)))
        x = (
            timestamps.iloc[list(idx)].values
            if timestamps is not None
            else list(idx)
        )

        ax.plot(x, y_true[list(idx)], label="Actual", lw=1.5, color="#2196F3")
        ax.plot(
            x, y_pred[list(idx)],
            label=f"Predicted ({model_name})",
            lw=1.5, color="#F44336", alpha=0.85, linestyle="--",
        )
        ax.set_title(f"Forecast vs Actual — {model_name}", fontsize=14)
        ax.set_xlabel("Time")
        ax.set_ylabel("Demand (kWh)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        if timestamps is not None:
            fig.autofmt_xdate()

        path = self.plots_dir / f"{model_name}_forecast.png"
        fig.tight_layout()
        fig.savefig(path, dpi=FIGURE_DPI)
        plt.close(fig)
        log.info("Forecast plot saved → {}", path)

    def _plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
    ) -> None:
        residuals = y_true - y_pred

        plt.style.use(PLOT_STYLE)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1. Residual time series
        axes[0].plot(residuals, lw=0.8, color="#9C27B0", alpha=0.7)
        axes[0].axhline(0, color="black", lw=1)
        axes[0].set_title("Residuals over Time")
        axes[0].set_xlabel("Index")
        axes[0].set_ylabel("Residual")

        # 2. Residual distribution
        axes[1].hist(residuals, bins=60, color="#4CAF50", edgecolor="white", alpha=0.8)
        axes[1].axvline(0, color="red", lw=2)
        axes[1].set_title("Residual Distribution")
        axes[1].set_xlabel("Residual Value")

        # 3. Predicted vs Actual scatter
        axes[2].scatter(y_pred, y_true, s=4, alpha=0.3, color="#FF9800")
        lims = [
            min(y_pred.min(), y_true.min()),
            max(y_pred.max(), y_true.max()),
        ]
        axes[2].plot(lims, lims, "r--", lw=1.5, label="Perfect Prediction")
        axes[2].set_title("Predicted vs Actual")
        axes[2].set_xlabel("Predicted")
        axes[2].set_ylabel("Actual")
        axes[2].legend()

        fig.suptitle(f"Residual Analysis — {model_name}", fontsize=14, y=1.02)
        fig.tight_layout()

        path = self.plots_dir / f"{model_name}_residuals.png"
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        log.info("Residual plot saved → {}", path)

"""
End-to-End Training Pipeline
=============================
Orchestrates: ingestion → preprocessing → feature engineering →
model training → evaluation → anomaly detector fitting → artifact saving.

Run with:  python -m src.pipeline
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.ingestion.data_loader import DataLoader, detect_missing_timestamps
from src.preprocessing.cleaner import DataCleaner, temporal_split
from src.feature_engineering.features import FeatureEngineer, get_feature_target_split
from src.models.baseline_models import LinearRegressionForecaster, RidgeForecaster, LassoForecaster
from src.models.advanced_models import RandomForestForecaster, XGBoostForecaster, LightGBMForecaster
from src.models.lstm_model import LSTMForecaster
from src.evaluation.metrics import ModelEvaluator, compute_all_metrics, time_series_cv
from src.evaluation.eda import EDAAnalyser
from src.anomaly_detection.detector import EnsembleDetector, results_to_dataframe
from src.api.database import (
    init_db, SessionLocal, ModelMetrics, TimeSeriesData,
)
from src.utils.config import cfg
from src.utils.helpers import save_artifact, timestamp_now
from src.utils.logger import get_logger, log_execution_time

log = get_logger(__name__)

ARTIFACTS_DIR = Path("artifacts")
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"


class TrainingPipeline:
    """
    Full ML training pipeline with configurable stages.
    """

    def __init__(
        self,
        run_eda: bool = True,
        train_lstm: bool = False,          # LSTM takes longer; set True when needed
        run_cv: bool = True,
        data_path: Optional[str] = None,
    ) -> None:
        self.run_eda = run_eda
        self.train_lstm = train_lstm
        self.run_cv = run_cv
        self.data_path = data_path
        self.results: dict = {}

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        METRICS_DIR.mkdir(parents=True, exist_ok=True)

    @log_execution_time
    def run(self) -> dict:
        """Execute the full pipeline. Returns summary dict."""
        start = time.time()
        log.info("=" * 60)
        log.info("🚀 Training Pipeline Started")
        log.info("=" * 60)

        # ── Stage 1: Ingest ───────────────────────────────────────────
        log.info("\n📥 Stage 1: Data Ingestion")
        loader = DataLoader(cfg.data.raw_path)
        df_raw = loader.load(self.data_path)
        missing = detect_missing_timestamps(df_raw)
        log.info("Raw data: {} rows | {} missing timestamps", len(df_raw), len(missing))

        # ── Stage 2: EDA ──────────────────────────────────────────────
        if self.run_eda:
            log.info("\n📊 Stage 2: Exploratory Data Analysis")
            eda = EDAAnalyser()
            eda_stats = eda.run_full_eda(df_raw)
            self.results["eda_stats"] = eda_stats

        # ── Stage 3: Preprocess ───────────────────────────────────────
        log.info("\n🧹 Stage 3: Preprocessing")
        cleaner = DataCleaner()
        df_clean = cleaner.fit_transform(df_raw, freq=cfg.data.frequency)
        train_df, val_df, test_df = temporal_split(df_clean)

        # Fit scaler on training demand
        cleaner.fit_scaler(train_df["demand"])
        save_artifact(cleaner, MODELS_DIR / "cleaner.pkl")
        log.info("Cleaner stats: {}", cleaner.stats)

        # ── Stage 4: Feature Engineering ──────────────────────────────
        log.info("\n⚙️  Stage 4: Feature Engineering")
        fe = FeatureEngineer()
        train_fe = fe.fit_transform(train_df)
        val_fe = fe.transform(val_df)
        test_fe = fe.transform(test_df)
        save_artifact(fe, MODELS_DIR / "feature_engineer.pkl")

        X_train, y_train = get_feature_target_split(train_fe)
        X_val, y_val = get_feature_target_split(val_fe)
        X_test, y_test = get_feature_target_split(test_fe)

        log.info(
            "Features: X_train={}, X_val={}, X_test={}",
            X_train.shape, X_val.shape, X_test.shape,
        )

        # Combine train+val for final model training
        X_trainval = pd.concat([X_train, X_val], ignore_index=True)
        y_trainval = pd.concat([y_train, y_val], ignore_index=True)

        # ── Stage 5: Model Training ───────────────────────────────────
        log.info("\n🤖 Stage 5: Model Training")
        evaluator = ModelEvaluator()

        models = [
            LinearRegressionForecaster(),
            RidgeForecaster(),
            LassoForecaster(),
            RandomForestForecaster(),
        ]

        # Add gradient boosting if packages available
        for Cls in [XGBoostForecaster, LightGBMForecaster]:
            try:
                m = Cls()
                if getattr(m, "_available", True):
                    models.append(m)
            except Exception as exc:
                log.warning("Skipping {}: {}", Cls.__name__, exc)

        test_timestamps = test_fe["timestamp"] if "timestamp" in test_fe.columns else None

        for model in models:
            log.info("Training {}...", model.name)
            try:
                model.fit(X_trainval, y_trainval)
                metrics = evaluator.evaluate(model, X_test, y_test, test_timestamps)
                save_artifact(model, MODELS_DIR / f"{model.name}.pkl")

                # Cross-validation
                if self.run_cv:
                    cv_metrics = time_series_cv(model, X_train, y_train)
                    metrics.update(cv_metrics)

                self.results[model.name] = metrics
            except Exception as exc:
                log.error("Model {} failed: {}", model.name, exc)

        # ── LSTM ──────────────────────────────────────────────────────
        if self.train_lstm:
            log.info("Training LSTM (deep learning)...")
            try:
                lstm = LSTMForecaster()
                if lstm._available:
                    lstm.fit(X_train, y_train)
                    # For LSTM predict, pass X_test with demand column
                    test_with_demand = test_fe[["demand"] + list(X_test.columns)].copy()
                    lstm_preds = lstm.predict(test_with_demand)
                    n = min(len(lstm_preds), len(y_test))
                    lstm_metrics = compute_all_metrics(
                        y_test.values[:n], lstm_preds[:n], prefix="test"
                    )
                    log.info("LSTM metrics: {}", lstm_metrics)
                    lstm.save(MODELS_DIR / "lstm")
                    self.results["lstm"] = lstm_metrics
            except Exception as exc:
                log.error("LSTM training failed: {}", exc)

        # ── Stage 6: Select Best Model ────────────────────────────────
        log.info("\n🏆 Stage 6: Model Selection")
        comparison = evaluator.compare_models()
        best_name = evaluator.best_model_name()
        log.info("Best model: {} (RMSE={:.2f})", best_name, comparison.loc[best_name, "test_rmse"])

        # Save best model as canonical model
        import shutil
        best_path = MODELS_DIR / f"{best_name}.pkl"
        canonical_path = MODELS_DIR / "best_model.pkl"
        if best_path.exists():
            shutil.copy(best_path, canonical_path)
            log.info("Best model saved → {}", canonical_path)

        evaluator.save_results(str(METRICS_DIR / "evaluation_results.json"))

        # ── Stage 7: Anomaly Detector Fitting ────────────────────────
        log.info("\n🚨 Stage 7: Fitting Anomaly Detector")
        try:
            best_model = models[
                [m.name for m in models].index(best_name)
                if best_name in [m.name for m in models]
                else 0
            ]
            train_preds = best_model.predict(X_train)
            detector = EnsembleDetector()
            detector.fit(X_train, y_train, train_preds)
            save_artifact(detector, MODELS_DIR / "ensemble_detector.pkl")

            # Run detection on test set for summary
            test_preds = best_model.predict(X_test)
            anomaly_results = detector.predict(
                X_test, y_test.values, test_preds, test_timestamps,
            )
            anomaly_df = results_to_dataframe(anomaly_results)
            anomaly_df.to_csv(METRICS_DIR / "test_anomalies.csv", index=False)
            n_anom = anomaly_df["is_anomaly"].sum()
            log.info("Test anomalies: {}/{}", n_anom, len(anomaly_df))
        except Exception as exc:
            log.error("Anomaly detector fitting failed: {}", exc)

        # ── Stage 8: Persist Metrics to DB ────────────────────────────
        log.info("\n💾 Stage 8: Persisting Metrics to Database")
        try:
            init_db()
            db = SessionLocal()
            best_metrics = self.results.get(best_name, {})
            db.add(ModelMetrics(
                model_name=best_name,
                model_version="1.0.0",
                rmse=best_metrics.get("test_rmse"),
                mae=best_metrics.get("test_mae"),
                mape=best_metrics.get("test_mape"),
                r2=best_metrics.get("test_r2"),
                n_train_samples=len(X_trainval),
                n_test_samples=len(X_test),
                training_duration_s=round(time.time() - start, 2),
                notes=f"Best model from pipeline run at {timestamp_now()}",
            ))
            db.commit()
            db.close()
            log.info("Metrics persisted to DB")
        except Exception as exc:
            log.warning("DB persist failed: {}", exc)

        # ── Summary ───────────────────────────────────────────────────
        elapsed = time.time() - start
        summary = {
            "status": "success",
            "best_model": best_name,
            "elapsed_seconds": round(elapsed, 1),
            "n_models_trained": len([m for m in models if m.is_fitted]),
            "model_metrics": self.results,
            "artifacts_dir": str(MODELS_DIR),
        }

        with open(METRICS_DIR / "pipeline_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        log.info("=" * 60)
        log.info("✅ Pipeline complete in {:.1f}s", elapsed)
        log.info("   Best model: {} | RMSE={:.2f}", best_name,
                 self.results.get(best_name, {}).get("test_rmse", 0))
        log.info("=" * 60)
        return summary


# ── Entry Point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the training pipeline")
    parser.add_argument("--no-eda", action="store_true", help="Skip EDA plots")
    parser.add_argument("--lstm", action="store_true", help="Train LSTM model")
    parser.add_argument("--no-cv", action="store_true", help="Skip cross-validation")
    parser.add_argument("--data", type=str, default=None, help="Path to data file")
    args = parser.parse_args()

    pipeline = TrainingPipeline(
        run_eda=not args.no_eda,
        train_lstm=args.lstm,
        run_cv=not args.no_cv,
        data_path=args.data,
    )
    result = pipeline.run()
    print("\n📋 Pipeline Summary:")
    print(json.dumps(result, indent=2, default=str))

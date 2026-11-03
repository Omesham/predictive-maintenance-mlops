"""Rolling-window monitoring with condition-matched reference and calibrated drift thresholds.

Designed for the 25-feature C-MAPSS XGBoost champion. Users upload ordinary CSVs;
no dataset identifier columns or special filenames are needed.
"""

import hashlib
import json
import logging
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from prometheus_client import Gauge
from scipy.spatial import cKDTree
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

from app.api.predict import get_model

logger = logging.getLogger("predictive_maintenance")
REFERENCE_DIRECTORY = Path("/app/data/curated/train")
WINDOW_ROWS = 500
CHECK_EVERY_ROWS = 100
RETAIN_ROWS = 20000
REFERENCE_ROWS_PER_FILE = 1500
MATCH_NEIGHBORS = 3
CALIBRATION_WINDOWS_PER_DATASET = 5
DEFAULT_DISTANCE_THRESHOLD = 0.1

# Existing Grafana queries remain unchanged.
drifted_features_gauge = Gauge("data_drifted_features", "Number of features currently detected as drifted")
total_features_gauge = Gauge("data_drift_total_features", "Total number of features monitored for drift")


@lru_cache(maxsize=1)
def get_engine():
    required = ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"Missing PostgreSQL environment variables: {missing}")

    url = URL.create("postgresql+psycopg", username=os.environ["POSTGRES_USER"], password=os.environ["POSTGRES_PASSWORD"], host=os.getenv("POSTGRES_HOST", "postgres"), port=int(os.getenv("POSTGRES_PORT", "5432")), database=os.environ["POSTGRES_DB"])
    return create_engine(url, pool_pre_ping=True)


@lru_cache(maxsize=1)
def initialize_drift_tables():
    with get_engine().begin() as connection:
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS drift_samples (
                id BIGSERIAL PRIMARY KEY,
                model_key TEXT NOT NULL,
                features JSONB NOT NULL,
                received_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """))
        connection.execute(text("CREATE INDEX IF NOT EXISTS idx_drift_samples_model_id ON drift_samples (model_key, id DESC)"))
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS drift_status (
                model_key TEXT PRIMARY KEY,
                last_processed_id BIGINT NOT NULL DEFAULT 0,
                total_features INTEGER NOT NULL,
                drifted_count INTEGER,
                drift_scores JSONB NOT NULL DEFAULT '{}'::jsonb,
                evaluated_at TIMESTAMPTZ
            )
        """))
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS drift_calibration (
                model_key TEXT PRIMARY KEY,
                thresholds JSONB NOT NULL,
                calibration_windows INTEGER NOT NULL,
                calibrated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """))


def get_model_context():
    model = get_model()
    schema = model.metadata.get_input_schema()
    if schema is None:
        raise ValueError("Model input schema is unavailable.")

    features = [column.name for column in schema.inputs]
    model_run = str(getattr(model.metadata, "run_id", None) or getattr(model.metadata, "model_uuid", None) or "model")
    model_key = hashlib.sha256((model_run + "|" + "|".join(features)).encode("utf-8")).hexdigest()
    return model_key, features


@lru_cache(maxsize=4)
def load_reference_data(feature_names: tuple[str, ...]) -> pd.DataFrame:
    parquet_files = sorted(REFERENCE_DIRECTORY.glob("train_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No training Parquet files found in {REFERENCE_DIRECTORY}.")

    datasets = []
    for file_path in parquet_files:
        data = pd.read_parquet(file_path)
        missing = [name for name in feature_names if name not in data.columns]
        if missing:
            raise ValueError(f"Missing reference features in {file_path.name}: {missing}")
        data = data.loc[:, list(feature_names)]
        datasets.append(data.sample(n=min(len(data), REFERENCE_ROWS_PER_FILE), random_state=42))

    return pd.concat(datasets, ignore_index=True)


def match_reference_data(reference_data: pd.DataFrame, current_data: pd.DataFrame) -> pd.DataFrame:
    """Match only on operating settings, not sensors or dataset labels."""
    settings = [name for name in reference_data.columns if name.startswith("operational_settings") and name in current_data.columns]
    if not settings:
        raise ValueError("No operational settings found for condition matching.")

    reference_settings = reference_data[settings].to_numpy(dtype=float)
    current_settings = current_data[settings].to_numpy(dtype=float)
    if not np.isfinite(reference_settings).all() or not np.isfinite(current_settings).all():
        raise ValueError("Operating settings must be finite numbers.")

    center = reference_settings.mean(axis=0)
    scale = reference_settings.std(axis=0)
    scale[scale < 1e-10] = 1.0
    tree = cKDTree((reference_settings - center) / scale)
    _, indices = tree.query((current_settings - center) / scale, k=min(MATCH_NEIGHBORS, len(reference_data)))
    indices = np.asarray(indices).reshape(-1)
    return reference_data.iloc[indices].reset_index(drop=True)


def evaluate_drift_scores(current_data: pd.DataFrame, reference_data: pd.DataFrame) -> dict[str, float]:
    """Use Evidently's distance metrics with a condition-matched reference."""
    matched_reference = match_reference_data(reference_data, current_data)
    if not np.isfinite(current_data.to_numpy(dtype=float)).all() or not np.isfinite(matched_reference.to_numpy(dtype=float)).all():
        raise ValueError("Drift comparison contains missing or non-finite feature values.")

    # Constant sensors produce undefined correlations in Evidently's summaries;
    # retain them for drift scoring, and require finite scores for every feature below.
    with np.errstate(divide="ignore", invalid="ignore"):
        snapshot = Report(metrics=[DataDriftPreset(num_method="wasserstein", num_threshold=DEFAULT_DISTANCE_THRESHOLD, cat_method="jensenshannon", cat_threshold=DEFAULT_DISTANCE_THRESHOLD)]).run(reference_data=matched_reference, current_data=current_data)
        metrics = snapshot.dict().get("metrics", [])
    scores = {}

    for metric in metrics:
        config = metric.get("config", {})
        if config.get("type") != "evidently:metric_v2:ValueDrift":
            continue
        column, score = config.get("column"), metric.get("value")
        if column is None or score is None or not np.isfinite(float(score)):
            continue
        scores[column] = float(score)

    missing = set(current_data.columns) - set(scores)
    if missing:
        raise RuntimeError(f"Evidently could not calculate drift scores for {sorted(missing)}")
    return scores


def get_calibrated_thresholds() -> dict[str, float]:
    initialize_drift_tables()
    model_key, _ = get_model_context()
    with get_engine().connect() as connection:
        thresholds = connection.execute(text("SELECT thresholds FROM drift_calibration WHERE model_key = :key"), {"key": model_key}).scalar_one_or_none()
    if thresholds is None:
        raise RuntimeError("Drift thresholds are not calibrated. Run: python -m app.services.drift --calibrate")
    return thresholds if isinstance(thresholds, dict) else json.loads(thresholds)


def detect_data_drift(current_data: pd.DataFrame) -> dict:
    _, feature_names = get_model_context()
    current_features = current_data.loc[:, feature_names].copy()
    reference_data = load_reference_data(tuple(feature_names))
    thresholds = get_calibrated_thresholds()
    scores = evaluate_drift_scores(current_features, reference_data)
    drifted = [name for name in feature_names if scores[name] > float(thresholds[name])]
    return {"total_features": len(scores), "drifted_count": len(drifted), "drifted_features": drifted, "drift_scores": scores}


def calibrate_drift_thresholds():
    """Estimate per-feature no-drift cutoffs from disjoint engine groups.

    Uses training engines to estimate false-alarm behavior, without using test
    datasets or model predictions. Requires independent held-out validation.
    """
    initialize_drift_tables()
    model_key, feature_names = get_model_context()
    parquet_files = sorted(REFERENCE_DIRECTORY.glob("train_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError("No reference Parquet files found.")

    reference_parts, holdouts = [], []
    for file_path in parquet_files:
        data = pd.read_parquet(file_path)
        missing = [name for name in [*feature_names, "unit_number"] if name not in data.columns]
        if missing:
            raise ValueError(f"Cannot calibrate {file_path.name}; missing: {missing}")
        engine_ids = data["unit_number"].drop_duplicates().sample(frac=1, random_state=42)
        holdout_ids = set(engine_ids.iloc[:max(1, len(engine_ids) // 5)])
        calibration_rows = data[data["unit_number"].isin(holdout_ids)].loc[:, feature_names]
        reference_rows = data[~data["unit_number"].isin(holdout_ids)].loc[:, feature_names]
        if len(calibration_rows) < WINDOW_ROWS or len(reference_rows) < WINDOW_ROWS:
            raise ValueError(f"Not enough independent engine readings to calibrate {file_path.name}")
        holdouts.append((file_path.stem, calibration_rows))
        reference_parts.append(reference_rows.sample(n=min(len(reference_rows), REFERENCE_ROWS_PER_FILE), random_state=42))

    reference_data = pd.concat(reference_parts, ignore_index=True)
    scores_by_feature = {name: [] for name in feature_names}
    for filename, holdout in holdouts:
        for seed in range(CALIBRATION_WINDOWS_PER_DATASET):
            current = holdout.sample(n=WINDOW_ROWS, random_state=100 + seed).reset_index(drop=True)
            scores = evaluate_drift_scores(current, reference_data)
            for feature in feature_names:
                scores_by_feature[feature].append(scores[feature])
        print(f"Calibrated source: {filename}", flush=True)

    # Pilot 95th percentile of same-population distances, never lower than 0.1.
    thresholds = {name: max(DEFAULT_DISTANCE_THRESHOLD, float(np.quantile(values, 0.95))) for name, values in scores_by_feature.items()}
    with get_engine().begin() as connection:
        connection.execute(text("""
            INSERT INTO drift_calibration (model_key, thresholds, calibration_windows)
            VALUES (:key, CAST(:thresholds AS jsonb), :windows)
            ON CONFLICT (model_key) DO UPDATE SET thresholds = EXCLUDED.thresholds,
                calibration_windows = EXCLUDED.calibration_windows, calibrated_at = NOW()
        """), {"key": model_key, "thresholds": json.dumps(thresholds), "windows": sum(len(v) for v in scores_by_feature.values()) // len(feature_names)})
    print(f"Saved pilot thresholds for {len(feature_names)} features across {sum(len(v) for v in scores_by_feature.values()) // len(feature_names)} same-source windows.", flush=True)

    # Re-evaluate the current stored window so Grafana never displays an old,
    # pre-calibration decision under the new threshold configuration.
    with get_engine().connect() as connection:
        count, latest_id = connection.execute(text("SELECT COUNT(*), MAX(id) FROM drift_samples WHERE model_key = :key"), {"key": model_key}).one()
        existing_window = connection.execute(text("SELECT features FROM drift_samples WHERE model_key = :key ORDER BY id DESC LIMIT :limit"), {"key": model_key, "limit": WINDOW_ROWS}).scalars().all() if count >= WINDOW_ROWS else []

    if existing_window:
        report = detect_data_drift(pd.DataFrame(existing_window))
        with get_engine().begin() as connection:
            connection.execute(text("""
                UPDATE drift_status SET last_processed_id = :latest, total_features = :total,
                    drifted_count = :drifted, drift_scores = CAST(:scores AS jsonb), evaluated_at = NOW()
                WHERE model_key = :key
            """), {"key": model_key, "latest": latest_id, "total": report["total_features"], "drifted": report["drifted_count"], "scores": json.dumps(report["drift_scores"])})
        print(f"Current rolling window re-evaluated: {report['drifted_count']}/{report['total_features']} drifted.", flush=True)
    return thresholds


def store_and_monitor(current_data: pd.DataFrame):
    """Background task: store validated feature rows and evaluate rolling windows."""
    try:
        initialize_drift_tables()
        model_key, feature_names = get_model_context()
        rows = [{"model_key": model_key, "features": json.dumps(record, allow_nan=False)} for record in current_data.loc[:, feature_names].to_dict("records")]
        if not rows:
            return

        with get_engine().begin() as connection:
            connection.execute(text("SELECT pg_advisory_xact_lock(hashtext(:key))"), {"key": model_key})
            connection.execute(text("INSERT INTO drift_samples (model_key, features) VALUES (:model_key, CAST(:features AS jsonb))"), rows)
            connection.execute(text("INSERT INTO drift_status (model_key, total_features) VALUES (:key, :n) ON CONFLICT (model_key) DO NOTHING"), {"key": model_key, "n": len(feature_names)})
            connection.execute(text("DELETE FROM drift_samples WHERE model_key = :key AND id < (SELECT MAX(id) - :keep FROM drift_samples WHERE model_key = :key)"), {"key": model_key, "keep": RETAIN_ROWS})
            state = connection.execute(text("SELECT last_processed_id FROM drift_status WHERE model_key = :key"), {"key": model_key}).scalar_one()
            count, latest_id = connection.execute(text("SELECT COUNT(*), MAX(id) FROM drift_samples WHERE model_key = :key"), {"key": model_key}).one()

            if count < WINDOW_ROWS or (state and latest_id - state < CHECK_EVERY_ROWS):
                logger.info("Drift waiting for more readings: %s/%s available", min(count, WINDOW_ROWS), WINDOW_ROWS)
                return

            window = connection.execute(text("SELECT features FROM drift_samples WHERE model_key = :key ORDER BY id DESC LIMIT :limit"), {"key": model_key, "limit": WINDOW_ROWS}).scalars().all()
            report = detect_data_drift(pd.DataFrame(window))
            connection.execute(text("""
                UPDATE drift_status
                SET last_processed_id = :latest, total_features = :total, drifted_count = :drifted,
                    drift_scores = CAST(:scores AS jsonb), evaluated_at = NOW()
                WHERE model_key = :key
            """), {"key": model_key, "latest": latest_id, "total": report["total_features"], "drifted": report["drifted_count"], "scores": json.dumps(report["drift_scores"])})
            logger.info("Rolling data drift (condition-matched): %s/%s features drifted over %s readings", report["drifted_count"], report["total_features"], WINDOW_ROWS)
    except Exception:
        logger.exception("Rolling data drift monitoring failed; prediction is unaffected")


def get_latest_metric(column: str) -> float:
    try:
        initialize_drift_tables()
        model_key, _ = get_model_context()
        if column not in ("drifted_count", "total_features"):
            raise ValueError("Invalid drift metric")
        with get_engine().connect() as connection:
            result = connection.execute(text(f"SELECT {column} FROM drift_status WHERE model_key = :key AND evaluated_at IS NOT NULL"), {"key": model_key}).scalar_one_or_none()
        return float(result) if result is not None else float("nan")
    except Exception:
        logger.exception("Unable to read drift status for Prometheus")
        return float("nan")


drifted_features_gauge.set_function(lambda: get_latest_metric("drifted_count"))
total_features_gauge.set_function(lambda: get_latest_metric("total_features"))

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--calibrate":
        calibrate_drift_thresholds()
    else:
        raise SystemExit("Usage: python -m app.services.drift --calibrate")

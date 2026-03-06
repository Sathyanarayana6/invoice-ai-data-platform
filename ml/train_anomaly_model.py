"""
train_anomaly_model.py
-----------------------
Trains an Isolation Forest anomaly detection model with MLflow tracking.

What is Isolation Forest?
- An unsupervised ML algorithm (no labels needed)
- It works by randomly isolating data points
- Normal points are hard to isolate (need many cuts)
- Anomalies are easy to isolate (need few cuts)
- Returns an anomaly score for each record

Why unsupervised?
- We don't have labeled fraud data
- Real invoice fraud datasets are rare and expensive
- Isolation Forest finds statistical outliers without needing labels

What MLflow does here:
- Logs every experiment (parameters, metrics, model artifacts)
- Lets you compare runs visually
- Registers the best model for production use
- Run `mlflow ui` to see the dashboard
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
import json

# ── Config ─────────────────────────────────────────────────────────────────────

FEATURES_PATH  = "lake/features/invoice_features.parquet"
MODELS_PATH    = "lake/models"
MLFLOW_TRACKING= "./mlruns"   # Local folder where MLflow stores experiments

# Model hyperparameters to experiment with
EXPERIMENTS = [
    {"n_estimators": 100, "contamination": 0.05, "max_samples": "auto"},
    {"n_estimators": 200, "contamination": 0.08, "max_samples": "auto"},
    {"n_estimators": 150, "contamination": 0.05, "max_samples": 256},
]

# ── Feature Selection ──────────────────────────────────────────────────────────

# These are the numerical columns we feed into the model
# We exclude IDs and string columns — models only understand numbers
FEATURE_COLS = [
    "amount",
    "amount_zscore",
    "amount_log",
    "is_round_amount",
    "days_overdue",
    "invoice_month",
    "invoice_dow",
    "is_weekend",
    "invoice_quarter",
    "status_PAID",
    "status_PENDING",
    "status_OVERDUE",
    "status_DISPUTED",
    "status_CANCELLED",
    "vendor_total_invoices",
    "vendor_avg_amount",
    "vendor_overdue_rate",
    "vendor_dispute_rate",
    "vendor_risk_score",
    "amount_vs_vendor_avg",
]

# ── Data Loading ───────────────────────────────────────────────────────────────

def load_features() -> pd.DataFrame:
    """Loads the feature set from Feature Engineering step."""
    df = pd.read_parquet(FEATURES_PATH)

    # Keep only numeric feature columns that exist
    available = [c for c in FEATURE_COLS if c in df.columns]
    df_features = df[available].fillna(0)

    print(f"      Loaded {len(df_features):,} records, {len(available)} features")
    return df, df_features, available

# ── Model Training ─────────────────────────────────────────────────────────────

def train_and_log(
    df_full: pd.DataFrame,
    X: pd.DataFrame,
    feature_cols: list,
    params: dict,
    run_num: int
) -> tuple:
    """
    Trains one Isolation Forest model and logs everything to MLflow.

    What gets logged:
    - Parameters: n_estimators, contamination, max_samples
    - Metrics: anomaly_rate, mean_score, n_anomalies
    - Artifacts: the trained model, feature list, sample predictions
    - Tags: run metadata
    """

    with mlflow.start_run(run_name=f"isolation_forest_run_{run_num}"):

        # ── Log parameters ─────────────────────────────────────────────────────
        # Parameters are the settings you chose BEFORE training
        mlflow.log_params(params)
        mlflow.log_param("n_features",  len(feature_cols))
        mlflow.log_param("n_records",   len(X))

        # ── Scale features ─────────────────────────────────────────────────────
        # StandardScaler: transforms each feature to mean=0, std=1
        # Why? So that amount ($50,000) doesn't dominate over is_weekend (0 or 1)
        scaler  = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # ── Train model ────────────────────────────────────────────────────────
        model = IsolationForest(
            n_estimators  = params["n_estimators"],
            contamination = params["contamination"],
            max_samples   = params["max_samples"],
            random_state  = 42,
            n_jobs        = -1   # Use all CPU cores
        )
        model.fit(X_scaled)

        # ── Score all records ──────────────────────────────────────────────────
        # predict() returns: 1 = normal, -1 = anomaly
        # decision_function() returns: negative score = more anomalous
        predictions    = model.predict(X_scaled)
        anomaly_scores = model.decision_function(X_scaled)

        # ── Compute metrics ────────────────────────────────────────────────────
        # Metrics are measurements of model performance AFTER training
        n_anomalies  = (predictions == -1).sum()
        anomaly_rate = n_anomalies / len(predictions)
        mean_score   = anomaly_scores.mean()
        min_score    = anomaly_scores.min()

        # ── Log metrics ────────────────────────────────────────────────────────
        mlflow.log_metric("n_anomalies",  int(n_anomalies))
        mlflow.log_metric("anomaly_rate", round(float(anomaly_rate), 4))
        mlflow.log_metric("mean_score",   round(float(mean_score), 4))
        mlflow.log_metric("min_score",    round(float(min_score), 4))

        # ── Log model ──────────────────────────────────────────────────────────
        # This saves the actual trained model so we can load it later
        mlflow.sklearn.log_model(
            model,
            artifact_path="isolation_forest_model",
            registered_model_name="InvoiceAnomalyDetector"
        )

        # ── Save sample anomalies as artifact ─────────────────────────────────
        df_scored = df_full.copy()
        df_scored["anomaly_score"]     = anomaly_scores
        df_scored["is_anomaly"]        = (predictions == -1)
        df_scored["anomaly_label"]     = df_scored["is_anomaly"].map(
            {True: "ANOMALY", False: "NORMAL"}
        )

        # Top 20 most anomalous invoices
        top_anomalies = (
            df_scored[df_scored["is_anomaly"]]
            .sort_values("anomaly_score")
            .head(20)[["invoice_id", "vendor_id", "amount",
                        "anomaly_score", "anomaly_label"]]
        )

        os.makedirs("lake/models/tmp", exist_ok=True)
        anomaly_path = "lake/models/tmp/top_anomalies.csv"
        top_anomalies.to_csv(anomaly_path, index=False)
        mlflow.log_artifact(anomaly_path)

        # Log feature list
        feature_path = "lake/models/tmp/features_used.json"
        with open(feature_path, "w") as f:
            json.dump(feature_cols, f)
        mlflow.log_artifact(feature_path)

        # Get the MLflow run ID for reference
        run_id = mlflow.active_run().info.run_id

        print(f"\n  Run {run_num} results:")
        print(f"    n_estimators  : {params['n_estimators']}")
        print(f"    contamination : {params['contamination']}")
        print(f"    Anomalies found: {n_anomalies:,} ({anomaly_rate*100:.1f}%)")
        print(f"    MLflow run ID  : {run_id}")

        return model, scaler, anomaly_rate, run_id


# ── Main ───────────────────────────────────────────────────────────────────────

def run_training():
    print("=" * 50)
    print("MODEL TRAINING — Starting")
    print("=" * 50)

    # Set MLflow tracking location
    mlflow.set_tracking_uri(MLFLOW_TRACKING)
    mlflow.set_experiment("invoice_anomaly_detection")

    print("\n[1/3] Loading features...")
    df_full, X, feature_cols = load_features()

    print("\n[2/3] Running experiments...")
    print(f"      Training {len(EXPERIMENTS)} model variants\n")

    results = []
    for i, params in enumerate(EXPERIMENTS, 1):
        model, scaler, anomaly_rate, run_id = train_and_log(
            df_full, X, feature_cols, params, i
        )
        results.append({
            "run_num"      : i,
            "params"       : params,
            "anomaly_rate" : anomaly_rate,
            "run_id"       : run_id,
            "model"        : model,
            "scaler"       : scaler
        })

    print("\n[3/3] Experiment summary:")
    print(f"\n  {'Run':<5} {'n_est':<8} {'contam':<10} {'anomaly_rate':<15}")
    print(f"  {'-'*40}")
    for r in results:
        print(
            f"  {r['run_num']:<5} "
            f"{r['params']['n_estimators']:<8} "
            f"{r['params']['contamination']:<10} "
            f"{r['anomaly_rate']*100:.1f}%"
        )

    print(f"\n  MLflow UI: run `mlflow ui` in terminal then open http://localhost:5000")

    print("\n" + "=" * 50)
    print("MODEL TRAINING — Complete")
    print("=" * 50)

    return results


if __name__ == "__main__":
    run_training()
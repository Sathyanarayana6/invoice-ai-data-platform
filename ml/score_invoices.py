"""
score_invoices.py
------------------
Loads the trained model from MLflow and scores new invoices.

This is the "production inference" step:
- Training happens once (or periodically)
- Scoring happens on every new batch of invoices
- This script is what Airflow would trigger daily

What it produces:
- A scored dataset with anomaly flags
- A summary report of flagged invoices
- Saved to lake/gold/anomaly_scores/
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler
import os

# ── Config ─────────────────────────────────────────────────────────────────────

FEATURES_PATH   = "lake/features/invoice_features.parquet"
SCORES_PATH     = "lake/gold/anomaly_scores"
MLFLOW_TRACKING = "./mlruns"
MODEL_NAME      = "InvoiceAnomalyDetector"

FEATURE_COLS = [
    "amount", "amount_zscore", "amount_log", "is_round_amount",
    "days_overdue", "invoice_month", "invoice_dow", "is_weekend",
    "invoice_quarter", "status_PAID", "status_PENDING", "status_OVERDUE",
    "status_DISPUTED", "status_CANCELLED", "vendor_total_invoices",
    "vendor_avg_amount", "vendor_overdue_rate", "vendor_dispute_rate",
    "vendor_risk_score", "amount_vs_vendor_avg",
]

# ── Load Model ─────────────────────────────────────────────────────────────────

def load_latest_model():
    """
    Loads the latest version of our model from MLflow Model Registry.

    In production, you'd load the 'Production' stage model.
    We load the latest version here for simplicity.

    This is the key MLflow pattern:
    - Training script saves model to registry
    - Scoring script loads from registry by name
    - No hardcoded file paths — registry manages versions
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING)

    client = mlflow.tracking.MlflowClient()

    try:
        # Get latest model version
        versions = client.get_latest_versions(MODEL_NAME)
        if not versions:
            raise Exception(f"No versions found for model: {MODEL_NAME}")

        latest = versions[-1]
        model_uri = f"models:/{MODEL_NAME}/{latest.version}"
        print(f"      Loading model: {MODEL_NAME} v{latest.version}")
        print(f"      Model URI    : {model_uri}")

        model = mlflow.sklearn.load_model(model_uri)
        return model, latest.version

    except Exception as e:
        print(f"      MLflow registry failed: {e}")
        print("      Falling back to latest run...")

        # Fallback: load from latest experiment run
        experiment = mlflow.get_experiment_by_name("invoice_anomaly_detection")
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"]
        )
        run_id    = runs.iloc[0]["run_id"]
        model_uri = f"runs:/{run_id}/isolation_forest_model"
        model     = mlflow.sklearn.load_model(model_uri)
        print(f"      Loaded from run: {run_id}")
        return model, run_id


# ── Score Invoices ─────────────────────────────────────────────────────────────

def score_invoices(model, df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the trained model to score all invoices.

    Returns:
    - anomaly_score : continuous score (more negative = more anomalous)
    - is_anomaly    : boolean flag
    - risk_level    : human-readable category (LOW / MEDIUM / HIGH / CRITICAL)
    """
    available = [c for c in FEATURE_COLS if c in df_features.columns]
    X         = df_features[available].fillna(0)

    # Re-fit scaler on scoring data
    # In production you'd save/load the scaler too — simplified here
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Score
    predictions    = model.predict(X_scaled)
    anomaly_scores = model.decision_function(X_scaled)

    df_features = df_features.copy()
    df_features["anomaly_score"] = anomaly_scores
    df_features["is_anomaly"]    = (predictions == -1)

    # Risk levels based on score percentiles
    # This converts raw scores into actionable categories
    p25 = np.percentile(anomaly_scores, 25)
    p10 = np.percentile(anomaly_scores, 10)
    p5  = np.percentile(anomaly_scores, 5)

    def assign_risk(score):
        if score <= p5:
            return "CRITICAL"
        elif score <= p10:
            return "HIGH"
        elif score <= p25:
            return "MEDIUM"
        else:
            return "LOW"

    df_features["risk_level"] = df_features["anomaly_score"].apply(assign_risk)

    return df_features


# ── Generate Report ────────────────────────────────────────────────────────────

def generate_report(df_scored: pd.DataFrame):
    """Prints a summary of anomalies found."""

    total      = len(df_scored)
    anomalies  = df_scored["is_anomaly"].sum()
    critical   = (df_scored["risk_level"] == "CRITICAL").sum()
    high       = (df_scored["risk_level"] == "HIGH").sum()

    print(f"\n  Scoring Report:")
    print(f"  {'─'*40}")
    print(f"  Total invoices scored : {total:,}")
    print(f"  Anomalies detected    : {anomalies:,} ({anomalies/total*100:.1f}%)")
    print(f"  Critical risk         : {critical:,}")
    print(f"  High risk             : {high:,}")

    print(f"\n  Top 10 most anomalous invoices:")
    top = (
        df_scored[df_scored["is_anomaly"]]
        .sort_values("anomaly_score")
        .head(10)[["invoice_id", "vendor_id", "amount",
                   "anomaly_score", "risk_level"]]
    )
    print(top.to_string(index=False))


# ── Main ───────────────────────────────────────────────────────────────────────

def run_scoring():
    print("=" * 50)
    print("INVOICE SCORING — Starting")
    print("=" * 50)

    print("\n[1/4] Loading trained model from MLflow...")
    model, version = load_latest_model()
    print(f"      Model loaded successfully")

    print("\n[2/4] Loading invoice features...")
    df_features = pd.read_parquet(FEATURES_PATH)
    print(f"      {len(df_features):,} invoices to score")

    print("\n[3/4] Scoring invoices...")
    df_scored = score_invoices(model, df_features)

    print("\n[4/4] Saving scored results...")
    os.makedirs(SCORES_PATH, exist_ok=True)
    output_path = f"{SCORES_PATH}/scored_invoices.parquet"
    df_scored.to_parquet(output_path, index=False)
    print(f"      Saved to {output_path}")

    generate_report(df_scored)

    print("\n" + "=" * 50)
    print("INVOICE SCORING — Complete")
    print("=" * 50)

    return df_scored


if __name__ == "__main__":
    run_scoring()
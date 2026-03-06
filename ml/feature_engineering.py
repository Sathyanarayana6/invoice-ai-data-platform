"""
feature_engineering.py
------------------------
Transforms Gold layer data into ML-ready features.

What is Feature Engineering?
- Raw data (vendor names, dates, amounts) can't be fed directly into ML models
- Models only understand numbers
- Feature engineering converts business data into meaningful numerical signals

What features we create:
- Vendor-level behavioral features (payment patterns, risk indicators)
- Invoice-level features (amount z-scores, timing features)
- Aggregate features (rolling averages, ratios)

These features will train our anomaly detection model to flag:
- Duplicate payments
- Unusually large invoices from small vendors
- Invoices from vendors with sudden behavior changes
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import os

# ── Config ─────────────────────────────────────────────────────────────────────

GOLD_PATH     = "lake/gold"
SILVER_PATH   = "lake/silver/invoices"
FEATURES_PATH = "lake/features"

# ── Read Data ──────────────────────────────────────────────────────────────────

def read_silver() -> pd.DataFrame:
    """Reads clean Silver data for feature engineering."""
    import glob
    files = glob.glob(f"{SILVER_PATH}/**/*.parquet", recursive=True)
    dfs   = [pd.read_parquet(f) for f in files]
    df    = pd.concat(dfs, ignore_index=True)

    # Restore proper types
    df["amount"]       = pd.to_numeric(df["amount"],       errors="coerce")
    df["days_overdue"] = pd.to_numeric(df["days_overdue"], errors="coerce")
    df["invoice_date"] = pd.to_datetime(df["invoice_date"],errors="coerce")
    df["is_overdue"]   = df["is_overdue"].map({"True": True, "False": False})

    return df


def read_vendor_summary() -> pd.DataFrame:
    """Reads Gold vendor summary for vendor-level features."""
    df = pd.read_parquet(f"{GOLD_PATH}/vendor_summary/data.parquet")
    df["total_amount"]   = pd.to_numeric(df["total_amount"],   errors="coerce")
    df["avg_amount"]     = pd.to_numeric(df["avg_amount"],     errors="coerce")
    df["overdue_rate"]   = pd.to_numeric(df["overdue_rate"],   errors="coerce")
    df["dispute_rate"]   = pd.to_numeric(df["dispute_rate"],   errors="coerce")
    df["risk_score"]     = pd.to_numeric(df["risk_score"],     errors="coerce")
    df["total_invoices"] = pd.to_numeric(df["total_invoices"], errors="coerce")
    return df

# ── Feature Engineering Functions ─────────────────────────────────────────────

def create_invoice_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates invoice-level numerical features.

    Z-score: How many standard deviations away from the mean is this value?
    - A z-score of 0 = exactly average
    - A z-score of 3+ = very unusual (potential anomaly)
    - This is one of the most common anomaly detection techniques
    """
    features = pd.DataFrame()

    features["invoice_id"] = df["invoice_id"]
    features["vendor_id"]  = df["vendor_id"]

    # ── Amount features ────────────────────────────────────────────────────────

    # Raw amount
    features["amount"] = df["amount"].fillna(0)

    # Amount z-score across all invoices
    # High z-score = unusually large/small amount
    mean_amt = df["amount"].mean()
    std_amt  = df["amount"].std()
    features["amount_zscore"] = ((df["amount"] - mean_amt) / (std_amt + 1e-9)).fillna(0)

    # Log of amount (compresses large outliers, standard ML practice)
    features["amount_log"] = np.log1p(df["amount"].fillna(0))

    # Is amount suspiciously round? (e.g. exactly $10,000 — common in fraud)
    features["is_round_amount"] = (df["amount"] % 1000 == 0).astype(int)

    # ── Timing features ────────────────────────────────────────────────────────

    features["days_overdue"]   = df["days_overdue"].fillna(0)
    features["invoice_month"]  = df["invoice_date"].dt.month.fillna(0)
    features["invoice_dow"]    = df["invoice_date"].dt.dayofweek.fillna(0)  # 0=Mon, 6=Sun
    features["is_weekend"]     = (features["invoice_dow"] >= 5).astype(int)
    features["invoice_quarter"]= df["invoice_date"].dt.quarter.fillna(0)

    # ── Status features ────────────────────────────────────────────────────────

    # One-hot encode status (models need numbers not strings)
    status_dummies = pd.get_dummies(df["status"], prefix="status")
    for col in ["status_PAID", "status_PENDING", "status_OVERDUE",
                "status_DISPUTED", "status_CANCELLED"]:
        features[col] = status_dummies.get(col, 0).astype(int)

    # ── Category features ──────────────────────────────────────────────────────

    category_dummies = pd.get_dummies(df["category"], prefix="cat")
    features = pd.concat([features, category_dummies], axis=1)

    return features


def create_vendor_features(
    invoice_features: pd.DataFrame,
    vendor_summary: pd.DataFrame
) -> pd.DataFrame:
    """
    Joins vendor-level risk metrics onto each invoice.

    Why?
    - An invoice for $50,000 from a vendor whose average is $500 is suspicious
    - An invoice from a vendor with 40% overdue rate is higher risk
    - These vendor-level signals are powerful anomaly indicators
    """
    vendor_feats = vendor_summary[[
        "vendor_id",
        "total_invoices",
        "avg_amount",
        "overdue_rate",
        "dispute_rate",
        "risk_score",
        "total_amount"
    ]].copy()

    vendor_feats.columns = [
        "vendor_id",
        "vendor_total_invoices",
        "vendor_avg_amount",
        "vendor_overdue_rate",
        "vendor_dispute_rate",
        "vendor_risk_score",
        "vendor_total_amount"
    ]

    # Merge vendor features onto invoice features
    df = invoice_features.merge(vendor_feats, on="vendor_id", how="left")

    # How does this invoice's amount compare to vendor's average?
    # ratio > 3 means this invoice is 3x larger than normal for this vendor — suspicious
    df["amount_vs_vendor_avg"] = (
        df["amount"] / (df["vendor_avg_amount"] + 1e-9)
    ).fillna(1.0)

    return df


# ── Main ───────────────────────────────────────────────────────────────────────

def run_feature_engineering():
    print("=" * 50)
    print("FEATURE ENGINEERING — Starting")
    print("=" * 50)

    print("\n[1/5] Reading Silver layer...")
    silver_df = read_silver()
    print(f"      {len(silver_df):,} invoices loaded")

    print("\n[2/5] Reading Gold vendor summary...")
    vendor_df = read_vendor_summary()
    print(f"      {len(vendor_df):,} vendors loaded")

    print("\n[3/5] Creating invoice-level features...")
    invoice_features = create_invoice_features(silver_df)
    print(f"      Created {len(invoice_features.columns)} features")

    print("\n[4/5] Adding vendor-level features...")
    final_features = create_vendor_features(invoice_features, vendor_df)
    print(f"      Final feature set: {len(final_features.columns)} columns")

    print("\n[5/5] Saving feature set...")
    os.makedirs(FEATURES_PATH, exist_ok=True)
    final_features.to_parquet(f"{FEATURES_PATH}/invoice_features.parquet", index=False)
    print(f"      Saved to {FEATURES_PATH}/invoice_features.parquet")

    print("\nFeature summary:")
    print(f"  Records          : {len(final_features):,}")
    print(f"  Features         : {len(final_features.columns)}")
    print(f"  Avg amount zscore: {final_features['amount_zscore'].mean():.4f}")
    print(f"  Round amounts    : {final_features['is_round_amount'].sum():,}")

    print("\n" + "=" * 50)
    print("FEATURE ENGINEERING — Complete")
    print("=" * 50)

    return final_features


if __name__ == "__main__":
    run_feature_engineering()
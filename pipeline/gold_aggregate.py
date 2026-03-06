"""
gold_aggregate.py
------------------
Gold Layer — Business-Ready Aggregations

Rule: Answer business questions. This is what analysts, dashboards,
      and ML models read from.

What this script produces (3 Gold tables):
1. vendor_summary     — Per-vendor performance metrics
2. monthly_trends     — Invoice volume and value trends over time
3. category_analysis  — Which product categories drive the most revenue/disputes
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
from datetime import datetime

# ── Config ─────────────────────────────────────────────────────────────────────

SILVER_PATH = "lake/silver/invoices"
GOLD_PATH   = "lake/gold"

# ── Read Silver ────────────────────────────────────────────────────────────────

def read_silver(path: str) -> pd.DataFrame:
    """Reads the full clean Silver dataset."""
    table = pq.read_table(path)
    df    = table.to_pandas()

    # Convert back to proper types (Silver stored as string for Parquet)
    df["amount"]       = pd.to_numeric(df["amount"],       errors="coerce")
    df["days_overdue"] = pd.to_numeric(df["days_overdue"], errors="coerce")
    df["invoice_date"] = pd.to_datetime(df["invoice_date"],errors="coerce")
    df["is_overdue"]   = df["is_overdue"].map({"True": True, "False": False})

    print(f"      Read {len(df):,} records from Silver")
    return df

# ── Gold Table 1: Vendor Summary ───────────────────────────────────────────────

def build_vendor_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Answers: How is each vendor performing?

    Business questions this answers:
    - Which vendors send the most invoices?
    - Which vendors have the highest overdue rates?
    - Which vendors are highest value?
    - Which vendors have dispute problems?

    This is a classic GROUP BY aggregation — the foundation of analytics.
    """
    summary = df.groupby(["vendor_id", "vendor_name", "state"]).agg(
        total_invoices      = ("invoice_id",    "count"),
        total_amount        = ("amount",         "sum"),
        avg_amount          = ("amount",         "mean"),
        max_amount          = ("amount",         "max"),
        paid_count          = ("status",         lambda x: (x == "PAID").sum()),
        overdue_count       = ("status",         lambda x: (x == "OVERDUE").sum()),
        disputed_count      = ("status",         lambda x: (x == "DISPUTED").sum()),
        pending_count       = ("status",         lambda x: (x == "PENDING").sum()),
        avg_days_overdue    = ("days_overdue",   "mean"),
    ).reset_index()

    # Derived rate metrics — these are the KPIs business teams actually use
    summary["payment_rate"]  = (summary["paid_count"]    / summary["total_invoices"] * 100).round(2)
    summary["overdue_rate"]  = (summary["overdue_count"] / summary["total_invoices"] * 100).round(2)
    summary["dispute_rate"]  = (summary["disputed_count"]/ summary["total_invoices"] * 100).round(2)

    # Round monetary values
    summary["total_amount"] = summary["total_amount"].round(2)
    summary["avg_amount"]   = summary["avg_amount"].round(2)

    # Risk score — simple formula combining overdue and dispute rates
    # Higher = more risky vendor relationship
    summary["risk_score"] = (
        summary["overdue_rate"] * 0.6 +
        summary["dispute_rate"] * 0.4
    ).round(2)

    # Sort by total amount descending (highest value vendors first)
    summary = summary.sort_values("total_amount", ascending=False).reset_index(drop=True)

    return summary


# ── Gold Table 2: Monthly Trends ───────────────────────────────────────────────

def build_monthly_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Answers: How are invoice volumes and values trending over time?

    Business questions this answers:
    - Are we processing more invoices this month vs last?
    - Is total payment volume growing?
    - Are overdue rates getting better or worse?

    This is the data that feeds time-series charts in dashboards.
    """
    df["year_month"] = df["invoice_date"].dt.to_period("M").astype(str)

    trends = df.groupby("year_month").agg(
        total_invoices   = ("invoice_id", "count"),
        total_amount     = ("amount",     "sum"),
        avg_amount       = ("amount",     "mean"),
        paid_count       = ("status",     lambda x: (x == "PAID").sum()),
        overdue_count    = ("status",     lambda x: (x == "OVERDUE").sum()),
        disputed_count   = ("status",     lambda x: (x == "DISPUTED").sum()),
    ).reset_index()

    trends["overdue_rate"] = (trends["overdue_count"] / trends["total_invoices"] * 100).round(2)
    trends["total_amount"] = trends["total_amount"].round(2)
    trends["avg_amount"]   = trends["avg_amount"].round(2)

    # Month-over-month change in total amount
    trends = trends.sort_values("year_month")
    trends["amount_mom_change"] = trends["total_amount"].pct_change().round(4) * 100

    return trends


# ── Gold Table 3: Category Analysis ───────────────────────────────────────────

def build_category_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Answers: Which product categories drive the most revenue and risk?

    Business questions this answers:
    - Which categories have the most invoice volume?
    - Which categories have the highest dispute rates?
    - Where should we focus collections efforts?
    """
    analysis = df.groupby("category").agg(
        total_invoices   = ("invoice_id", "count"),
        total_amount     = ("amount",     "sum"),
        avg_amount       = ("amount",     "mean"),
        overdue_count    = ("status",     lambda x: (x == "OVERDUE").sum()),
        disputed_count   = ("status",     lambda x: (x == "DISPUTED").sum()),
        unique_vendors   = ("vendor_id",  "nunique"),
    ).reset_index()

    analysis["overdue_rate"]  = (analysis["overdue_count"] / analysis["total_invoices"] * 100).round(2)
    analysis["dispute_rate"]  = (analysis["disputed_count"]/ analysis["total_invoices"] * 100).round(2)
    analysis["total_amount"]  = analysis["total_amount"].round(2)
    analysis["avg_amount"]    = analysis["avg_amount"].round(2)

    analysis = analysis.sort_values("total_amount", ascending=False)

    return analysis


# ── Save Gold Tables ───────────────────────────────────────────────────────────

def save_gold_table(df: pd.DataFrame, table_name: str):
    """Saves a Gold table as a single Parquet file."""
    path = f"{GOLD_PATH}/{table_name}"
    os.makedirs(path, exist_ok=True)

    df["_created_at"] = datetime.utcnow().isoformat()
    table = pa.Table.from_pandas(df.astype(str))
    pq.write_table(table, f"{path}/data.parquet")
    print(f"      Saved {table_name}: {len(df):,} rows → {path}/data.parquet")


# ── Main ───────────────────────────────────────────────────────────────────────

def run_gold_aggregation():
    print("=" * 50)
    print("GOLD AGGREGATION — Starting")
    print("=" * 50)

    print("\n[1/5] Reading Silver layer...")
    df = read_silver(SILVER_PATH)

    print("\n[2/5] Building vendor_summary...")
    vendor_summary = build_vendor_summary(df)
    print(f"      Top vendor: {vendor_summary.iloc[0]['vendor_name']} "
          f"(${float(vendor_summary.iloc[0]['total_amount']):,.2f})")

    print("\n[3/5] Building monthly_trends...")
    monthly_trends = build_monthly_trends(df)

    print("\n[4/5] Building category_analysis...")
    category_analysis = build_category_analysis(df)

    print("\n[5/5] Saving Gold tables...")
    save_gold_table(vendor_summary,    "vendor_summary")
    save_gold_table(monthly_trends,    "monthly_trends")
    save_gold_table(category_analysis, "category_analysis")

    print("\nGold layer summary:")
    print(f"  Vendors tracked       : {len(vendor_summary):,}")
    print(f"  Months of data        : {len(monthly_trends):,}")
    print(f"  Categories analyzed   : {len(category_analysis):,}")
    print(f"  Total invoice value   : ${df['amount'].sum():,.2f}")
    print(f"  Overall overdue rate  : {(df['status']=='OVERDUE').mean()*100:.1f}%")

    print("\n" + "=" * 50)
    print("GOLD AGGREGATION — Complete")
    print("=" * 50)

    return vendor_summary, monthly_trends, category_analysis


if __name__ == "__main__":
    run_gold_aggregation()
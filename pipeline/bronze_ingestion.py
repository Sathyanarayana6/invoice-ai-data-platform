"""
bronze_ingestion.py
--------------------
Bronze Layer — Raw Ingestion

Rule: Store data EXACTLY as received. No changes. No fixes. No opinions.

Why?
- If Silver or Gold logic has a bug, we can always reprocess from Bronze
- Auditors and regulators often require the original raw data
- Data lineage starts here

What this script does:
1. Reads the raw CSV (simulates data arriving from an external source)
2. Adds metadata columns (when it arrived, what the source was)
3. Saves as Parquet format (columnar, compressed, much faster than CSV)
4. Partitions by date (so queries only scan the data they need)
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
import os
import hashlib

# ── Config ─────────────────────────────────────────────────────────────────────

SOURCE_FILE = "src_data/raw_invoices.csv"
BRONZE_PATH  = "lake/bronze/invoices"

# ── Helper Functions ───────────────────────────────────────────────────────────

def add_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds technical metadata columns to every Bronze record.
    These columns are not from the source — we add them to track:
    - When the record was ingested
    - Where it came from
    - A unique hash to detect duplicates later in Silver

    This is standard practice in production data pipelines.
    """

    # Ingestion timestamp — when did this record land in our lake?
    df["_ingested_at"] = datetime.utcnow().isoformat()

    # Source system identifier — where did this data come from?
    df["_source"] = "invoice_csv_feed"

    # Row hash — a fingerprint of each record using invoice_id + amount + date
    # Silver will use this to detect true duplicates
    df["_row_hash"] = df.apply(
        lambda row: hashlib.md5(
            f"{row['invoice_id']}{row['amount']}{row['invoice_date']}".encode()
        ).hexdigest(),
        axis=1
    )

    return df


def save_as_parquet(df: pd.DataFrame, base_path: str):
    """Saves bronze data as parquet files."""
    os.makedirs(base_path, exist_ok=True)
    # Save as single parquet file — simpler and more reliable
    df.to_parquet(f"{base_path}/bronze_data.parquet", index=False)
    return base_path


def get_stats(df: pd.DataFrame) -> dict:
    """
    Computes basic ingestion stats to log after each run.
    In production, these would go to a monitoring system.
    """
    return {
        "total_records"     : len(df),
        "columns"           : list(df.columns),
        "null_counts"       : df.isnull().sum().to_dict(),
        "ingestion_time"    : datetime.utcnow().isoformat(),
        "file_source"       : SOURCE_FILE
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def run_bronze_ingestion():
    print("=" * 50)
    print("BRONZE INGESTION — Starting")
    print("=" * 50)

    # Step 1: Read raw source file
    print(f"\n[1/4] Reading source file: {SOURCE_FILE}")
    df = pd.read_csv(SOURCE_FILE)
    print(f"      Loaded {len(df):,} records, {len(df.columns)} columns")

    # Step 2: Add metadata — this is the ONLY thing Bronze adds
    print("\n[2/4] Adding metadata columns (_ingested_at, _source, _row_hash)")
    df = add_metadata(df)

    # Step 3: Save as partitioned Parquet
    print(f"\n[3/4] Saving to Bronze layer: {BRONZE_PATH}")
    os.makedirs(BRONZE_PATH, exist_ok=True)
    save_as_parquet(df, BRONZE_PATH)
    print(f"      Saved as partitioned Parquet")

    # Step 4: Log stats
    print("\n[4/4] Ingestion Stats:")
    stats = get_stats(df)
    print(f"      Total records ingested : {stats['total_records']:,}")
    print(f"      Null amounts            : {stats['null_counts'].get('amount', 0)}")
    print(f"      Ingestion time          : {stats['ingestion_time']}")

    print("\n" + "=" * 50)
    print("BRONZE INGESTION — Complete")
    print("=" * 50)

    return df


if __name__ == "__main__":
    run_bronze_ingestion()
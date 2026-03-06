"""
silver_transform.py
--------------------
Silver Layer — Clean, Standardize, Validate

Rule: Fix everything. No dirty data leaves Silver.

What this script does:
1. Reads from Bronze layer
2. Removes duplicates (using the row hash we created in Bronze)
3. Standardizes date formats (all those inconsistent formats → one standard)
4. Fixes amounts (remove nulls, remove negatives)
5. Standardizes text fields (vendor names, status, category)
6. Adds derived columns (days_to_due, is_overdue)
7. Saves clean data as Parquet to Silver layer
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
from datetime import datetime

# ── Config ─────────────────────────────────────────────────────────────────────

BRONZE_PATH = "lake/bronze/invoices"
SILVER_PATH = "lake/silver/invoices"

# ── Transformation Functions ───────────────────────────────────────────────────

def read_bronze(path: str) -> pd.DataFrame:
    """
    Reads all Parquet files from the Bronze layer.
    """
    import glob
    # Find all parquet files recursively under the bronze path
    files = glob.glob(f"{path}/**/*.parquet", recursive=True)
    
    if not files:
        raise FileNotFoundError(f"No parquet files found in {path}")
    
    dfs = [pd.read_parquet(f) for f in files]
    df  = pd.concat(dfs, ignore_index=True)
    print(f"      Read {len(df):,} records from Bronze ({len(files)} files)")
    return df

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes duplicate records using the _row_hash column we created in Bronze.

    Why hash-based deduplication?
    - invoice_id alone isn't enough — same ID could be resubmitted with different amount
    - Hash covers invoice_id + amount + date together
    - Two records are only truly duplicate if ALL three match
    """
    before = len(df)
    df = df.drop_duplicates(subset=["_row_hash"], keep="first")
    after = len(df)
    removed = before - after
    print(f"      Removed {removed:,} duplicate records")
    return df


def standardize_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts all the inconsistent date formats to a single standard: YYYY-MM-DD

    Bronze had three formats:
    - 2024-01-15  (ISO format)
    - 01/15/2024  (US format)
    - 15-01-2024  (European format)

    pd.to_datetime with infer_datetime_format=True handles all of them.
    """
    # infer_datetime_format tries multiple formats automatically
    df["invoice_date"] = pd.to_datetime(
        df["invoice_date"],
        errors="coerce"   # If it can't parse, set to NaT (null date) — we handle below
    )
    df["due_date"] = pd.to_datetime(
        df["due_date"],
        errors="coerce"
    )

    # Drop records where date couldn't be parsed (unparseable = unusable)
    before = len(df)
    df = df.dropna(subset=["invoice_date"])
    print(f"      Dropped {before - len(df):,} records with unparseable dates")

    return df


def fix_amounts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles three amount problems:
    1. Null amounts → fill with 0.0 and flag them
    2. Negative amounts → data entry errors, make positive
    3. Convert from string (Bronze stored everything as string) to float
    """
    # Convert to numeric first (Bronze stored as string)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    # Flag records that had null amounts before we fill them
    df["_amount_was_null"] = df["amount"].isna()

    # Fill nulls with 0
    null_count = df["amount"].isna().sum()
    df["amount"] = df["amount"].fillna(0.0)
    print(f"      Filled {null_count:,} null amounts with 0.0")

    # Fix negative amounts (data entry errors — amounts can't be negative)
    neg_count = (df["amount"] < 0).sum()
    df["amount"] = df["amount"].abs()
    print(f"      Fixed {neg_count:,} negative amounts")

    # Round to 2 decimal places
    df["amount"] = df["amount"].round(2)

    return df


def standardize_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes text fields for consistency:
    - Uppercase status and category (PAID not paid/Paid)
    - Strip whitespace from vendor names
    - Title case vendor names
    """
    df["status"]      = df["status"].str.upper().str.strip()
    df["category"]    = df["category"].str.title().str.strip()
    df["vendor_name"] = df["vendor_name"].str.title().str.strip()
    df["state"]       = df["state"].str.upper().str.strip()

    return df


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds business-meaningful columns derived from existing data.
    These are too useful to recalculate every time in Gold.

    days_to_due     : How many days between invoice date and due date
    days_overdue    : How many days past due (0 if not overdue)
    is_overdue      : Boolean flag — is this invoice overdue?
    payment_quarter : Which quarter was this invoice issued in (Q1/Q2/Q3/Q4)
    """
    today = pd.Timestamp.today().normalize()

    # Days between invoice date and due date (should always be ~30)
    df["days_to_due"] = (df["due_date"] - df["invoice_date"]).dt.days

    # Days overdue — only positive if due_date is in the past AND status is not PAID
    df["days_overdue"] = (today - df["due_date"]).dt.days.clip(lower=0)
    df.loc[df["status"] == "PAID", "days_overdue"] = 0

    # Simple boolean overdue flag
    df["is_overdue"] = (df["status"] == "OVERDUE") | (
        (df["due_date"] < today) & (df["status"].isin(["PENDING"]))
    )

    # Quarter for time-based analysis
    df["payment_quarter"] = "Q" + df["invoice_date"].dt.quarter.astype(str)
    df["invoice_year"]    = df["invoice_date"].dt.year

    return df


def save_silver(df: pd.DataFrame, path: str):
    """
    Saves the clean Silver data as partitioned Parquet.
    Partitioned by year and quarter for efficient time-range queries.
    """
    df["_processed_at"] = datetime.utcnow().isoformat()

    # Convert dates to string for Parquet compatibility
    df["invoice_date"] = df["invoice_date"].dt.strftime("%Y-%m-%d")
    df["due_date"]     = df["due_date"].dt.strftime("%Y-%m-%d")

    os.makedirs(path, exist_ok=True)
    table = pa.Table.from_pandas(df.astype(str))

    pq.write_to_dataset(
        table,
        root_path=path,
        partition_cols=["invoice_year", "payment_quarter"],
        existing_data_behavior="overwrite_or_ignore"
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def run_silver_transform():
    print("=" * 50)
    print("SILVER TRANSFORM — Starting")
    print("=" * 50)

    print("\n[1/7] Reading Bronze layer...")
    df = read_bronze(BRONZE_PATH)

    print("\n[2/7] Removing duplicates...")
    df = remove_duplicates(df)

    print("\n[3/7] Standardizing dates...")
    df = standardize_dates(df)

    print("\n[4/7] Fixing amounts...")
    df = fix_amounts(df)

    print("\n[5/7] Standardizing text fields...")
    df = standardize_text(df)

    print("\n[6/7] Adding derived columns...")
    df = add_derived_columns(df)

    print(f"\n[7/7] Saving {len(df):,} clean records to Silver layer...")
    save_silver(df, SILVER_PATH)

    print("\nSilver layer summary:")
    print(f"  Clean records    : {len(df):,}")
    print(f"  Status breakdown :\n{df['status'].value_counts().to_string()}")
    print(f"  Amount range     : ${df['amount'].min():,.2f} – ${df['amount'].max():,.2f}")
    print(f"  Overdue invoices : {df['is_overdue'].sum():,}")

    print("\n" + "=" * 50)
    print("SILVER TRANSFORM — Complete")
    print("=" * 50)

    return df


if __name__ == "__main__":
    run_silver_transform()
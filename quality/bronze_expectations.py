import pandas as pd
import glob
import os

BRONZE_PATH = "lake/bronze/invoices"
REPORT_PATH = "lake/quality_reports"

def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  {status} - {name} {detail}")
    return condition

def run_bronze_expectations():
    print("BRONZE DATA QUALITY")
    files = glob.glob(f"{BRONZE_PATH}/**/*.parquet", recursive=True)
    if not files:
        files = glob.glob(f"{BRONZE_PATH}/*.parquet")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"Loaded {len(df)} records")
    results = []
    for col in ["invoice_id","vendor_id","vendor_name","invoice_date","amount","status"]:
        results.append(check(f"Column {col} exists", col in df.columns))
    results.append(check("invoice_id no nulls", df["invoice_id"].isna().sum()==0))
    results.append(check("vendor_id no nulls", df["vendor_id"].isna().sum()==0))
    results.append(check("Row count ok", 1000<=len(df)<=100000))
    for col in ["_ingested_at","_source","_row_hash"]:
        results.append(check(f"Metadata {col}", col in df.columns))
    passed = sum(results)
    failed = len(results)-passed
    print(f"Passed: {passed}/{len(results)}")
    os.makedirs(REPORT_PATH, exist_ok=True)
    return passed, failed

if __name__ == "__main__":
    run_bronze_expectations()

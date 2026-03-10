import pandas as pd
import glob
import os

SILVER_PATH = "lake/silver/invoices"
REPORT_PATH = "lake/quality_reports"

def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  {status} - {name} {detail}")
    return condition

def run_silver_expectations():
    print("SILVER DATA QUALITY")
    files = glob.glob(f"{SILVER_PATH}/**/*.parquet", recursive=True)
    if not files:
        files = glob.glob(f"{SILVER_PATH}/*.parquet")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["days_overdue"] = pd.to_numeric(df["days_overdue"], errors="coerce").fillna(0)
    print(f"Loaded {len(df)} records")
    results = []
    results.append(check("No null amounts", df["amount"].isna().sum()==0))
    results.append(check("Amounts >= 0", (df["amount"]>=0).all()))
    results.append(check("Amounts <= 1M", (df["amount"]<=1000000).all()))
    valid_s = ["PAID","PENDING","OVERDUE","DISPUTED","CANCELLED"]
    results.append(check("Valid statuses", df["status"].isin(valid_s).all()))
    results.append(check("No null invoice_id", df["invoice_id"].isna().sum()==0))
    results.append(check("No null vendor_id", df["vendor_id"].isna().sum()==0))
    results.append(check("invoice_id unique", df["invoice_id"].duplicated().sum()==0))
    results.append(check("days_overdue >= 0", (df["days_overdue"]>=0).all()))
    results.append(check("Row count ok", 1000<=len(df)<=15000, f"({len(df)})"))
    passed = sum(results)
    failed = len(results)-passed
    print(f"Passed: {passed}/{len(results)}")
    os.makedirs(REPORT_PATH, exist_ok=True)
    return passed, failed

if __name__ == "__main__":
    run_silver_expectations()

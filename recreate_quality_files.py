import os

bronze = """import pandas as pd
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
"""

silver = """import pandas as pd
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
"""

gold = """import pandas as pd
import os

GOLD_PATH = "lake/gold"
REPORT_PATH = "lake/quality_reports"

def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  {status} - {name} {detail}")
    return condition

def run_gold_expectations():
    print("GOLD DATA QUALITY")
    results = []
    vdf = pd.read_parquet(f"{GOLD_PATH}/vendor_summary/data.parquet")
    for col in ["total_amount","overdue_rate","risk_score","total_invoices"]:
        vdf[col] = pd.to_numeric(vdf[col], errors="coerce")
    results.append(check("Vendor count=50", len(vdf)==50))
    results.append(check("No null vendor names", vdf["vendor_name"].isna().sum()==0))
    results.append(check("total_amounts >= 0", (vdf["total_amount"]>=0).all()))
    results.append(check("overdue_rate 0-100", vdf["overdue_rate"].between(0,100).all()))
    results.append(check("No null risk scores", vdf["risk_score"].isna().sum()==0))
    tdf = pd.read_parquet(f"{GOLD_PATH}/monthly_trends/data.parquet")
    for col in ["total_amount","overdue_rate"]:
        tdf[col] = pd.to_numeric(tdf[col], errors="coerce")
    results.append(check("Month count 12-36", 12<=len(tdf)<=36))
    results.append(check("Monthly totals >= 0", (tdf["total_amount"]>=0).all()))
    results.append(check("No null year_month", tdf["year_month"].isna().sum()==0))
    cdf = pd.read_parquet(f"{GOLD_PATH}/category_analysis/data.parquet")
    cdf["total_amount"] = pd.to_numeric(cdf["total_amount"], errors="coerce")
    results.append(check("Category count=7", len(cdf)==7))
    results.append(check("Category totals >= 0", (cdf["total_amount"]>=0).all()))
    passed = sum(results)
    failed = len(results)-passed
    print(f"Passed: {passed}/{len(results)}")
    os.makedirs(REPORT_PATH, exist_ok=True)
    return passed, failed

if __name__ == "__main__":
    run_gold_expectations()
"""

with open("quality/bronze_expectations.py", "w", encoding="utf-8") as f: f.write(bronze)
with open("quality/silver_expectations.py", "w", encoding="utf-8") as f: f.write(silver)
with open("quality/gold_expectations.py",   "w", encoding="utf-8") as f: f.write(gold)
print("All 3 files written successfully")
import pandas as pd
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

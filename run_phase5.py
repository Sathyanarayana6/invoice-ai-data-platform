"""
run_phase5.py
--------------
Runs data quality checks across all three medallion layers.
"""

from quality.bronze_expectations import run_bronze_expectations
from quality.silver_expectations import run_silver_expectations
from quality.gold_expectations   import run_gold_expectations

if __name__ == "__main__":
    print("\n PHASE 5 — DATA QUALITY\n")

    print("STEP 1: Bronze layer checks...")
    b_pass, b_fail = run_bronze_expectations()

    print("\nSTEP 2: Silver layer checks...")
    s_pass, s_fail = run_silver_expectations()

    print("\nSTEP 3: Gold layer checks...")
    g_pass, g_fail = run_gold_expectations()

    print("\n" + "=" * 50)
    print("OVERALL DATA QUALITY SUMMARY")
    print("=" * 50)
    total_pass = b_pass + s_pass + g_pass
    total_fail = b_fail + s_fail + g_fail
    total      = total_pass + total_fail
    print(f"  Bronze : {b_pass} passed, {b_fail} failed")
    print(f"  Silver : {s_pass} passed, {s_fail} failed")
    print(f"  Gold   : {g_pass} passed, {g_fail} failed")
    print(f"  Total  : {total_pass}/{total} passed")
    print(f"\n  Reports saved to lake/quality_reports/")
    print("=" * 50)
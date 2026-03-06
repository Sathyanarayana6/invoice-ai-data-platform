"""
run_phase1.py
--------------
Runs the entire Phase 1 pipeline in order:
  1. Generate synthetic data
  2. Bronze ingestion
  3. Silver transformation
  4. Gold aggregation
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src_data.synthetic_invoice_generator import main as generate_data
from pipeline.bronze_ingestion import run_bronze_ingestion
from pipeline.silver_transform import run_silver_transform
from pipeline.gold_aggregate import run_gold_aggregation

if __name__ == "__main__":
    print("\n🚀 PHASE 1 — MEDALLION PIPELINE\n")

    print("STEP 1: Generating synthetic data...")
    generate_data()

    print("\nSTEP 2: Bronze ingestion...")
    run_bronze_ingestion()

    print("\nSTEP 3: Silver transformation...")
    run_silver_transform()

    print("\nSTEP 4: Gold aggregation...")
    run_gold_aggregation()

    print("\n✅ Phase 1 Complete! Check the lake/ folder for output files.")
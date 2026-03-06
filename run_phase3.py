"""
run_phase3.py
--------------
Runs the full Phase 3 ML pipeline:
  1. Feature engineering
  2. Model training (3 experiments logged to MLflow)
  3. Score all invoices
"""

from ml.feature_engineering  import run_feature_engineering
from ml.train_anomaly_model   import run_training
from ml.score_invoices        import run_scoring

if __name__ == "__main__":
    print("\n PHASE 3 — ML PIPELINE\n")

    print("STEP 1: Feature Engineering...")
    run_feature_engineering()

    print("\nSTEP 2: Training anomaly detection models...")
    run_training()

    print("\nSTEP 3: Scoring all invoices...")
    run_scoring()

    print("\nPhase 3 Complete!")
    print("  Run `mlflow ui` to see experiment dashboard at http://localhost:5000")
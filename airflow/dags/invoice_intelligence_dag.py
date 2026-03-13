"""
invoice_intelligence_dag.py
-----------------------------
Airflow DAG — Orchestrates the full Invoice Intelligence pipeline.

What is a DAG?
- Directed Acyclic Graph
- A flowchart of tasks with dependencies
- Airflow runs these on a schedule and retries on failure

Our DAG flow:
  generate_data
       |
  bronze_ingestion
       |
  silver_transform (+ quality check)
       |
  gold_aggregation (+ quality check)
       |
  feature_engineering
       |
  ml_scoring
       |
  rag_index_refresh

Schedule: Runs daily at midnight
"""

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

# Add project root to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

# ── Default Arguments ──────────────────────────────────────────────────────────

default_args = {
    "owner"           : "sathya",
    "depends_on_past" : False,
    "start_date"      : datetime(2026, 1, 1),
    "email_on_failure": False,
    "email_on_retry"  : False,
    "retries"         : 1,
    "retry_delay"     : timedelta(minutes=5),
}

# ── Task Functions ─────────────────────────────────────────────────────────────

def task_generate_data():
    from src_data.synthetic_invoice_generator import main
    main()

def task_bronze_ingestion():
    from pipeline.bronze_ingestion import run_bronze_ingestion
    run_bronze_ingestion()

def task_silver_transform():
    from pipeline.silver_transform import run_silver_transform
    run_silver_transform()

def task_bronze_quality():
    from quality.bronze_expectations import run_bronze_expectations
    passed, failed = run_bronze_expectations()
    if failed > 0:
        raise ValueError(f"Bronze quality check failed: {failed} checks failed")

def task_silver_quality():
    from quality.silver_expectations import run_silver_expectations
    passed, failed = run_silver_expectations()
    if failed > 0:
        raise ValueError(f"Silver quality check failed: {failed} checks failed")

def task_gold_aggregation():
    from pipeline.gold_aggregate import run_gold_aggregation
    run_gold_aggregation()

def task_gold_quality():
    from quality.gold_expectations import run_gold_expectations
    passed, failed = run_gold_expectations()
    if failed > 0:
        raise ValueError(f"Gold quality check failed: {failed} checks failed")

def task_feature_engineering():
    from ml.feature_engineering import run_feature_engineering
    run_feature_engineering()

def task_ml_scoring():
    from ml.score_invoices import run_scoring
    run_scoring()

def task_rag_refresh():
    from rag.document_chunker import run_document_chunking
    from rag.embed_and_index  import run_embedding_and_indexing
    run_document_chunking()
    run_embedding_and_indexing()

# ── DAG Definition ─────────────────────────────────────────────────────────────

with DAG(
    dag_id="invoice_intelligence_pipeline",
    default_args=default_args,
    description="End-to-end invoice AI data pipeline",
    schedule="0 0 * * *",   # Daily at midnight (cron syntax)
    catchup=False,                    # Don't backfill missed runs
    tags=["invoice", "ai", "data-engineering"],
) as dag:

    # ── Define Tasks ───────────────────────────────────────────────────────────

    t_generate = PythonOperator(
        task_id="generate_data",
        python_callable=task_generate_data,
    )

    t_bronze = PythonOperator(
        task_id="bronze_ingestion",
        python_callable=task_bronze_ingestion,
    )

    t_bronze_quality = PythonOperator(
        task_id="bronze_quality_check",
        python_callable=task_bronze_quality,
    )

    t_silver = PythonOperator(
        task_id="silver_transform",
        python_callable=task_silver_transform,
    )

    t_silver_quality = PythonOperator(
        task_id="silver_quality_check",
        python_callable=task_silver_quality,
    )

    t_gold = PythonOperator(
        task_id="gold_aggregation",
        python_callable=task_gold_aggregation,
    )

    t_gold_quality = PythonOperator(
        task_id="gold_quality_check",
        python_callable=task_gold_quality,
    )

    t_features = PythonOperator(
        task_id="feature_engineering",
        python_callable=task_feature_engineering,
    )

    t_ml = PythonOperator(
        task_id="ml_scoring",
        python_callable=task_ml_scoring,
    )

    t_rag = PythonOperator(
        task_id="rag_index_refresh",
        python_callable=task_rag_refresh,
    )

    # ── Define Dependencies ────────────────────────────────────────────────────
    # >> means "must complete before"
    # This is the execution order Airflow enforces

    t_generate >> t_bronze >> t_bronze_quality >> t_silver >> t_silver_quality >> t_gold >> t_gold_quality >> t_features >> t_ml >> t_rag
# Invoice AI Data Platform

An end-to-end AI data engineering platform that processes B2B invoice data through a full medallion architecture, real-time streaming, ML anomaly detection, and natural language querying via RAG.

Built to demonstrate production-grade data engineering skills across the full AI data stack.

---

## Architecture

```
Raw Invoices (CSV/Kafka)
        │
        ▼
┌─────────────────────────────────────────────┐
│           BRONZE LAYER (Raw)                │
│  • Parquet storage                          │
│  • Row hashing for deduplication            │
│  • Metadata tracking (_ingested_at, source) │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│           SILVER LAYER (Clean)              │
│  • Deduplication (200 duplicates removed)   │
│  • Date standardization (3 formats → 1)     │
│  • Null/negative amount handling            │
│  • Derived columns (days_overdue, risk)     │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│           GOLD LAYER (Business-Ready)       │
│  • vendor_summary (50 vendors, risk scores) │
│  • monthly_trends (25 months)               │
│  • category_analysis (7 categories)         │
└─────────────────────────────────────────────┘
        │
        ├──────────────────┬─────────────────────
        ▼                  ▼
┌──────────────┐   ┌──────────────────────────┐
│  ML Pipeline │   │      RAG Pipeline        │
│              │   │                          │
│  • Feature   │   │  • Document chunking     │
│    engineering│  │  • OpenAI embeddings     │
│  • Isolation │   │    (text-embedding-3-    │
│    Forest    │   │     small)               │
│  • MLflow    │   │  • ChromaDB vector store │
│    tracking  │   │  • GPT-4o-mini answers   │
│  • 166       │   │  • Natural language      │
│    anomalies │   │    querying              │
│    detected  │   │                          │
└──────────────┘   └──────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│         ORCHESTRATION (Airflow)             │
│  • 10-task DAG                              │
│  • Daily schedule                           │
│  • Quality gates between layers             │
│  • Auto-retry on failure                    │
└─────────────────────────────────────────────┘
```

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Data Processing | Python, Pandas, PySpark, Parquet |
| Streaming | Apache Kafka, Kafdrop |
| Storage | Medallion Architecture (Bronze/Silver/Gold) |
| ML | Scikit-learn (Isolation Forest), MLflow |
| AI/RAG | LangChain, ChromaDB, OpenAI (text-embedding-3-small, GPT-4o-mini) |
| Data Quality | Great Expectations (pandas-based checks) |
| Orchestration | Apache Airflow |
| Infrastructure | Docker, WSL2 |
| Version Control | Git, GitHub |

---

## Project Structure

```
invoice-ai-data-platform/
├── src_data/
│   └── synthetic_invoice_generator.py   # Generates 10K+ invoice records
├── pipeline/
│   ├── bronze_ingestion.py              # Raw ingestion with metadata
│   ├── silver_transform.py              # Clean, deduplicate, standardize
│   └── gold_aggregate.py                # Business-ready aggregations
├── streaming/
│   ├── docker-compose.yml               # Kafka + Zookeeper + Kafdrop
│   ├── invoice_producer.py              # Publishes invoice events to Kafka
│   └── invoice_consumer.py              # Reads from Kafka → Bronze layer
├── ml/
│   ├── feature_engineering.py           # 30 features from Silver + Gold
│   ├── train_anomaly_model.py           # Isolation Forest + MLflow tracking
│   └── score_invoices.py                # Production inference pipeline
├── rag/
│   ├── document_chunker.py              # Converts Gold data to text chunks
│   ├── embed_and_index.py               # OpenAI embeddings → ChromaDB
│   └── retrieval_chain.py               # GPT-4o-mini RAG answers
├── quality/
│   ├── bronze_expectations.py           # 12 Bronze quality checks
│   ├── silver_expectations.py           # 9 Silver quality checks
│   └── gold_expectations.py             # 10 Gold quality checks
├── airflow/
│   └── dags/
│       └── invoice_intelligence_dag.py  # 10-task orchestration DAG
├── run_phase1.py                        # Medallion pipeline runner
├── run_phase3.py                        # ML pipeline runner
├── run_phase4.py                        # RAG pipeline runner
└── run_phase5.py                        # Data quality runner
```

---

## Key Metrics

- **10,200** raw invoice records generated with intentional data quality issues
- **200** duplicate records detected and removed in Silver layer
- **3,321** clean records after Silver transformation
- **50** vendors tracked with risk scores in Gold layer
- **30** ML features engineered per invoice
- **166** anomalies detected (5% anomaly rate) via Isolation Forest
- **3** MLflow experiment runs tracked and compared
- **275** document chunks indexed into ChromaDB
- **31/31** data quality checks passing across all layers
- **10-task** Airflow DAG orchestrating the full pipeline

---

## How To Run

### Prerequisites
- Python 3.10+
- Docker Desktop
- WSL2 (for Airflow)

### Install dependencies
```bash
pip install pandas pyspark pyarrow faker boto3 mlflow scikit-learn langchain chromadb openai kafka-python python-dotenv
```

### Set up environment
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your-openai-api-key
```

### Phase 1 — Medallion Pipeline
```bash
python run_phase1.py
```

### Phase 2 — Kafka Streaming
```bash
cd streaming
docker-compose up -d
python streaming/invoice_consumer.py  # Terminal 1
python streaming/invoice_producer.py  # Terminal 2
```

### Phase 3 — ML Pipeline
```bash
python run_phase3.py
mlflow ui --backend-store-uri ./mlruns  # View experiments at localhost:5000
```

### Phase 4 — RAG Pipeline
```bash
python run_phase4.py
```

### Phase 5 — Data Quality
```bash
python run_phase5.py
```

### Phase 6 — Airflow Orchestration (WSL2)
```bash
source ~/airflow-venv/bin/activate
export AIRFLOW_HOME=~/airflow
airflow standalone
# Open http://localhost:8080
```

---

## Data Flow

```
synthetic_invoice_generator.py
  → 10,200 raw invoices (CSV)
  → Kafka producer publishes events
  → Kafka consumer writes to Bronze

Bronze (raw Parquet)
  → Silver (clean, deduplicated)
  → Gold (vendor_summary, monthly_trends, category_analysis)

Gold → Feature Engineering → Isolation Forest → MLflow Model Registry
Gold → Document Chunking → OpenAI Embeddings → ChromaDB → GPT-4o-mini

Airflow DAG orchestrates all steps daily with quality gates
```

---

## Author

**Sathyanarayana Balla**
AI & Data Engineer
- AWS Certified Cloud Practitioner
- CompTIA Network+
- M.S. Cybersecurity Operations, Webster University

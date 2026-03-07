"""
document_chunker.py
--------------------
Converts Gold layer data into text documents, then chunks them for RAG.

What is Chunking?
- LLMs have a context window limit (can't read 10,000 records at once)
- We split data into small overlapping chunks (500 tokens each)
- Each chunk becomes a searchable unit in our vector database
- Overlap (50 tokens) ensures we don't cut off important context at boundaries

What we convert to text:
1. Vendor summaries → "Vendor Koch-Decker has 42 invoices, 12% overdue rate..."
2. Invoice anomalies → "Invoice INV-001501 flagged as CRITICAL, amount $0.00..."
3. Monthly trends → "In January 2024, 145 invoices totaling $2.3M were processed..."

Why convert structured data to text?
- LLMs are trained on text — they understand language, not tables
- Natural language queries work better against natural language documents
- Business users ask "which vendors have disputes?" not SQL queries
"""

import pandas as pd
import pyarrow.parquet as pq
import os
import json
from typing import List, Dict

# ── Config ─────────────────────────────────────────────────────────────────────

GOLD_PATH    = "lake/gold"
SCORES_PATH  = "lake/gold/anomaly_scores/scored_invoices.parquet"
DOCS_PATH    = "lake/rag_documents"
CHUNK_SIZE   = 500    # tokens per chunk (approximate using characters/4)
CHUNK_OVERLAP= 50     # overlap between chunks

# ── Document Generators ────────────────────────────────────────────────────────

def generate_vendor_documents(vendor_df: pd.DataFrame) -> List[Dict]:
    """
    Converts each vendor's summary into a natural language document.
    Each vendor becomes one document.
    """
    documents = []

    for _, row in vendor_df.iterrows():
        # Convert row values safely
        vendor_name    = str(row.get("vendor_name",    "Unknown"))
        vendor_id      = str(row.get("vendor_id",      "Unknown"))
        state          = str(row.get("state",          "Unknown"))
        total_invoices = str(row.get("total_invoices", "0"))
        total_amount   = str(row.get("total_amount",   "0"))
        avg_amount     = str(row.get("avg_amount",     "0"))
        overdue_rate   = str(row.get("overdue_rate",   "0"))
        dispute_rate   = str(row.get("dispute_rate",   "0"))
        risk_score     = str(row.get("risk_score",     "0"))
        paid_count     = str(row.get("paid_count",     "0"))
        overdue_count  = str(row.get("overdue_count",  "0"))
        disputed_count = str(row.get("disputed_count", "0"))

        text = f"""
Vendor Report: {vendor_name}
Vendor ID: {vendor_id}
State: {state}

Invoice Summary:
- Total invoices: {total_invoices}
- Total invoice value: ${total_amount}
- Average invoice amount: ${avg_amount}
- Paid invoices: {paid_count}
- Overdue invoices: {overdue_count}
- Disputed invoices: {disputed_count}

Risk Analysis:
- Overdue rate: {overdue_rate}%
- Dispute rate: {dispute_rate}%
- Risk score: {risk_score}

Assessment: {"High risk vendor requiring attention." if float(risk_score) > 10 else "Low to medium risk vendor."}
        """.strip()

        documents.append({
            "doc_id"   : f"vendor_{vendor_id}",
            "doc_type" : "vendor_summary",
            "vendor_id": vendor_id,
            "text"     : text
        })

    return documents


def generate_anomaly_documents(scores_df: pd.DataFrame) -> List[Dict]:
    """
    Converts flagged anomalous invoices into text documents.
    Only includes CRITICAL and HIGH risk invoices to keep volume manageable.
    """
    documents = []

    # Filter to anomalies only
    anomalies = scores_df[
        scores_df["risk_level"].isin(["CRITICAL", "HIGH"])
    ].head(200)  # Cap at 200 for demo

    for _, row in anomalies.iterrows():
        invoice_id    = str(row.get("invoice_id",    "Unknown"))
        vendor_id     = str(row.get("vendor_id",     "Unknown"))
        amount        = str(row.get("amount",        "0"))
        risk_level    = str(row.get("risk_level",    "Unknown"))
        anomaly_score = str(row.get("anomaly_score", "0"))
        days_overdue  = str(row.get("days_overdue",  "0"))

        text = f"""
Anomaly Alert: Invoice {invoice_id}
Risk Level: {risk_level}
Vendor ID: {vendor_id}
Invoice Amount: ${amount}
Anomaly Score: {anomaly_score}
Days Overdue: {days_overdue}

Reason for flagging: This invoice was identified as anomalous by the 
Isolation Forest model. {"Zero dollar amount detected - possible data entry error or fraudulent submission." if float(amount) == 0 else "Unusual pattern detected compared to vendor historical behavior."}

Recommended action: {"Immediate review required." if risk_level == "CRITICAL" else "Review within 48 hours."}
        """.strip()

        documents.append({
            "doc_id"    : f"anomaly_{invoice_id}",
            "doc_type"  : "anomaly_alert",
            "invoice_id": invoice_id,
            "risk_level": risk_level,
            "text"      : text
        })

    return documents


def generate_trend_documents(trends_df: pd.DataFrame) -> List[Dict]:
    """
    Converts monthly trend data into natural language summaries.
    Each month becomes one document.
    """
    documents = []

    for _, row in trends_df.iterrows():
        year_month     = str(row.get("year_month",     "Unknown"))
        total_invoices = str(row.get("total_invoices", "0"))
        total_amount   = str(row.get("total_amount",   "0"))
        avg_amount     = str(row.get("avg_amount",     "0"))
        overdue_rate   = str(row.get("overdue_rate",   "0"))
        overdue_count  = str(row.get("overdue_count",  "0"))
        disputed_count = str(row.get("disputed_count", "0"))

        text = f"""
Monthly Invoice Report: {year_month}

Volume:
- Total invoices processed: {total_invoices}
- Total invoice value: ${total_amount}
- Average invoice amount: ${avg_amount}

Payment Health:
- Overdue invoices: {overdue_count} ({overdue_rate}% overdue rate)
- Disputed invoices: {disputed_count}

Trend note: {"Elevated overdue rate this month - collections team should be alerted." if float(overdue_rate) > 15 else "Overdue rate within normal range."}
        """.strip()

        documents.append({
            "doc_id"    : f"trend_{year_month}",
            "doc_type"  : "monthly_trend",
            "year_month": year_month,
            "text"      : text
        })

    return documents


# ── Chunker ────────────────────────────────────────────────────────────────────

def chunk_document(doc: Dict, chunk_size: int = CHUNK_SIZE,
                   overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    """
    Splits a document's text into overlapping chunks.

    Why overlap?
    - Imagine a sentence that spans the boundary between two chunks
    - Without overlap, the sentence gets cut in half and loses meaning
    - With overlap, both chunks contain the full sentence
    - 50 token overlap = ~200 character overlap
    """
    text   = doc["text"]
    # Approximate tokens as characters / 4 (rough GPT tokenization estimate)
    char_size    = chunk_size * 4
    char_overlap = overlap * 4

    chunks = []
    start  = 0
    idx    = 0

    while start < len(text):
        end = start + char_size
        chunk_text = text[start:end]

        chunk = {
            **doc,  # Carry all metadata from parent document
            "chunk_id"  : f"{doc['doc_id']}_chunk_{idx}",
            "chunk_text": chunk_text,
            "chunk_idx" : idx,
        }
        chunks.append(chunk)

        # Move forward by chunk_size minus overlap
        start += (char_size - char_overlap)
        idx   += 1

        # If remaining text is smaller than overlap, we're done
        if end >= len(text):
            break

    return chunks


# ── Main ───────────────────────────────────────────────────────────────────────

def run_document_chunking():
    print("=" * 50)
    print("DOCUMENT CHUNKING — Starting")
    print("=" * 50)

    print("\n[1/6] Loading Gold layer data...")
    vendor_df  = pd.read_parquet(f"{GOLD_PATH}/vendor_summary/data.parquet")
    trends_df  = pd.read_parquet(f"{GOLD_PATH}/monthly_trends/data.parquet")
    scores_df  = pd.read_parquet(SCORES_PATH)
    print(f"      Vendors: {len(vendor_df)}, Months: {len(trends_df)}, Scores: {len(scores_df)}")

    print("\n[2/6] Generating vendor documents...")
    vendor_docs = generate_vendor_documents(vendor_df)
    print(f"      Generated {len(vendor_docs)} vendor documents")

    print("\n[3/6] Generating anomaly documents...")
    anomaly_docs = generate_anomaly_documents(scores_df)
    print(f"      Generated {len(anomaly_docs)} anomaly documents")

    print("\n[4/6] Generating trend documents...")
    trend_docs = generate_trend_documents(trends_df)
    print(f"      Generated {len(trend_docs)} trend documents")

    all_docs = vendor_docs + anomaly_docs + trend_docs
    print(f"\n[5/6] Total documents: {len(all_docs)}")

    print("\n[6/6] Chunking all documents...")
    all_chunks = []
    for doc in all_docs:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
    print(f"      Total chunks: {len(all_chunks)}")

    # Save chunks to disk
    os.makedirs(DOCS_PATH, exist_ok=True)
    with open(f"{DOCS_PATH}/chunks.json", "w") as f:
        json.dump(all_chunks, f, indent=2)
    print(f"      Saved to {DOCS_PATH}/chunks.json")

    print("\n" + "=" * 50)
    print("DOCUMENT CHUNKING — Complete")
    print("=" * 50)

    return all_chunks


if __name__ == "__main__":
    run_document_chunking()
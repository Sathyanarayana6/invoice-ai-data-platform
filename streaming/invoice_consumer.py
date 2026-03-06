"""
invoice_consumer.py
--------------------
Kafka Consumer — Reads invoice events and writes to Bronze layer

What this does:
- Listens continuously to the 'invoices-raw' Kafka topic
- Every time a new invoice message arrives, it processes it
- Batches messages (every 10 messages or every 30 seconds)
- Writes each batch to Bronze layer as Parquet

Real world equivalent:
- This process runs 24/7 on a server
- The moment a distributor sends an invoice, this consumer picks it up
- Within seconds it's in Bronze, ready for Silver processing
"""

import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from kafka import KafkaConsumer
from datetime import datetime
import os
import hashlib

# ── Config ─────────────────────────────────────────────────────────────────────

KAFKA_BROKER   = "localhost:9092"
TOPIC_NAME     = "invoices-raw"
CONSUMER_GROUP = "bronze-ingestion-group"   # Group ID — Kafka tracks offset per group
BRONZE_PATH    = "lake/bronze/streaming"    # Separate from batch bronze
BATCH_SIZE     = 10                         # Write to disk every 10 messages
TIMEOUT_MS     = 30000                      # Stop waiting after 30 seconds of no messages

# ── Consumer Setup ─────────────────────────────────────────────────────────────

def create_consumer():
    """
    Creates a Kafka consumer.

    group_id: Kafka remembers where this consumer group left off.
              If the consumer restarts, it picks up where it stopped.
              This is called "offset management" — critical for reliability.

    auto_offset_reset: If this is a brand new consumer group with no history,
                       start from the 'earliest' message in the topic.

    value_deserializer: Converts incoming bytes → Python dict (reverse of producer)
    """
    consumer = KafkaConsumer(
        TOPIC_NAME,
        bootstrap_servers=KAFKA_BROKER,
        group_id=CONSUMER_GROUP,
        auto_offset_reset="earliest",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        # How long to wait for new messages before returning empty
        consumer_timeout_ms=TIMEOUT_MS,
        enable_auto_commit=True
    )
    return consumer


# ── Bronze Writer ──────────────────────────────────────────────────────────────

def add_metadata(records: list) -> pd.DataFrame:
    """
    Same metadata pattern as batch Bronze ingestion.
    Consistency between batch and streaming Bronze is important —
    downstream Silver doesn't need to know which path data came from.
    """
    df = pd.DataFrame(records)

    df["_ingested_at"] = datetime.utcnow().isoformat()
    df["_source"]      = "kafka_stream"
    df["_row_hash"]    = df.apply(
        lambda row: hashlib.md5(
            f"{row.get('invoice_id','')}{row.get('amount','')}{row.get('invoice_date','')}".encode()
        ).hexdigest(),
        axis=1
    )

    return df


def write_batch_to_bronze(records: list, batch_num: int):
    """
    Writes a batch of records to Bronze as Parquet.
    Each batch gets its own file named with timestamp.
    """
    if not records:
        return

    df    = add_metadata(records)
    ts    = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path  = f"{BRONZE_PATH}/batch_{batch_num:04d}_{ts}.parquet"

    os.makedirs(BRONZE_PATH, exist_ok=True)
    df.astype(str).to_parquet(path, index=False)

    print(f"\n  >> Wrote batch {batch_num} → {len(records)} records → {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def run_consumer():
    """
    Main consumer loop.
    Runs continuously until no messages arrive for TIMEOUT_MS milliseconds.
    """
    print("=" * 50)
    print("KAFKA CONSUMER — Starting")
    print(f"  Broker       : {KAFKA_BROKER}")
    print(f"  Topic        : {TOPIC_NAME}")
    print(f"  Consumer Group: {CONSUMER_GROUP}")
    print(f"  Batch size   : {BATCH_SIZE}")
    print(f"  Writing to   : {BRONZE_PATH}")
    print("=" * 50)
    print("\nListening for messages... (will stop after 30s of silence)\n")

    consumer    = create_consumer()
    batch       = []
    batch_num   = 0
    total_read  = 0

    try:
        for message in consumer:
            # message.value is already deserialized to dict by value_deserializer
            event   = message.value
            payload = event.get("payload", event)  # Extract invoice from envelope

            batch.append(payload)
            total_read += 1

            print(
                f"[{total_read:>4}] Received: {payload.get('invoice_id','?')} | "
                f"Vendor: {str(payload.get('vendor_name','?'))[:20]:<20} | "
                f"Offset: {message.offset}"
            )

            # Write batch to Bronze when batch size reached
            if len(batch) >= BATCH_SIZE:
                batch_num += 1
                write_batch_to_bronze(batch, batch_num)
                batch = []   # Reset batch

    except Exception as e:
        print(f"\nConsumer stopped: {e}")

    finally:
        # Write any remaining records in the last partial batch
        if batch:
            batch_num += 1
            write_batch_to_bronze(batch, batch_num)

        consumer.close()

    print("\n" + "=" * 50)
    print(f"CONSUMER COMPLETE")
    print(f"  Total messages read : {total_read}")
    print(f"  Total batches written: {batch_num}")
    print(f"  Bronze path         : {BRONZE_PATH}")
    print("=" * 50)


if __name__ == "__main__":
    run_consumer()
"""
invoice_producer.py
--------------------
Kafka Producer — Simulates real-time invoice arrivals

What this does:
- Reads invoices from our CSV (simulates an external system sending invoices)
- Publishes each invoice as a JSON message to the Kafka topic 'invoices-raw'
- Sends one invoice every 0.5 seconds to simulate a live stream
- Prints confirmation for each message sent

Real world equivalent:
- A distributor's system sends an invoice event when they ship an order
- That event hits Kafka immediately
- Our consumer picks it up and writes it to Bronze
"""

import json
import time
import pandas as pd
from kafka import KafkaProducer
from datetime import datetime
import random

# ── Config ─────────────────────────────────────────────────────────────────────

KAFKA_BROKER = "localhost:9092"
TOPIC_NAME   = "invoices-raw"
SOURCE_FILE  = "src_data/raw_invoices.csv"
DELAY        = 0.5   # seconds between messages — adjust to go faster/slower

# ── Producer Setup ─────────────────────────────────────────────────────────────

def create_producer():
    """
    Creates a Kafka producer.
    
    value_serializer: Converts Python dict → JSON bytes
    Every message Kafka sends is raw bytes — serialization handles the conversion
    """
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
        # Retry up to 3 times if broker is temporarily unavailable
        retries=3,
        # Wait up to 1 second to batch small messages together (efficiency)
        linger_ms=100
    )
    return producer


# ── Message Builder ────────────────────────────────────────────────────────────

def build_message(row: dict) -> dict:
    """
    Wraps an invoice record in an event envelope.
    
    In real systems, messages carry metadata beyond just the data:
    - event_type: what kind of event is this?
    - event_timestamp: when did this event occur?
    - source_system: which system sent it?
    
    This pattern is called an "event envelope" — standard in event-driven systems.
    """
    return {
        "event_type"      : "INVOICE_RECEIVED",
        "event_timestamp" : datetime.utcnow().isoformat(),
        "source_system"   : "invoice_feed_v1",
        "payload"         : row   # The actual invoice data
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def run_producer(max_messages=100):
    """
    Reads invoices from CSV and publishes them to Kafka one by one.
    
    max_messages: How many invoices to send (default 100 for demo)
                  Set to None to send all 10,200
    """
    print("=" * 50)
    print("KAFKA PRODUCER — Starting")
    print(f"  Broker : {KAFKA_BROKER}")
    print(f"  Topic  : {TOPIC_NAME}")
    print(f"  Delay  : {DELAY}s between messages")
    print("=" * 50)

    # Load invoices
    df = pd.read_csv(SOURCE_FILE)
    if max_messages:
        df = df.head(max_messages)

    print(f"\nConnecting to Kafka broker at {KAFKA_BROKER}...")
    producer = create_producer()
    print("Connected.\n")

    sent_count = 0
    error_count = 0

    for _, row in df.iterrows():
        try:
            # Convert row to dict and build event envelope
            message = build_message(row.to_dict())

            # Send to Kafka topic
            # key = vendor_id ensures all invoices from same vendor
            # go to the same partition (ordering guarantee per vendor)
            future = producer.send(
                TOPIC_NAME,
                key=row["vendor_id"].encode("utf-8"),
                value=message
            )

            # Block until the message is confirmed sent
            record_metadata = future.get(timeout=10)

            sent_count += 1
            print(
                f"[{sent_count:>4}] Sent invoice {row['invoice_id']} | "
                f"Vendor: {row['vendor_name'][:20]:<20} | "
                f"Amount: ${row['amount'] if row['amount'] else 0:>10} | "
                f"Partition: {record_metadata.partition} | "
                f"Offset: {record_metadata.offset}"
            )

            time.sleep(DELAY)

        except Exception as e:
            error_count += 1
            print(f"ERROR sending {row['invoice_id']}: {e}")

    # Flush ensures all buffered messages are sent before we exit
    producer.flush()
    producer.close()

    print("\n" + "=" * 50)
    print(f"PRODUCER COMPLETE")
    print(f"  Sent   : {sent_count}")
    print(f"  Errors : {error_count}")
    print("=" * 50)


if __name__ == "__main__":
    run_producer(max_messages=100)
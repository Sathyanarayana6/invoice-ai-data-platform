"""
synthetic_invoice_generator.py
--------------------------------
Generates realistic fake invoice data using the Faker library.
This simulates what a real B2B invoice processing company like Fintech would receive daily.

What this produces:
- 10,000 invoice records
- Intentionally includes dirty data (nulls, duplicates, bad formats)
  because real data is never clean
"""

import pandas as pd
import numpy as np
from faker import Faker
import random
import os

# Faker gives us realistic fake names, companies, dates etc.
fake = Faker()

# Set a seed so we get the same data every time we run
# Important for reproducibility — a core data engineering principle
random.seed(42)
np.random.seed(42)
Faker.seed(42)

# ── Constants ──────────────────────────────────────────────────────────────────

NUM_RECORDS     = 10_000   # Total invoices to generate
NUM_VENDORS     = 50       # Number of unique vendors
DUPLICATE_RATE  = 0.02     # 2% duplicate invoices (real-world problem)
NULL_RATE       = 0.03     # 3% null values (real-world problem)

# Invoice statuses a B2B invoice can have
STATUSES = ["PAID", "PENDING", "OVERDUE", "DISPUTED", "CANCELLED"]
STATUS_WEIGHTS = [0.55, 0.25, 0.12, 0.05, 0.03]  # Realistic distribution

# Product categories for invoices
CATEGORIES = [
    "Alcohol", "Food & Beverage", "Cleaning Supplies",
    "Office Equipment", "Logistics", "Raw Materials", "Software Services"
]

# ── Generate Vendors ───────────────────────────────────────────────────────────

def generate_vendors(n=NUM_VENDORS):
    """
    Creates a fixed list of vendor companies.
    We keep vendors consistent so we can track them across invoices.
    """
    vendors = []
    for i in range(n):
        vendors.append({
            "vendor_id"  : f"VND-{str(i+1).zfill(4)}",   # VND-0001, VND-0002...
            "vendor_name": fake.company(),
            "vendor_state": fake.state_abbr()
        })
    return pd.DataFrame(vendors)

# ── Generate Invoices ──────────────────────────────────────────────────────────

def generate_invoices(vendors_df, n=NUM_RECORDS):
    """
    Creates invoice records with intentional data quality issues:
    - Some null amounts (NULL_RATE)
    - Some duplicate invoice IDs (DUPLICATE_RATE)
    - Some inconsistent date formats
    - Some negative amounts (data entry errors)

    This is realistic — real pipelines must handle all of this.
    """
    invoices = []

    for i in range(n):
        vendor = vendors_df.sample(1).iloc[0]  # Pick a random vendor

        # Generate invoice date in the past 2 years
        invoice_date = fake.date_between(start_date="-2y", end_date="today")

        # Due date is 30 days after invoice date
        due_date = invoice_date + pd.Timedelta(days=30)

        # Base invoice amount — realistic B2B range
        amount = round(random.uniform(100, 50000), 2)

        # ── Introduce intentional data quality issues ──

        # 3% chance of null amount (missing data)
        if random.random() < NULL_RATE:
            amount = None

        # 1% chance of negative amount (data entry error)
        if random.random() < 0.01 and amount is not None:
            amount = -amount

        # Randomly assign inconsistent date formats (real-world problem)
        date_format = random.choice(["%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y"])
        invoice_date_str = invoice_date.strftime(date_format)

        invoices.append({
            "invoice_id"    : f"INV-{str(i+1).zfill(6)}",
            "vendor_id"     : vendor["vendor_id"],
            "vendor_name"   : vendor["vendor_name"],   # Will have inconsistencies
            "invoice_date"  : invoice_date_str,        # Inconsistent formats!
            "due_date"      : due_date.strftime("%Y-%m-%d"),
            "amount"        : amount,                  # Some nulls, some negative
            "status"        : random.choices(STATUSES, STATUS_WEIGHTS)[0],
            "category"      : random.choice(CATEGORIES),
            "state"         : vendor["vendor_state"],
            "notes"         : fake.sentence() if random.random() > 0.3 else None,
            "created_at"    : fake.date_time_between(
                                start_date=invoice_date,
                                end_date=invoice_date + pd.Timedelta(days=1)
                              ).isoformat()
        })

    df = pd.DataFrame(invoices)

    # ── Add duplicate rows ─────────────────────────────────────────────────────
    # Take 2% of records and duplicate them (simulates double-submission)
    n_duplicates = int(n * DUPLICATE_RATE)
    duplicates   = df.sample(n_duplicates)
    df = pd.concat([df, duplicates], ignore_index=True)

    # Shuffle so duplicates aren't all at the end
    df = df.sample(frac=1).reset_index(drop=True)

    return df

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Generating vendor data...")
    vendors_df = generate_vendors()
    print(f"  Created {len(vendors_df)} vendors")

    print("Generating invoice data...")
    invoices_df = generate_invoices(vendors_df)
    print(f"  Created {len(invoices_df)} invoice records")
    print(f"  Intentional nulls in amount: {invoices_df['amount'].isna().sum()}")
    print(f"  Intentional duplicates added: {int(NUM_RECORDS * DUPLICATE_RATE)}")

    # Save to CSV in the data folder
    os.makedirs("src_data", exist_ok=True)
    invoices_df.to_csv("src_data/raw_invoices.csv", index=False)
    vendors_df.to_csv("src_data/vendors.csv", index=False)

    print("\nFiles saved:")
    print("  src_data/raw_invoices.csv")
    print("  src_data/vendors.csv")
    print(f"\nSample record:")
    print(invoices_df.iloc[0].to_dict())

if __name__ == "__main__":
    main()
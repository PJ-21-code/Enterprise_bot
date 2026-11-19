"""
Synthetic Enterprise Finance Database Generator
------------------------------------------------
Generates a realistic 1,000-transaction financial database in SQLite
across 50 customer accounts with:
  - Realistic multi-category transactions (debit, credit, transfer)
  - Controlled ground-truth split-transaction clusters (Positive Cases)
  - Controlled near-miss negative test cases (Negative Controls)
  - Log-normal amount distribution conforming to enterprise financial patterns

Outputs:
  - Database: benchmarks/synthetic_finance_1000.db
  - Ground Truth Labels: benchmarks/ground_truth_splits.json
"""

import os
import sqlite3
import random
import json
import numpy as np
from datetime import datetime, timedelta

DB_PATH = "./benchmarks/synthetic_finance_1000.db"
GT_PATH = "./benchmarks/ground_truth_splits.json"
TOTAL_RECORDS = 1000
NUM_ACCOUNTS = 50
START_CUSTOMER_ID = 101

# Set reproducible random seeds
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

START_DATE = datetime(2026, 1, 1, 8, 0, 0)
END_DATE = datetime(2026, 6, 30, 18, 0, 0)
TOTAL_SPAN_SECONDS = int((END_DATE - START_DATE).total_seconds())

EXPENSE_DESCRIPTIONS = [
    ("debit", "Office supplies and stationery purchase"),
    ("debit", "Cloud infrastructure hosting AWS/GCP"),
    ("debit", "Enterprise software SaaS subscription"),
    ("debit", "Corporate travel flight and lodging"),
    ("debit", "Executive legal consultation retainer"),
    ("debit", "Client dinner entertainment expense"),
    ("debit", "Hardware peripheral maintenance upgrade"),
    ("debit", "Courier and expedited logistics fees"),
    ("debit", "Facility maintenance and utilities billing"),
    ("debit", "Marketing campaign digital advertising spend"),
    ("credit", "Client advisory service retainer payment"),
    ("credit", "Enterprise software license renewal fee"),
    ("credit", "Quarterly consulting milestone receivable"),
    ("credit", "Commercial bank interest deposit"),
    ("credit", "Vendor rebate and tax credit refund"),
    ("transfer", "Vendor contract milestone disbursement"),
    ("transfer", "Engineering contractor wire payment"),
    ("transfer", "Intercompany entity operational transfer"),
    ("transfer", "Procurement equipment capital lease payout"),
    ("transfer", "Payroll auxiliary staff distribution wire")
]


def generate_database():
    print(f"Generating synthetic enterprise database: {DB_PATH}")
    os.makedirs("./benchmarks", exist_ok=True)

    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS transactions (
        transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,
        customer_id INTEGER NOT NULL,
        amount REAL NOT NULL,
        type TEXT NOT NULL,
        description TEXT NOT NULL
    )
    """)

    all_transactions = []
    ground_truth_clusters = []

    # -------------------------------------------------------------
    # 1. PLANT 12 TRUE POSITIVE SPLIT-TRANSACTION CLUSTERS
    # Threshold = $10,000; Range = [$8,000, $10,000]; Window <= 36h
    # -------------------------------------------------------------
    positive_cluster_configs = [
        {"cluster_id": "C01", "customer_id": 105, "hours_span": 14, "amounts": [8500.00, 9200.00], "desc": "Contractor software payment tranche"},
        {"cluster_id": "C02", "customer_id": 112, "hours_span": 28, "amounts": [8100.00, 8900.00, 9500.00], "desc": "Facility renovation progress wire"},
        {"cluster_id": "C03", "customer_id": 118, "hours_span": 6,  "amounts": [9950.00, 9800.00], "desc": "Hardware server procurement sub-split"},
        {"cluster_id": "C04", "customer_id": 123, "hours_span": 20, "amounts": [8250.00, 8400.00], "desc": "Consulting advisory installment"},
        {"cluster_id": "C05", "customer_id": 130, "hours_span": 32, "amounts": [8600.00, 8800.00, 9100.00, 8450.00], "desc": "Marketing campaign vendor settlement"},
        {"cluster_id": "C06", "customer_id": 134, "hours_span": 18, "amounts": [8050.00, 9150.00], "desc": "Logistics fleet maintenance disbursement"},
        {"cluster_id": "C07", "customer_id": 139, "hours_span": 8,  "amounts": [8900.00, 8750.00], "desc": "Legal compliance documentation fee"},
        {"cluster_id": "C08", "customer_id": 141, "hours_span": 22, "amounts": [8300.00, 9400.00, 8200.00], "desc": "Security architecture audit payout"},
        {"cluster_id": "C09", "customer_id": 144, "hours_span": 30, "amounts": [9600.00, 9750.00], "desc": "Data center rack expansion fee"},
        {"cluster_id": "C10", "customer_id": 147, "hours_span": 12, "amounts": [8400.00, 8950.00], "desc": "Enterprise cloud integration tranche"},
        {"cluster_id": "C11", "customer_id": 149, "hours_span": 24, "amounts": [8150.00, 9350.00], "desc": "Vendor software licensing partition"},
        {"cluster_id": "C12", "customer_id": 150, "hours_span": 16, "amounts": [9000.00, 8700.00, 9250.00], "desc": "Executive search recruiting installment"}
    ]

    base_cluster_dates = [
        START_DATE + timedelta(days=i * 14 + 3, hours=random.randint(8, 12))
        for i in range(len(positive_cluster_configs))
    ]

    for idx, cfg in enumerate(positive_cluster_configs):
        c_id = cfg["cluster_id"]
        cust = cfg["customer_id"]
        c_start = base_cluster_dates[idx]
        amounts = cfg["amounts"]
        k = len(amounts)
        span_hrs = cfg["hours_span"]

        cluster_timestamps = []
        for a_idx, amt in enumerate(amounts):
            offset_hrs = (span_hrs * a_idx) / (k - 1) if k > 1 else 0
            txn_time = c_start + timedelta(hours=offset_hrs, minutes=random.randint(0, 45))
            time_str = txn_time.strftime("%Y-%m-%d %H:%M:%S")
            cluster_timestamps.append(time_str)

            all_transactions.append({
                "date": time_str,
                "customer_id": cust,
                "amount": float(amt),
                "type": "transfer",
                "description": f"{cfg['desc']} (Tranche {a_idx+1}/{k})",
                "is_planted_cluster": True,
                "cluster_id": c_id
            })

        ground_truth_clusters.append({
            "cluster_id": c_id,
            "customer_id": cust,
            "transaction_count": len(amounts),
            "amounts": amounts,
            "total_amount": sum(amounts),
            "start_time": cluster_timestamps[0],
            "end_time": cluster_timestamps[-1],
            "time_span_hours": round(span_hrs, 2),
            "requires_review": True
        })

    # -------------------------------------------------------------
    # 2. PLANT CONTROLLED NEGATIVE TEST CASES (NEAR-MISSES)
    # These must NOT be flagged as split-clusters
    # -------------------------------------------------------------
    # Negative Control 1: Single isolated transaction in [$8k, $10k]
    all_transactions.append({
        "date": (START_DATE + timedelta(days=20, hours=10)).strftime("%Y-%m-%d %H:%M:%S"),
        "customer_id": 108,
        "amount": 9150.00,
        "type": "transfer",
        "description": "Legitimate single equipment purchase (Isolated in range)",
        "is_planted_cluster": False,
        "cluster_id": None
    })

    # Negative Control 2: Two transactions in range but spaced 54 hours apart (> 36 hours)
    neg2_time1 = START_DATE + timedelta(days=35, hours=9)
    neg2_time2 = neg2_time1 + timedelta(hours=54)
    all_transactions.append({
        "date": neg2_time1.strftime("%Y-%m-%d %H:%M:%S"),
        "customer_id": 115,
        "amount": 8500.00,
        "type": "transfer",
        "description": "Bi-weekly vendor payout (Time span > 36h)",
        "is_planted_cluster": False,
        "cluster_id": None
    })
    all_transactions.append({
        "date": neg2_time2.strftime("%Y-%m-%d %H:%M:%S"),
        "customer_id": 115,
        "amount": 8800.00,
        "type": "transfer",
        "description": "Bi-weekly vendor payout (Time span > 36h)",
        "is_planted_cluster": False,
        "cluster_id": None
    })

    # Negative Control 3: Two transactions within 8h, but one is < $8,000 ($7,200)
    neg3_time1 = START_DATE + timedelta(days=50, hours=11)
    neg3_time2 = neg3_time1 + timedelta(hours=8)
    all_transactions.append({
        "date": neg3_time1.strftime("%Y-%m-%d %H:%M:%S"),
        "customer_id": 120,
        "amount": 7200.00,  # Below 80% threshold
        "type": "debit",
        "description": "Corporate travel group booking (Below candidate range)",
        "is_planted_cluster": False,
        "cluster_id": None
    })
    all_transactions.append({
        "date": neg3_time2.strftime("%Y-%m-%d %H:%M:%S"),
        "customer_id": 120,
        "amount": 8500.00,  # In range
        "type": "debit",
        "description": "Corporate conference booking (Single in candidate range)",
        "is_planted_cluster": False,
        "cluster_id": None
    })

    # Negative Control 4: Two transactions within 10h, but one exceeds threshold ($10,500)
    neg4_time1 = START_DATE + timedelta(days=65, hours=14)
    neg4_time2 = neg4_time1 + timedelta(hours=10)
    all_transactions.append({
        "date": neg4_time1.strftime("%Y-%m-%d %H:%M:%S"),
        "customer_id": 125,
        "amount": 10500.00,  # Above $10k threshold
        "type": "transfer",
        "description": "Enterprise software license annual (Above candidate range)",
        "is_planted_cluster": False,
        "cluster_id": None
    })
    all_transactions.append({
        "date": neg4_time2.strftime("%Y-%m-%d %H:%M:%S"),
        "customer_id": 125,
        "amount": 9200.00,   # In range
        "type": "transfer",
        "description": "Enterprise support addon (Single in candidate range)",
        "is_planted_cluster": False,
        "cluster_id": None
    })

    # -------------------------------------------------------------
    # 3. GENERATE REMAINING NORMAL TRANSACTIONS UP TO 1,000
    # Log-normal distribution spanning $15.00 to $45,000.00
    # -------------------------------------------------------------
    remaining_count = TOTAL_RECORDS - len(all_transactions)
    print(f"Generating {remaining_count} normal background transactions...")

    # Log-normal parameters: median ~$600, mean ~$1,500
    log_amounts = np.random.lognormal(mean=6.4, sigma=1.2, size=remaining_count * 2)

    valid_normal_amounts = []
    for a in log_amounts:
        amt = round(float(a), 2)
        # Explicitly avoid candidate window [8000.00, 10000.00] to prevent accidental clusters
        if 15.00 <= amt < 7900.00 or 10100.00 <= amt <= 50000.00:
            valid_normal_amounts.append(amt)
            if len(valid_normal_amounts) == remaining_count:
                break

    for i in range(remaining_count):
        amt = valid_normal_amounts[i]
        t_sec = random.randint(0, TOTAL_SPAN_SECONDS)
        t_dt = START_DATE + timedelta(seconds=t_sec)
        cust = random.randint(START_CUSTOMER_ID, START_CUSTOMER_ID + NUM_ACCOUNTS - 1)
        t_type, t_desc = random.choice(EXPENSE_DESCRIPTIONS)

        all_transactions.append({
            "date": t_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "customer_id": cust,
            "amount": amt,
            "type": t_type,
            "description": t_desc,
            "is_planted_cluster": False,
            "cluster_id": None
        })

    # Sort all transactions chronologically
    all_transactions.sort(key=lambda x: x["date"])

    # Insert into SQLite
    db_rows = [
        (t["date"], t["customer_id"], t["amount"], t["type"], t["description"])
        for t in all_transactions
    ]

    cursor.executemany("""
    INSERT INTO transactions (date, customer_id, amount, type, description)
    VALUES (?, ?, ?, ?, ?)
    """, db_rows)

    conn.commit()
    conn.close()

    # Save Ground Truth JSON
    gt_payload = {
        "metadata": {
            "dataset_name": "synthetic_finance_1000",
            "total_transactions": len(all_transactions),
            "total_accounts": NUM_ACCOUNTS,
            "account_range": [START_CUSTOMER_ID, START_CUSTOMER_ID + NUM_ACCOUNTS - 1],
            "approval_threshold": 10000.0,
            "candidate_range": [8000.0, 10000.0],
            "window_hours": 36,
            "total_ground_truth_clusters": len(ground_truth_clusters)
        },
        "ground_truth_clusters": ground_truth_clusters
    }

    with open(GT_PATH, "w", encoding="utf-8") as f:
        json.dump(gt_payload, f, indent=2)

    print(f"Successfully generated {len(all_transactions)} transactions in {DB_PATH}")
    print(f"Ground-truth labeled clusters saved to {GT_PATH}")


if __name__ == "__main__":
    generate_database()

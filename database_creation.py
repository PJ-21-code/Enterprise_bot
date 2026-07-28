import sqlite3

# 1. Connect to SQLite database (creates 'enterprise_finance.db' locally)
conn = sqlite3.connect("enterprise_finance.db")
cursor = conn.cursor()

# 2. Create a realistic transactions table matching your fraud tools
cursor.execute("""
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT,
    customer_id INTEGER,
    amount REAL,
    type TEXT,
    description TEXT
)
""")

# 3. Insert 20 sample mock transaction data items
sample_data = [
    ("2026-01-01", 101, 450.50, "debit", "Office supplies purchase"),
    ("2026-01-02", 102, 12000.00, "transfer", "Vendor payout wire"),
    ("2026-01-03", 101, 89.99, "debit", "Software subscription"),
    ("2026-01-04", 103, 5400.00, "credit", "Client invoice deposit"),
    ("2026-01-05", 104, 120.00, "debit", "Travel expenses"),
    ("2026-01-06", 102, 999.99, "transfer", "Equipment purchase"),
    ("2026-01-07", 105, 3400.25, "credit", "Consulting service fee"),
    ("2026-01-08", 101, 1500.00, "debit", "Marketing campaign ad spend"),
    ("2026-01-09", 106, 499.00, "debit", "Cloud hosting server fees"),
    ("2026-01-10", 102, 12000.00, "transfer", "Vendor payout wire (duplicate amount)"),
    ("2026-01-11", 103, 230.40, "debit", "Client dinner entertainment"),
    ("2026-01-12", 107, 15000.50, "credit", "Enterprise software license"),
    ("2026-01-13", 104, 75.00, "debit", "Courier delivery charge"),
    ("2026-01-14", 105, 4500.00, "transfer", "Contractor payment milestone 1"),
    ("2026-01-15", 105, 4500.00, "transfer", "Contractor payment milestone 2"),
    ("2026-01-16", 108, 310.00, "debit", "Office utility bills"),
    ("2026-01-17", 101, 899.50, "debit", "Hardware peripheral upgrade"),
    ("2026-01-18", 106, 12500.00, "credit", "Annual support renewal"),
    ("2026-01-19", 103, 620.00, "debit", "Legal consultation retainer"),
    ("2026-01-20", 102, 550.00, "debit", "Team building event catering"),
]

cursor.executemany(
    """
INSERT INTO transactions (date, customer_id, amount, type, description)
VALUES (?, ?, ?, ?, ?)
""",
    sample_data,
)

# Commit changes and close connection
conn.commit()
conn.close()

print(
    "Successfully created 'enterprise_finance.db' with 20 sample transaction"
    " records!"
)
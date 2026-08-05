"""
db.py - load the five synthetic CSVs into a single SQLite database (telco.db)
and add indexes so JOINs are fast.

This mirrors the db.py from the llm-sql-eval project: CSV -> pandas -> to_sql,
then CREATE INDEX on the foreign keys, then sanity-check row counts.

Run (after generating the data):
  python db.py
"""

import os
import sqlite3
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
DB_PATH = os.path.join(HERE, "telco.db")

TABLES = {
    "plans":     "plans.csv",
    "customers": "customers.csv",
    "usage":     "usage.csv",
    "bills":     "bills.csv",
    "tickets":   "tickets.csv",
}


def build():
    conn = sqlite3.connect(DB_PATH)

    # ---- 1. Load each CSV into its own table (replace on re-run) ----
    for table, csv_name in TABLES.items():
        df = pd.read_csv(os.path.join(DATA_DIR, csv_name))
        df.to_sql(table, conn, if_exists="replace", index=False)
        print(f"  loaded {table:10s} <- {csv_name:15s} ({len(df)} rows)")

    # ---- 2. Indexes on the foreign keys (customer_id, plan_id) ----
    cur = conn.cursor()
    cur.executescript("""
        CREATE INDEX IF NOT EXISTS idx_customers_plan   ON customers(plan_id);
        CREATE INDEX IF NOT EXISTS idx_usage_customer   ON usage(customer_id);
        CREATE INDEX IF NOT EXISTS idx_bills_customer   ON bills(customer_id);
        CREATE INDEX IF NOT EXISTS idx_tickets_customer ON tickets(customer_id);
    """)
    conn.commit()

    # ---- 3. Sanity checks: referential integrity + a couple of aggregates ----
    print("\n--- Verification ---")
    for table in TABLES:
        n = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table:10s} rows: {n}")

    orphan_usage = cur.execute("""
        SELECT COUNT(*) FROM usage u
        LEFT JOIN customers c ON u.customer_id = c.customer_id
        WHERE c.customer_id IS NULL
    """).fetchone()[0]
    print(f"\n  orphan usage rows (should be 0): {orphan_usage}")

    active = cur.execute("SELECT COUNT(*) FROM customers WHERE status='active'").fetchone()[0]
    churned = cur.execute("SELECT COUNT(*) FROM customers WHERE status='churned'").fetchone()[0]
    print(f"  customers: {active} active / {churned} churned")

    revenue = cur.execute("SELECT ROUND(SUM(amount), 2) FROM bills WHERE status='paid'").fetchone()[0]
    print(f"  total paid revenue (2024): EUR {revenue}")

    conn.close()
    print(f"\nDone. Wrote {DB_PATH}")


if __name__ == "__main__":
    build()

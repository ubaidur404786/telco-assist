"""
schema_extractor.py - read telco.db and write the exact CREATE TABLE statements
to schema.sql. That schema text is what we paste into the LLM prompt so the model
knows the tables/columns it can query.

It also prints a column inventory and 3 sample rows per table so you can eyeball
real values while writing golden questions.

Run (after building the DB):
  python schema_extractor.py
"""

import os
import sqlite3
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "telco.db")
SCHEMA_OUT = os.path.join(HERE, "schema.sql")


def user_tables(cur):
    return [r[0] for r in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()]


def save_schema(conn):
    cur = conn.cursor()
    with open(SCHEMA_OUT, "w", encoding="utf-8") as f:
        f.write("-- Schema for HexaMobile text-to-SQL generation prompt\n")
        f.write(f"-- Source DB: {os.path.basename(DB_PATH)}\n\n")
        for t in user_tables(cur):
            ddl = cur.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (t,)
            ).fetchone()[0]
            f.write(ddl.strip() + ";\n\n")
    print(f"Saved schema -> {SCHEMA_OUT}")


def show_schema(conn):
    cur = conn.cursor()
    for t in user_tables(cur):
        cols = cur.execute(f"PRAGMA table_info({t})").fetchall()
        n = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"\n--- {t}  ({n} rows, {len(cols)} cols) ---")
        for c in cols:
            print(f"  {c[1]:22s} {c[2] or 'TEXT'}")
        sample = pd.read_sql(f"SELECT * FROM {t} LIMIT 3", conn)
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(sample.to_string(index=False))


if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    show_schema(conn)
    save_schema(conn)
    conn.close()

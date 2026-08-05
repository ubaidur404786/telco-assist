"""
sql_make_golden.py - freeze the gold answers.

Runs every gold query from sql_golden.py against telco.db and records its EXACT
result (columns + rows) as JSON in sql_golden.csv. The eval later compares each
model's output to this frozen result, so we only pay the cost of running the gold
queries once.

Run:
  python sql_make_golden.py     ->  sql_golden.csv
"""

import os
import csv
import json
import sqlite3

from sql_golden import GOLDEN, ORDER_SENSITIVE_IDS

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "telco.db")
OUT_PATH = os.path.join(HERE, "sql_golden.csv")


def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    rows_out = []
    for g in GOLDEN:
        cur.execute(g["sql"])
        cols = [d[0] for d in cur.description]
        data = cur.fetchall()
        rows_out.append({
            "id": g["id"],
            "difficulty": g["diff"],
            "question": g["q"],
            "gold_sql": " ".join(g["sql"].split()),   # collapse whitespace to one line
            "gold_result_json": json.dumps(
                {"columns": cols, "rows": [list(r) for r in data]}, ensure_ascii=False
            ),
            "n_rows": len(data),
            "order_sensitive": str(g["id"] in ORDER_SENSITIVE_IDS).upper(),
        })
    conn.close()

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "id", "difficulty", "question", "gold_sql",
            "gold_result_json", "n_rows", "order_sensitive",
        ])
        w.writeheader()
        w.writerows(rows_out)

    print(f"Wrote {OUT_PATH} with {len(rows_out)} rows")


if __name__ == "__main__":
    main()

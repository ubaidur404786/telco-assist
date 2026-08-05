"""
context.py - build the CONTEXT block we feed the model before each question.

A schema (CREATE TABLE statements) tells the model which columns exist, but NOT what
values live in them. Without help, a model cannot know that region is spelled
'Ile-de-France', or that ticket status is one of open/resolved/escalated - so it
guesses, and value-dependent questions fail for the wrong reason.

Real text-to-SQL systems fix this by adding "value hints": the distinct values of
low-cardinality categorical columns. This module builds:

    schema.sql  +  value hints (auto-queried from telco.db)

Import build_context() wherever you need the prompt context.
"""

import os
import sqlite3

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "telco.db")
SCHEMA_PATH = os.path.join(HERE, "schema.sql")

# Columns worth enumerating. High-cardinality columns (names, cities, ids) are
# intentionally excluded - listing 2000 names would bloat the prompt and help nothing.
CATEGORICAL_COLUMNS = [
    ("customers", "status"),
    ("customers", "region"),
    ("plans", "plan_name"),
    ("plans", "contract_type"),
    ("bills", "status"),
    ("tickets", "category"),
    ("tickets", "channel"),
    ("tickets", "status"),
    ("usage", "month"),
]


def load_schema():
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        return f.read().strip()


def value_hints(max_distinct=30):
    """Return a text block of 'table.column: v1, v2, ...' for each categorical column
    whose distinct-value count is small enough to be useful."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    lines = []
    for table, col in CATEGORICAL_COLUMNS:
        vals = [r[0] for r in cur.execute(
            f"SELECT DISTINCT {col} FROM {table} WHERE {col} IS NOT NULL ORDER BY {col}"
        ).fetchall()]
        if 0 < len(vals) <= max_distinct:
            shown = ", ".join(str(v) for v in vals)
            lines.append(f"{table}.{col}: {shown}")
    conn.close()
    return "\n".join(lines)


def build_context():
    """The full context string: schema + value hints."""
    return (
        f"{load_schema()}\n\n"
        f"-- Distinct values for categorical columns (use these EXACT spellings):\n"
        f"{value_hints()}"
    )


if __name__ == "__main__":
    print(build_context())

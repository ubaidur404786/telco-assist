"""
sql_evaluator.py - the metric. EXECUTION ACCURACY, value-only comparison.

Ported from the llm-sql-eval (IPL) project. We do NOT compare the SQL text - two
very different queries can be equally correct. Instead we run both and compare the
RESULT SETS:

  - column NAMES are ignored (SUM(x) AS total vs AS revenue -> same)
  - numbers compared numerically (25 == 25.0, tiny float noise tolerated)
  - row order ignored by default; order_sensitive=True preserves it
  - a model may return ONE extra (or fewer) column than gold - e.g. "which region?"
    answered as region-only OR region+count - and still be counted correct (Path A).
"""

import pandas as pd


def _canon_cell(v):
    """Normalise a single cell so 25, 25.0 and '25' all compare equal."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "None"
    if isinstance(v, bool):
        return f"str:{v}"
    if isinstance(v, (int, float)):
        f = float(v)
        return f"num:{int(f)}" if f.is_integer() else f"num:{round(f, 4)}"
    s = str(v).strip()
    try:
        f = float(s)
        return f"num:{int(f)}" if f.is_integer() else f"num:{round(f, 4)}"
    except ValueError:
        return f"str:{s}"


def _rows_as_sets(df):
    """Each row -> a frozenset of canonical values (column-position agnostic)."""
    return [frozenset(_canon_cell(v) for v in row) for row in df.values.tolist()]


def _rows_as_tuples(df, order_sensitive):
    rows = [tuple(_canon_cell(v) for v in row) for row in df.values.tolist()]
    return rows if order_sensitive else sorted(rows)


def compare_dataframes(gold_df, gen_df, order_sensitive=False):
    if len(gold_df) != len(gen_df):
        return False

    gcols, xcols = gold_df.shape[1], gen_df.shape[1]

    # Same column count -> strict value comparison.
    if gcols == xcols:
        return _rows_as_tuples(gold_df, order_sensitive) == _rows_as_tuples(gen_df, order_sensitive)

    # Differ by exactly one column -> lenient subset check (Path A).
    if abs(gcols - xcols) != 1:
        return False

    small, large = (gold_df, gen_df) if gcols < xcols else (gen_df, gold_df)
    small_rows, large_rows = _rows_as_sets(small), _rows_as_sets(large)
    if not order_sensitive:
        small_rows = sorted(small_rows, key=lambda s: sorted(s))
        large_rows = sorted(large_rows, key=lambda s: sorted(s))
    return all(s.issubset(l) for s, l in zip(small_rows, large_rows))


def evaluate_one(gold_df, gen_df, order_sensitive=False):
    if gen_df is None:
        return {"correct": False, "reason": "sql_error"}
    try:
        ok = compare_dataframes(gold_df, gen_df, order_sensitive)
        return {"correct": ok, "reason": "match" if ok else "mismatch"}
    except Exception as e:
        return {"correct": False, "reason": f"compare_error: {e}"}

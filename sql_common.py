"""
sql_common.py - text-to-SQL helpers shared by the eval (sql_eval.py) and the live
SQL agent (sql_agent.py). Keeping them in one place means the agent runs the EXACT
prompt + cleaning that we evaluated, so eval scores actually predict live behaviour.
"""

import re
import pandas as pd

# The generation instruction (same one the eval used).
SYSTEM_MSG = (
    "You are a text-to-SQL generator for a SQLite database. Given the schema and a "
    "question, return a SINGLE SQL query that answers it. Use SQLite syntax. "
    "Return only the SQL - no explanation, no markdown."
)


def build_user_msg(context, question):
    return f"Schema and allowed values:\n{context}\n\nQuestion: {question}\n\nSQL:"


def clean_sql(raw):
    """Strip markdown fences / prose so we're left with runnable SQL."""
    if not raw:
        return ""
    text = raw.strip()
    fence = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    m = re.search(r"\b(SELECT|WITH)\b", text, re.IGNORECASE)
    if m:
        text = text[m.start():]
    return text.strip().strip("`").rstrip(";").strip()


def run_sql(conn, sql):
    """Run SQL and return a DataFrame, or None if it errored."""
    try:
        return pd.read_sql_query(sql, conn)
    except Exception:
        return None

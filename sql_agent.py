"""
sql_agent.py - the DATA engine. Turns a natural-language question into an answer by:
  1. generating SQL (same prompt/cleaning we evaluated in Phase 2),
  2. running it on telco.db,
  3. asking the model to phrase the RESULT in one sentence.

Anti-hallucination rule: the NUMBERS come from the database, never from the model.
The model only writes the SQL and phrases the row we hand back to it - it is never
asked to "know" a figure. That separation is what makes the answer trustworthy.

`gen(system, user) -> str` is injected, so the agent uses whichever model won the
Phase 2 leaderboard (or a mock in tests).
"""

import os
import sqlite3

from context import build_context
from sql_common import SYSTEM_MSG, build_user_msg, clean_sql, run_sql

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "telco.db")

_CONTEXT = None
def _context():
    global _CONTEXT
    if _CONTEXT is None:
        _CONTEXT = build_context()
    return _CONTEXT


PHRASE_SYSTEM = (
    "You are HexaMobile's data assistant. You are given a question and the SQL result that "
    "answers it. Reply in a natural, friendly tone of two or three sentences: briefly restate "
    "what was asked, give the answer, and close politely. Use ONLY the values in the result for "
    "any numbers or names - never invent or estimate figures, and do not add facts that are not "
    "in the result. If the result is empty, say no matching data was found."
)


def answer(gen, question):
    """Return {answer, sql, table}. `table` is the DataFrame (ground truth)."""
    sql = clean_sql(gen(SYSTEM_MSG, build_user_msg(_context(), question)))

    conn = sqlite3.connect(DB_PATH)
    df = run_sql(conn, sql)
    conn.close()

    if df is None:
        return {"answer": "Sorry, I couldn't build a valid query for that question.",
                "sql": sql, "table": None}

    result_text = df.to_string(index=False) if not df.empty else "(no rows)"
    phrased = gen(PHRASE_SYSTEM,
                  f"Question: {question}\nSQL: {sql}\nResult:\n{result_text}\n\nAnswer:")
    return {"answer": (phrased or "").strip(), "sql": sql, "table": df}

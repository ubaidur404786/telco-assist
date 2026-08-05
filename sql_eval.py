"""
sql_eval.py - run every free model over the golden SQL dataset and rank them.

Flow (per model, per question):
  1. build the prompt (context + question)
  2. model generates SQL
  3. clean the SQL (strip markdown/prose)
  4. run it on telco.db
  5. compare the result to the frozen gold result (execution accuracy)
  6. tally, then print a leaderboard and save sql_eval_results.csv

Usage:
  python sql_eval.py --mock        # offline self-test: no API key needed
  python sql_eval.py               # real run: uses whatever provider keys are in .env
  python sql_eval.py --limit 8     # quick run on the first 8 questions
"""

import os
import re
import csv
import json
import argparse
import sqlite3

import pandas as pd
from dotenv import load_dotenv

import models
from context import build_context
from sql_common import SYSTEM_MSG, build_user_msg, clean_sql, run_sql

load_dotenv()

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "telco.db")
GOLDEN_CSV = os.path.join(HERE, "sql_golden.csv")
RESULTS_CSV = os.path.join(HERE, "sql_eval_results.csv")


# ---------- gold helper ----------
def gold_df(gold_result_json):
    obj = json.loads(gold_result_json)
    return pd.DataFrame(obj["rows"], columns=obj["columns"])


# ---------- model runners (real or mock) ----------
def build_runners(mock, golden):
    """Return a list of (name, provider, generate_fn) where generate_fn(system, user)->sql."""
    if mock:
        # Offline self-test doubles that need no API key:
        q2sql = {row["question"]: row["gold_sql"] for _, row in golden.iterrows()}

        def oracle(system, user):                 # returns the gold SQL -> ~100%
            for q, sql in q2sql.items():
                if q in user:
                    return sql
            return ""

        def naive(system, user):                  # one blunt query -> only #1 is right
            return "SELECT COUNT(*) AS n FROM customers"

        return [("OracleMock", "mock", oracle), ("NaiveMock", "mock", naive)]

    specs = models.available_models()
    return [(sp.name, sp.provider, (lambda s, u, sp=sp: models.generate(sp, s, u)))
            for sp in specs]


# ---------- the eval ----------
def run_eval(mock=False, limit=None):
    from sql_evaluator import evaluate_one

    context = build_context()
    golden = pd.read_csv(GOLDEN_CSV)
    if limit:
        golden = golden.head(limit)
    conn = sqlite3.connect(DB_PATH)

    runners = build_runners(mock, golden)
    if not runners:
        print("No models available. Add GROQ_API_KEY or OPENROUTER_API_KEY to .env, "
              "or run with --mock.")
        return

    all_rows, scoreboard = [], {}

    for name, provider, gen in runners:
        print(f"\n{'='*64}\nMODEL: {name}  ({provider})\n{'='*64}")
        by_diff = {}                              # difficulty -> [correct, total]
        rows, skipped = [], False

        for i, (_, g) in enumerate(golden.iterrows()):
            qid, question, diff = g["id"], g["question"], g["difficulty"]
            order_sensitive = str(g["order_sensitive"]).upper() == "TRUE"
            gd = gold_df(g["gold_result_json"])

            try:
                sql = clean_sql(gen(SYSTEM_MSG, build_user_msg(context, question)))
            except Exception as e:
                if i == 0:                        # fails on the first call -> dead model, skip it
                    print(f"  SKIPPED model ({str(e)[:70]})")
                    skipped = True
                    break
                print(f"  #{qid:2} [{diff:6}] GEN-ERROR: {str(e)[:70]}")
                rows.append(dict(model=name, id=qid, difficulty=diff,
                                 correct=False, reason="gen_error", sql=""))
                by_diff.setdefault(diff, [0, 0])[1] += 1
                continue

            gen_res = run_sql(conn, sql)
            verdict = evaluate_one(gd, gen_res, order_sensitive)

            c = by_diff.setdefault(diff, [0, 0])
            c[1] += 1
            if verdict["correct"]:
                c[0] += 1

            mark = "OK " if verdict["correct"] else "XX "
            print(f"  #{qid:2} [{diff:6}] {mark} {verdict['reason']}")
            rows.append(dict(model=name, id=qid, difficulty=diff,
                             correct=verdict["correct"], reason=verdict["reason"],
                             sql=" ".join(sql.split())))

        if skipped:
            continue
        all_rows.extend(rows)
        total_c = sum(v[0] for v in by_diff.values())
        total_n = sum(v[1] for v in by_diff.values())
        scoreboard[name] = (total_c, total_n, by_diff)
        print(f"\n  SCORE: {total_c}/{total_n} = {100*total_c/total_n:.1f}%")

    conn.close()

    # ---- leaderboard ----
    print(f"\n{'='*64}\nLEADERBOARD (execution accuracy)\n{'='*64}")
    ranked = sorted(scoreboard.items(), key=lambda kv: kv[1][0], reverse=True)
    print(f"  {'model':26s} {'overall':>9}   easy  medium  hard")
    for name, (c, n, by_diff) in ranked:
        def pct(d):
            cc, nn = by_diff.get(d, [0, 0])
            return f"{100*cc/nn:.0f}%" if nn else "  -"
        print(f"  {name:26s} {c:>3}/{n:<3} {100*c/n:5.1f}%  "
              f"{pct('easy'):>4}  {pct('medium'):>5}  {pct('hard'):>4}")

    if ranked:
        winner = ranked[0][0]
        print(f"\n  WINNER -> {winner}   (this is the model the SQL agent will use)")

    pd.DataFrame(all_rows).to_csv(RESULTS_CSV, index=False, quoting=csv.QUOTE_ALL)
    print(f"\nDetailed results saved -> {os.path.basename(RESULTS_CSV)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="offline self-test, no API key")
    ap.add_argument("--limit", type=int, default=None, help="only first N questions")
    args = ap.parse_args()
    run_eval(mock=args.mock, limit=args.limit)

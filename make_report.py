"""
make_report.py - turn the eval result CSVs into paste-ready Markdown leaderboards.

After you run the real evals (sql_eval.py, rag_eval.py) with an API key, this reads
their output CSVs and prints the leaderboard tables. Pass --write to also save RESULTS.md.
Copy the tables into the README's "Results" section.

  python make_report.py            # print tables
  python make_report.py --write    # also write RESULTS.md
"""

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SQL_CSV = os.path.join(HERE, "sql_eval_results.csv")
RAG_CSV = os.path.join(HERE, "rag_eval_results.csv")
OUT = os.path.join(HERE, "RESULTS.md")


def _truthy(x):
    return str(x).strip().lower() in ("true", "1", "yes")


def _read(path):
    """Read a results CSV, tolerating a stray malformed row from an older run."""
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")


def sql_report():
    if not os.path.exists(SQL_CSV):
        return "_No sql_eval_results.csv yet — run `python sql_eval.py`._\n"

    df = _read(SQL_CSV)
    if "correct" not in df.columns:
        return "_sql_eval_results.csv looks malformed — re-run `python sql_eval.py`._\n"
    df["correct"] = df["correct"].map(_truthy)

    rows = []
    for model, g in df.groupby("model"):
        overall = 100 * g["correct"].mean()

        def pct(d):
            gg = g[g["difficulty"] == d]
            return f"{100*gg['correct'].mean():.0f}%" if len(gg) else "-"

        rows.append((overall, model, pct("easy"), pct("medium"), pct("hard")))
    rows.sort(reverse=True)                       # best overall first

    out = ["### Text-to-SQL leaderboard - execution accuracy\n",
           "| Model | Overall | Easy | Medium | Hard |",
           "|---|---|---|---|---|"]
    for overall, model, e, m, h in rows:
        out.append(f"| {model} | {overall:.1f}% | {e} | {m} | {h} |")
    if rows:
        out.append(f"\n**Winner (SQL agent):** {rows[0][1]}")
    return "\n".join(out) + "\n"


def rag_report():
    if not os.path.exists(RAG_CSV):
        return "_No rag_eval_results.csv yet — run `python rag_eval.py`._\n"

    df = _read(RAG_CSV)
    if "correct" not in df.columns:
        return "_rag_eval_results.csv looks malformed — re-run `python rag_eval.py`._\n"
    df["correct"] = df["correct"].map(_truthy)

    rows = sorted(
        ((100 * g["correct"].mean(), model) for model, g in df.groupby("model")),
        reverse=True,
    )
    out = ["### RAG generation leaderboard - LLM-as-judge\n",
           "| Model | Correct |", "|---|---|"]
    for score, model in rows:
        out.append(f"| {model} | {score:.1f}% |")
    if rows:
        out.append(f"\n**Winner (RAG agent):** {rows[0][1]}")
    return "\n".join(out) + "\n"


def main():
    report = "# Results\n\n" + sql_report() + "\n" + rag_report()
    print(report)
    if "--write" in sys.argv:
        with open(OUT, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\nWrote {os.path.basename(OUT)}")


if __name__ == "__main__":
    main()

"""
rag_eval.py - evaluate the RAG engine on two axes and pick the best free model.

RAG has TWO failure modes, so we measure two things separately:

  1. RETRIEVAL  (model-independent) - did we fetch the right document?
     Metric: hit-rate@k (was the gold source among the top-k chunks?) and MRR
     (how highly was it ranked?). This tests the embeddings + vector store, and
     runs fully OFFLINE (no API key needed).

  2. GENERATION (per model) - given good context, did the model write a correct,
     faithful answer? Metric: LLM-as-judge. A strong model grades each answer
     against the reference and returns correct / not-correct. The out-of-scope
     question checks that the model REFUSES instead of hallucinating.

Usage:
  python rag_eval.py --mock      # offline self-test of the whole pipeline
  python rag_eval.py             # retrieval eval + real generation leaderboard (needs a key)
  python rag_eval.py --k 5       # retrieve 5 chunks instead of 3
"""

import os
import re
import csv
import json
import argparse

import pandas as pd
from dotenv import load_dotenv

import models
from rag_answer import answer
from rag_golden import GOLDEN

load_dotenv()

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(HERE, "rag_eval_results.csv")


# =========================================================================
# 1. RETRIEVAL EVAL  (offline)
# =========================================================================
def retrieval_eval(k=3):
    from rag_common import retrieve

    answerable = [g for g in GOLDEN if g["source"]]
    print(f"\n{'='*64}\nRETRIEVAL EVAL  (hit-rate@{k}, over {len(answerable)} answerable questions)\n{'='*64}")

    hits_at_k, rr_sum = 0, 0.0
    for g in answerable:
        sources = [h["source"] for h in retrieve(g["q"], k=k)]
        if g["source"] in sources:
            hits_at_k += 1
            rank = sources.index(g["source"]) + 1
            rr_sum += 1.0 / rank
            mark = f"OK  (rank {rank})"
        else:
            mark = "MISS"
        print(f"  #{g['id']:2}  {mark:12s} want={g['source']:28s} got={sources}")

    n = len(answerable)
    print(f"\n  hit-rate@{k}: {hits_at_k}/{n} = {100*hits_at_k/n:.1f}%   |   MRR: {rr_sum/n:.3f}")


# =========================================================================
# 2. GENERATION EVAL  (per model, LLM-as-judge)
# =========================================================================
JUDGE_SYSTEM = (
    "You are a strict grader. Given a QUESTION, a REFERENCE answer, and a CANDIDATE "
    "answer, decide whether the candidate is factually correct and consistent with the "
    "reference. Ignore differences in wording or format. If the reference says the "
    "assistant should refuse, then a refusal is correct. Respond with ONLY a JSON "
    'object: {"correct": true or false, "reason": "short reason"}.'
)


def parse_verdict(raw):
    m = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if m:
        try:
            return bool(json.loads(m.group(0)).get("correct", False))
        except Exception:
            pass
    return "true" in (raw or "").lower() and "false" not in (raw or "").lower()


def llm_judge(judge_spec, question, reference, candidate):
    user = f"QUESTION: {question}\nREFERENCE: {reference}\nCANDIDATE: {candidate}\n\nJSON:"
    raw = models.generate(judge_spec, JUDGE_SYSTEM, user, temperature=0, max_tokens=150)
    return parse_verdict(raw)


# ---- mock doubles for the offline self-test ----
def mock_judge(question, reference, candidate):
    """Heuristic grader: enough word overlap with the reference => correct."""
    words = lambda s: set(re.findall(r"[a-z0-9]+", s.lower()))
    ref, cand = words(reference), words(candidate)
    if not ref:
        return False
    return len(ref & cand) / len(ref) >= 0.4


def build_runners_and_judge(mock, golden):
    if mock:
        q2ref = {g["q"]: g["reference"] for g in golden}

        def oracle(system, user):                       # echoes the reference -> high score
            for q, ref in q2ref.items():
                if q in user:
                    return ref
            return ""

        def naive(system, user):                        # always refuses -> ~0 under the crude
            return "I don't have that information. Please contact HexaMobile support."

        return [("OracleMock", oracle), ("NaiveMock", naive)], mock_judge, "heuristic (mock)"

    specs = models.available_models()
    if not specs:
        return [], None, None
    # Judge = the first available model (ideally a large one). Note: it may also be
    # a candidate; we accept that small self-bias for a free portfolio setup.
    judge_spec = specs[0]
    runners = [(sp.name, (lambda s, u, sp=sp: models.generate(sp, s, u))) for sp in specs]
    judge_fn = lambda q, r, c: llm_judge(judge_spec, q, r, c)
    return runners, judge_fn, judge_spec.name


def generation_eval(mock=False, limit=None, k=3):
    golden = GOLDEN[:limit] if limit else GOLDEN
    runners, judge_fn, judge_name = build_runners_and_judge(mock, golden)

    if not runners:
        print("\nNo models available for generation eval. Add a key to .env or use --mock.")
        return

    print(f"\n{'='*64}\nGENERATION EVAL  (LLM-as-judge: {judge_name})\n{'='*64}")
    all_rows, scoreboard = [], {}

    for name, gen in runners:
        print(f"\nMODEL: {name}")
        correct, rows = 0, []
        try:
            for g in golden:
                res = answer(gen, g["q"], k=k)
                ok = judge_fn(g["q"], g["reference"], res["answer"])
                correct += int(ok)
                cites = ",".join(sorted({h["source"] for h in res["hits"]}))
                print(f"  #{g['id']:2}  {'OK ' if ok else 'XX '}  [{cites}]")
                clean = res["answer"][:300].replace("\n", " ").replace("\r", " ")
                rows.append(dict(model=name, id=g["id"], correct=ok, answer=clean))
        except Exception as e:                    # dead/unavailable model -> skip, don't crash
            print(f"  SKIPPED ({str(e)[:80]})")
            continue
        all_rows.extend(rows)
        scoreboard[name] = correct
        print(f"  SCORE: {correct}/{len(golden)} = {100*correct/len(golden):.1f}%")

    # ---- leaderboard ----
    print(f"\n{'='*64}\nRAG GENERATION LEADERBOARD\n{'='*64}")
    ranked = sorted(scoreboard.items(), key=lambda kv: kv[1], reverse=True)
    for name, c in ranked:
        print(f"  {name:26s} {c:>2}/{len(golden)} = {100*c/len(golden):5.1f}%")
    if ranked:
        print(f"\n  WINNER -> {ranked[0][0]}   (this is the model the RAG agent will use)")

    pd.DataFrame(all_rows).to_csv(RESULTS_CSV, index=False, quoting=csv.QUOTE_ALL)
    print(f"\nDetailed results saved -> {os.path.basename(RESULTS_CSV)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="offline self-test, no API key")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--k", type=int, default=3, help="chunks to retrieve")
    args = ap.parse_args()

    retrieval_eval(k=args.k)                    # always runs (offline)
    generation_eval(mock=args.mock, limit=args.limit, k=args.k)

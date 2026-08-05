"""
assistant.py - the whole bot in one entry point: ask(question).

  question -> router -> (data) SQL agent   -> answer from telco.db
                     -> (knowledge) RAG agent -> answer from the docs

Model wiring: after running the Phase 2 & 3 leaderboards you pick the winning free
model and pin it here via the ASSISTANT_MODEL env var (its name from models.MODELS).
Otherwise we use the first available model (whichever provider key is in .env).

CLI:
  python assistant.py "How many customers churned in Ile-de-France?"
  python assistant.py "How do I cancel my contract?"

With no API key it still runs an OFFLINE PREVIEW: it shows the routing decision and,
for knowledge questions, the documents retrieval would ground the answer on.
"""

import os
import sys

from dotenv import load_dotenv

import models
from router import route, keyword_route
import sql_agent
import rag_agent

load_dotenv()


def make_generator():
    """Return (spec, gen_fn) for the chosen model, or (None, None) if no key is set."""
    specs = models.available_models()
    if not specs:
        return None, None
    preferred = os.getenv("ASSISTANT_MODEL")           # pin the leaderboard winner here
    spec = next((s for s in specs if s.name == preferred), specs[0])
    return spec, (lambda system, user: models.generate(spec, system, user))


def ask(question, gen):
    """Route the question and answer it. Requires a generator (gen)."""
    kind = route(question, gen=gen)
    if kind == "data":
        res = sql_agent.answer(gen, question)
        return {"kind": "data", "answer": res["answer"],
                "detail": {"sql": res["sql"], "table": res["table"]}}
    res = rag_agent.answer(gen, question)
    return {"kind": "knowledge", "answer": res["answer"], "detail": {"sources": res["sources"]}}


def _offline_preview(question):
    """No API key: show what the bot WOULD do (routing is free; retrieval is free)."""
    label, conf, d, k = keyword_route(question)
    print(f"  route: {label}  (data-score={d}, knowledge-score={k}, confidence={conf:.2f})")
    if label == "knowledge":
        from rag_common import retrieve
        print("  would ground the answer on:")
        for h in retrieve(question, k=3):
            print(f"    [{h['distance']:.3f}] {h['source']}  ({h['section']})")
    else:
        print("  would generate SQL and run it on telco.db.")
    print("\n  (add a key to .env for the full spoken answer)")


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]).strip() or "How many customers are active?"
    print(f"\nQ: {question}")

    spec, gen = make_generator()
    if gen is None:
        _offline_preview(question)
    else:
        print(f"  (model: {spec.name})")
        result = ask(question, gen)
        print(f"  route: {result['kind']}")
        print(f"\nA: {result['answer']}")
        if result["kind"] == "data":
            print(f"\n  sql: {result['detail']['sql']}")
        else:
            print(f"\n  sources: {result['detail']['sources']}")

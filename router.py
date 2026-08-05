"""
router.py - decide whether a question needs the DATA engine (text-to-SQL over
telco.db) or the KNOWLEDGE engine (RAG over the docs).

Two strategies:
  - keyword_route(): fast, free, offline. Scores the question against two word lists
    (analytics/database terms vs policy/how-to terms) and picks the higher score.
  - llm_route(): asks a model to classify. More robust on tricky phrasings, but costs
    a call. Used as a fallback when the keyword score is a tie/low-confidence.

route() combines them: trust a confident keyword decision, otherwise ask the LLM
(if a generator is provided). Run this file to eval the keyword router offline.
"""

import re

# Words that signal a question about the DATABASE (aggregates, entities, metrics).
DATA_WORDS = {
    "how many", "count", "number of", "average", "avg", "total", "sum", "most",
    "least", "highest", "lowest", "top", "per ", "each", "revenue", "churn",
    "churned", "percentage", "percent", "rate", "list", "customers", "customer",
    "bill", "bills", "ticket", "tickets", "region", "regions", "usage", "plan ",
    "plans", "subscribers", "active", "how much revenue", "which region",
}

# Words that signal a POLICY / HOW-TO question answered by the docs.
KNOWLEDGE_WORDS = {
    "how do i", "how can i", "how to", "cancel", "activate", "activation", "esim",
    "sim", "roaming", "roam", "apn", "signal", "no signal", "coverage", "outage",
    "refund", "policy", "charge me", "charged", "fee", "fees", "pass", "travel",
    "switch", "port", "number portability", "rio", "what happens if", "can i",
    "do you offer", "is there", "what is the apn", "reset",
}


def _score(question, words):
    q = f" {question.lower()} "
    return sum(1 for w in words if w in q)


def keyword_route(question):
    """Return (label, confidence, data_score, knowledge_score)."""
    d, k = _score(question, DATA_WORDS), _score(question, KNOWLEDGE_WORDS)
    if d == k:
        return "knowledge", 0.0, d, k          # tie -> default knowledge, low confidence
    label = "data" if d > k else "knowledge"
    confidence = abs(d - k) / (d + k)          # 0..1 separation
    return label, confidence, d, k


ROUTER_SYSTEM = (
    "Classify the user's question about a mobile operator into exactly one word: "
    "'data' if it asks for numbers/statistics from the customer database (counts, "
    "averages, totals, rankings, churn, revenue), or 'knowledge' if it asks about "
    "policies or how-to help (roaming, cancelling, activation, billing rules). "
    "Answer with only the single word: data or knowledge."
)


def llm_route(question, gen):
    out = (gen(ROUTER_SYSTEM, f"Question: {question}\nAnswer:") or "").strip().lower()
    return "data" if "data" in out else "knowledge"


def route(question, gen=None, min_confidence=0.34):
    """Confident keyword decision wins; otherwise defer to the LLM if we have one."""
    label, conf, d, k = keyword_route(question)
    if gen is not None and conf < min_confidence:
        return llm_route(question, gen)
    return label


# ------------------------- offline self-eval -------------------------
GOLDEN_ROUTES = [
    ("How many customers are active?", "data"),
    ("Which region has the most churn?", "data"),
    ("What is the average bill on the Unlimited Max plan?", "data"),
    ("How much revenue did the Smart 50GB plan generate?", "data"),
    ("List the top 3 regions by number of customers.", "data"),
    ("What percentage of customers have churned?", "data"),
    ("How many tickets are still open?", "data"),
    ("What is the total number of subscribers?", "data"),
    ("How do I cancel my contract?", "knowledge"),
    ("What does roaming cost outside the EU?", "knowledge"),
    ("How do I activate my eSIM?", "knowledge"),
    ("My phone has no signal, what should I do?", "knowledge"),
    ("What is the APN for mobile data?", "knowledge"),
    ("Is there a fee for cancelling early?", "knowledge"),
    ("How do I keep my number when switching operator?", "knowledge"),
    ("What travel passes do you offer?", "knowledge"),
]

if __name__ == "__main__":
    correct = 0
    print("Keyword router (offline) vs gold labels:\n")
    for q, gold in GOLDEN_ROUTES:
        pred, conf, d, k = keyword_route(q)
        ok = pred == gold
        correct += ok
        print(f"  {'OK ' if ok else 'XX '} pred={pred:9s} gold={gold:9s} (d={d} k={k})  {q}")
    n = len(GOLDEN_ROUTES)
    print(f"\n  accuracy: {correct}/{n} = {100*correct/n:.1f}%")

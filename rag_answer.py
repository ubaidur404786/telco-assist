"""
rag_answer.py - the RAG chain: retrieve -> build a grounded prompt -> generate.

This is the core of the "knowledge question" engine. Two rules make it trustworthy:
  1. The model must answer ONLY from the retrieved context.
  2. If the answer isn't in the context, it must say it doesn't know (no guessing).
This is what keeps a support bot from inventing a policy that doesn't exist.

The generator is passed in as a function gen(system, user) -> str, so the SAME chain
works with any model - or with a mock in the eval. rag_eval.py and (later) the
chatbot both call answer().
"""

from rag_common import retrieve

RAG_SYSTEM = (
    "You are HexaMobile's support assistant. Answer the customer's question using ONLY the "
    "context provided. Reply in a natural, helpful tone of a few sentences: open briefly, give "
    "the answer with the relevant detail from the context, and close politely. Do not add facts "
    "that are not in the context. If the context does not contain the answer, say you don't have "
    "that information and suggest contacting support - do not make anything up. Cite the source "
    "file(s) you used in square brackets, e.g. [roaming.md]."
)


def build_prompt(question, hits):
    """Assemble the retrieved chunks into a context block + the question."""
    context = "\n\n---\n\n".join(
        f"[source: {h['source']}]\n{h['text']}" for h in hits
    )
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"


def answer(gen, question, k=3):
    """Run the full chain. Returns {answer, hits} so callers can show citations."""
    hits = retrieve(question, k=k)
    user = build_prompt(question, hits)
    text = gen(RAG_SYSTEM, user)
    return {"answer": (text or "").strip(), "hits": hits}

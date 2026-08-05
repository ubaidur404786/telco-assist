"""
rag_agent.py - the KNOWLEDGE engine. A thin wrapper over the RAG chain from Phase 3
(rag_answer.answer) that also surfaces the list of cited source documents, so the UI
can show "answered from: roaming.md".

Same injected-generator pattern as the SQL agent: pass in whichever model won the
Phase 3 leaderboard.
"""

from rag_answer import answer as _rag_answer


def answer(gen, question, k=3):
    """Return {answer, sources, hits}."""
    res = _rag_answer(gen, question, k=k)
    sources = sorted({h["source"] for h in res["hits"]})
    return {"answer": res["answer"], "sources": sources, "hits": res["hits"]}

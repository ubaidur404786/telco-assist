"""
rag_common.py - the shared RAG plumbing: embeddings + the Chroma vector store.

Design choices (and why):
  - EMBEDDINGS via fastembed (ONNX runtime), not sentence-transformers. fastembed
    needs no PyTorch (~2 GB), just a ~130 MB ONNX model - a much better fit for a
    laptop and faster to cold-start. Model: BAAI/bge-small-en-v1.5, a strong small
    English retriever (384-dim vectors).
  - VECTOR STORE via Chroma, persisted to ./chroma_db so we embed once and reuse.
    We use cosine distance (the right metric for normalised text embeddings).

Everything else (ingest, answer, eval) imports retrieve() / embed() from here.
"""

import os

import chromadb
from fastembed import TextEmbedding

HERE = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(HERE, "chroma_db")
COLLECTION_NAME = "hexamobile_kb"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# ---- embeddings (model loaded once, lazily) ----
_embedder = None

def embedder():
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(EMBED_MODEL)
    return _embedder


def embed(texts):
    """Embed a list of strings -> list of plain python float lists (Chroma wants lists)."""
    return [v.tolist() for v in embedder().embed(list(texts))]


# ---- vector store ----
def client():
    return chromadb.PersistentClient(path=CHROMA_DIR)


def fresh_collection():
    """Drop and recreate the collection - used by the ingest step."""
    c = client()
    try:
        c.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    # cosine space: 1 - cosine_similarity, so smaller distance = more similar
    return c.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def get_collection():
    return client().get_collection(COLLECTION_NAME)


def retrieve(question, k=3):
    """Return the top-k chunks for a question as a list of dicts."""
    qv = embed([question])[0]
    res = get_collection().query(
        query_embeddings=[qv],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append({
            "text": doc,
            "source": meta.get("source", ""),
            "section": meta.get("section", ""),
            "distance": dist,
        })
    return hits

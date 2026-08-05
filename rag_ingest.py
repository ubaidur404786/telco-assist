"""
rag_ingest.py - turn the knowledge base (data/docs/*.md) into a searchable index.

Pipeline:
  read .md files  ->  chunk by section  ->  embed each chunk  ->  store in Chroma

Why chunk by section?
  A retriever works best when each chunk is a single coherent idea. Our docs are
  already written as "## Section" blocks (roaming in the EU, roaming outside the EU,
  travel passes, ...), so splitting on H2 headings gives natural, self-contained
  chunks. We prepend "DocTitle > Section" to each chunk so the embedding carries
  that context, and we keep the source filename in metadata for citations + eval.

Run (after installing chromadb + fastembed):
  python rag_ingest.py
"""

import os
import glob

from rag_common import fresh_collection, embed

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(HERE, "data", "docs")
MAX_CHARS = 1200          # if a section is longer, split it into paragraph windows


def split_sections(md_text):
    """Split a markdown doc into (doc_title, [(section_heading, section_body), ...])."""
    lines = md_text.splitlines()
    doc_title = "Untitled"
    sections = []
    heading, body = None, []

    def flush():
        if heading is not None and body:
            sections.append((heading, "\n".join(body).strip()))

    for line in lines:
        if line.startswith("# ") and doc_title == "Untitled":
            doc_title = line[2:].strip()
        elif line.startswith("## "):
            flush()
            heading, body = line[3:].strip(), []
        else:
            if heading is None:
                heading = doc_title        # text before the first H2 belongs to the intro
            body.append(line)
    flush()
    return doc_title, sections


def window(text, size):
    """Split an over-long section into ~size-char paragraph windows (keeps paragraphs whole)."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) > size:
            chunks.append(cur.strip())
            cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


def build_chunks():
    """Read every doc and return (ids, documents, metadatas)."""
    ids, docs, metas = [], [], []
    for path in sorted(glob.glob(os.path.join(DOCS_DIR, "*.md"))):
        source = os.path.basename(path)
        with open(path, encoding="utf-8") as f:
            title, sections = split_sections(f.read())

        for s_idx, (heading, body) in enumerate(sections):
            for w_idx, piece in enumerate(window(body, MAX_CHARS)):
                # Prepend context so the embedding "knows" which doc/section it is.
                chunk_text = f"{title} > {heading}\n\n{piece}"
                ids.append(f"{source}::{s_idx}::{w_idx}")
                docs.append(chunk_text)
                metas.append({"source": source, "section": heading, "title": title})
    return ids, docs, metas


def main():
    ids, docs, metas = build_chunks()
    print(f"Chunked {len(set(m['source'] for m in metas))} docs into {len(docs)} chunks")

    print("Embedding chunks (first run downloads the model)...")
    vectors = embed(docs)

    col = fresh_collection()
    col.add(ids=ids, documents=docs, metadatas=metas, embeddings=vectors)
    print(f"Indexed {col.count()} chunks into Chroma collection '{col.name}'")

    # quick retrieval smoke test
    from rag_common import retrieve
    hits = retrieve("How much does roaming cost outside the EU?", k=3)
    print("\nSmoke test -> 'roaming cost outside the EU?':")
    for h in hits:
        print(f"  [{h['distance']:.3f}] {h['source']:28s} {h['section']}")


if __name__ == "__main__":
    main()

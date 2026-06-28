"""
corpus_loader.py
----------------
Loads the synthetic guideline corpus and splits each guideline into overlapping
chunks. Every chunk keeps the PARENT document id so that retrieval evaluation
can map a chunk hit back to the guideline it came from.

We chunk by characters with overlap. In the lab we explain WHY:
  - too-small chunks lose context (a sentence with no surrounding meaning)
  - too-large chunks dilute the embedding (many topics in one vector)
  - overlap stops us cutting an idea in half at a boundary
"""
import json
import os
from typing import List, Dict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))


def load_documents() -> List[Dict]:
    docs = []
    with open(os.path.join(DATA, "corpus.jsonl"), encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    return docs


def simple_char_chunks(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    """A dependency-free char splitter (mirrors RecursiveCharacterTextSplitter idea)."""
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]
    chunks, start = [], 0
    step = max(1, chunk_size - overlap)
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += step
    return chunks


def build_chunk_records(chunk_size: int = 900, overlap: int = 150) -> List[Dict]:
    """
    Returns a flat list of chunk records, each with:
      chunk_id, parent_id, title, pathway_code, version, year, status,
      specialty, citation, source_uri, content
    """
    out = []
    for d in load_documents():
        # Chunk each section so that headings stay near their text.
        for s_idx, section in enumerate(d["sections"]):
            section_text = f"{d['pathway_title']} — {section['heading']}\n{section['text']}"
            for c_idx, chunk in enumerate(simple_char_chunks(section_text, chunk_size, overlap)):
                out.append({
                    "chunk_id": f"{d['id']}__s{s_idx}__c{c_idx}",
                    "parent_id": d["id"],
                    "title": d["title"],
                    "pathway_code": d["pathway_code"],
                    "section": section["heading"],
                    "version": d["version"],
                    "year": d["year"],
                    "status": d["status"],
                    "specialty": d["specialty"],
                    "citation": d["citation"],
                    "source_uri": d["source_uri"],
                    "content": chunk,
                })
    return out


if __name__ == "__main__":
    chunks = build_chunk_records()
    docs = load_documents()
    print(f"{len(docs)} documents -> {len(chunks)} chunks")
    print("Example chunk:", json.dumps(chunks[0], indent=2)[:400])

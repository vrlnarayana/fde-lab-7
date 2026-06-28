"""
01_build_index.py  (LAB 7-A)
----------------------------
Chunks the guideline corpus, creates embeddings with Azure OpenAI, and pushes
everything into an Azure AI Search index that supports keyword + vector +
semantic ranking.

Prerequisites:
  - .env filled in (see .env.example)
  - pip install -r requirements.txt
  - data/corpus.jsonl generated (run generate_corpus.py if missing)

Run:
    python 01_build_index.py
"""
import json
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from corpus_loader import load_documents
from azure_clients import get_vector_store


def main():
    docs = load_documents()
    print(f"Loaded {len(docs)} guideline documents from corpus.jsonl")

    # 1) CHUNKING.
    # We split per section so a heading stays attached to its text.
    # chunk_size/overlap are in CHARACTERS here for simplicity.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    lc_docs = []
    for d in docs:
        for s_idx, section in enumerate(d["sections"]):
            text = f"{d['pathway_title']} — {section['heading']}\n{section['text']}"
            for c_idx, chunk in enumerate(splitter.split_text(text)):
                # Metadata travels WITH the chunk. parent_id lets us trace a hit
                # back to the guideline (needed for citations and evaluation).
                meta = {
                    "parent_id": d["id"],
                    "title": d["title"],
                    "pathway_code": d["pathway_code"],
                    "section": section["heading"],
                    "version": d["version"],
                    "year": d["year"],
                    "status": d["status"],          # CURRENT or SUPERSEDED
                    "specialty": d["specialty"],
                    "citation": d["citation"],
                    "source_uri": d["source_uri"],
                }
                lc_docs.append(Document(page_content=chunk, metadata=meta))

    print(f"Created {len(lc_docs)} chunks. Embedding + uploading to Azure AI Search...")

    # 2) BUILD THE INDEX.
    # AzureSearch creates the index (vector + semantic config) on first use,
    # then embeds and uploads each chunk.
    store = get_vector_store(search_type="hybrid")
    store.add_documents(documents=lc_docs)

    print("Done. Index is ready.")
    print("Tip: open the Azure portal -> your Search service -> Indexes to see it,")
    print("and use Search Explorer to run a test query.")


if __name__ == "__main__":
    main()

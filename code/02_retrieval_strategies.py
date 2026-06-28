"""
02_retrieval_strategies.py  (LAB 7-B)
-------------------------------------
Shows the THREE retrieval strategies we compare in the lab, all on the same
Azure AI Search index:

  naive   = pure vector search        (search_type="similarity")
  hybrid  = BM25 keyword + vector,    (search_type="hybrid")  fused with RRF
  reranked= hybrid + semantic ranker  (search_type="semantic_hybrid")

Run a single query through all three:
    python 02_retrieval_strategies.py "How did sepsis screening change since 2014?"
"""
import sys
from azure_clients import get_vector_store

STRATEGIES = {
    "naive":    "similarity",
    "hybrid":   "hybrid",
    "reranked": "semantic_hybrid",
}


def retrieve(query: str, strategy: str, k: int = 5):
    """Return a list of LangChain Documents for the chosen strategy."""
    store = get_vector_store(search_type=STRATEGIES[strategy])
    return store.similarity_search(query, k=k)


def parent_ids(docs):
    """Collapse chunk hits into an ordered, de-duplicated list of guideline ids."""
    seen, ordered = set(), []
    for d in docs:
        pid = d.metadata.get("parent_id")
        if pid and pid not in seen:
            seen.add(pid)
            ordered.append(pid)
    return ordered


def demo(query: str, k: int = 5):
    print(f"\nQUERY: {query}\n" + "=" * 70)
    for name in STRATEGIES:
        docs = retrieve(query, name, k)
        print(f"\n[{name.upper()}] top guidelines:")
        for i, pid in enumerate(parent_ids(docs), 1):
            # find the citation from the first chunk of that parent
            cite = next((d.metadata.get("citation") for d in docs
                         if d.metadata.get("parent_id") == pid), pid)
            print(f"  {i}. {pid}  —  {cite}")


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "How did sepsis screening change since 2014?"
    demo(q)

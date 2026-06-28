"""
03_rag_pipeline.py  (LAB 7-B / capstone)
----------------------------------------
End-to-end RAG that ALWAYS cites the source guideline. This is the deliverable
the clinicians would actually use: ask a multi-hop clinical question, get an
answer with [citation] markers pointing at the guideline + version + year.

Run:
    python 03_rag_pipeline.py "A septic pregnant patient needs antibiotics. What guides the choice?"
"""
import sys
from langchain.prompts import ChatPromptTemplate
from azure_clients import get_vector_store, get_chat

SYSTEM = (
    "You are a clinical guidelines assistant for Meridian Health Network. "
    "Answer ONLY from the provided guideline excerpts. "
    "If the answer is not in the excerpts, say you cannot find it in the guidelines. "
    "Every clinical statement MUST end with a citation marker like "
    "[SEP-v3_0] that matches the excerpt you used. "
    "Prefer CURRENT guidelines over SUPERSEDED ones, but you may cite an older "
    "version when the question is about what changed over time."
)

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", "Question:\n{question}\n\nGuideline excerpts:\n{context}\n\nAnswer with citations:"),
])


def format_context(docs):
    blocks = []
    for d in docs:
        m = d.metadata
        blocks.append(
            f"[{m.get('parent_id')}] ({m.get('status')}, {m.get('year')}) "
            f"{m.get('title')}\nSection: {m.get('section')}\n{d.page_content}"
        )
    return "\n\n---\n\n".join(blocks)


def answer(question: str, strategy: str = "semantic_hybrid", k: int = 6):
    store = get_vector_store(search_type=strategy)
    docs = store.similarity_search(question, k=k)
    context = format_context(docs)

    chat = get_chat(temperature=0.0)
    chain = PROMPT | chat
    resp = chain.invoke({"question": question, "context": context})

    # Build the source list (unique parents in retrieval order)
    seen, sources = set(), []
    for d in docs:
        pid = d.metadata.get("parent_id")
        if pid and pid not in seen:
            seen.add(pid)
            sources.append(f"{pid} — {d.metadata.get('citation')}")
    return resp.content, sources


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or \
        "A septic pregnant patient needs antibiotics. What guides the choice?"
    ans, sources = answer(q)
    print("QUESTION:", q)
    print("\nANSWER:\n", ans)
    print("\nSOURCES:")
    for s in sources:
        print("  -", s)

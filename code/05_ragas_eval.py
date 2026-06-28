"""
05_ragas_eval.py  (LAB 7-C, answer-quality layer)
-------------------------------------------------
Retrieval metrics (recall/MRR/nDCG) tell you if the RIGHT documents were found.
Ragas tells you if the GENERATED ANSWER is good:
  - faithfulness        : is the answer supported by the retrieved context?
                          (low score = hallucination)
  - answer_relevancy    : does the answer actually address the question?
  - context_precision   : are the retrieved chunks relevant (not noise)?

We run this on a small sample so it stays cheap. It uses your Azure OpenAI
deployments as both the generator and the Ragas judge.

Run:
    python 05_ragas_eval.py --n 10 --strategy reranked
"""
import argparse, json, os

from azure_clients import get_vector_store, get_chat, get_embeddings
from langchain.prompts import ChatPromptTemplate

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD = os.path.abspath(os.path.join(HERE, "..", "golden_set", "golden_set.json"))

SYSTEM = ("You are a clinical guidelines assistant. Answer ONLY from the excerpts. "
          "If not present, say you cannot find it. Cite guideline ids in [brackets].")
PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", "Question:\n{q}\n\nExcerpts:\n{c}\n\nAnswer:"),
])
STRATEGIES = {"naive": "similarity", "hybrid": "hybrid", "reranked": "semantic_hybrid"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--strategy", default="reranked", choices=list(STRATEGIES))
    args = ap.parse_args()

    with open(GOLD, encoding="utf-8") as f:
        gold = [q for q in json.load(f) if q["category"] != "negative"][: args.n]

    store = get_vector_store(search_type=STRATEGIES[args.strategy])
    chat = get_chat(0.0)
    chain = PROMPT | chat

    questions, answers, contexts, references = [], [], [], []
    for q in gold:
        docs = store.similarity_search(q["question"], k=6)
        ctx = [d.page_content for d in docs]
        ans = chain.invoke({"q": q["question"], "c": "\n\n".join(ctx)}).content
        questions.append(q["question"])
        answers.append(ans)
        contexts.append(ctx)
        references.append(q["reference_answer"])

    # --- Ragas ---
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    ds = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "reference": references,
    })

    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=LangchainLLMWrapper(chat),
        embeddings=LangchainEmbeddingsWrapper(get_embeddings()),
    )
    print(f"\nRagas ({args.strategy}, n={len(gold)}):")
    print(result)


if __name__ == "__main__":
    main()
